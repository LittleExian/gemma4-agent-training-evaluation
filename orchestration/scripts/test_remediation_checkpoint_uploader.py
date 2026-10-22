import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import types
import remediation_checkpoint_uploader as m


class UploaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root/'work/full-epoch1'
        self.output.mkdir(parents=True)
        self.recipe = {'code_commit': 'a'*40, 'data_sha256': 'b'*64,
                       'training_recipe': {'epochs': 1, 'world_size': 2, 'total_steps': 5, 'save_every': 5}}

    def checkpoint(self, step=5):
        cp = self.output/f'checkpoint-{step}'
        (cp/'adapter').mkdir(parents=True)
        (cp/'adapter/adapter_config.json').write_text('{"r":16}')
        (cp/'adapter/adapter_model.safetensors').write_bytes(b'weights')
        (cp/'COMPLETE.json').write_text(json.dumps({'step': step, 'world_size': 2}))
        (cp/'adapter_audit.json').write_text('{"passed":true}')
        return cp

    def test_only_complete_audited_checkpoints(self):
        cp = self.checkpoint()
        self.assertEqual(m.discover(self.output), [cp])
        (cp/'adapter_audit.json').write_text('{"passed":false}')
        self.assertEqual(m.discover(self.output), [])
        (cp/'adapter_audit.json').write_text('{"passed":true}')
        (cp/'COMPLETE.json').write_text('{"step":5,"world_size":3}')
        self.assertEqual(m.discover(self.output), [])

    def test_smoke_and_symlink_not_discovered(self):
        cp = self.checkpoint()
        (cp/'adapter/adapter_model.safetensors').unlink()
        (cp/'adapter/adapter_model.safetensors').symlink_to(cp/'COMPLETE.json')
        self.assertEqual(m.discover(self.output), [])

    def test_recipe_rejects_secret_and_unknown_keys(self):
        self.assertEqual(m.validate_recipe(self.recipe), self.recipe)
        for bad in ({**self.recipe, 'token': 'secret'},
                    {**self.recipe, 'training_recipe': {'sample_ids': ['x']}},
                    {**self.recipe, 'code_commit': '../private'}):
            with self.assertRaises(ValueError):
                m.validate_recipe(bad)

    def test_auth_waits_then_requires_private_owned_regular_token(self):
        home = self.root/'auth'
        home.mkdir()
        self.assertFalse(m.auth_ready(home))
        token = home/'token'
        token.write_text('secret')
        token.chmod(0o644)
        with self.assertRaises(ValueError): m.auth_ready(home)
        token.chmod(0o600)
        self.assertTrue(m.auth_ready(home))
        token.unlink()
        token.symlink_to(self.root/'elsewhere')
        with self.assertRaises(ValueError): m.auth_ready(home)

    def test_payload_has_exact_three_safe_files(self):
        cp = self.checkpoint()
        payload, hashes = m.payload(cp, self.recipe)
        self.assertEqual(set(payload), {'adapter_config.json', 'adapter_model.safetensors', 'checkpoint-info.json'})
        info = json.loads(payload['checkpoint-info.json'])
        self.assertEqual(info['step'], 5)
        self.assertNotIn(str(self.root), payload['checkpoint-info.json'].decode())
        self.assertEqual(hashes['adapter_model.safetensors'], hashlib.sha256(b'weights').hexdigest())

    def test_identity_must_match_and_repo_must_exist(self):
        api = Mock()
        api.whoami.return_value = {'name': 'other'}
        with self.assertRaises(ValueError): m.check_account(api)
        api.whoami.return_value = {'name': 'eexian'}
        m.check_account(api)
        api.repo_info.assert_called_once_with(m.REPO, repo_type='model')
        api.create_repo.assert_not_called()

    def test_verify_pins_commit_and_hashes_config(self):
        api = Mock()
        prefix = 'runs/remediation-20261022/checkpoint-5/'
        hashes = {'adapter_model.safetensors': hashlib.sha256(b'weights').hexdigest(),
                  'adapter_config.json': hashlib.sha256(b'config').hexdigest(),
                  'checkpoint-info.json': hashlib.sha256(b'info').hexdigest()}
        rows = []
        for name, digest in hashes.items():
            rows.append(type('File', (), {'path': prefix+name,
                         'lfs': {'sha256': digest} if name.endswith('safetensors') else None})())
        api.list_repo_tree.return_value = rows
        downloader = Mock(side_effect=lambda name: {'adapter_config.json': b'config', 'checkpoint-info.json': b'info'}[name])
        m.verify_remote(api, prefix, 'commit123', hashes, downloader)
        self.assertEqual(api.list_repo_tree.call_args.kwargs['revision'], 'commit123')
        rows[0].lfs = {'sha256': '0'*64}
        with self.assertRaises(ValueError): m.verify_remote(api, prefix, 'commit123', hashes, downloader)

    def test_upload_verified_before_state(self):
        cp = self.checkpoint()
        api = Mock()
        api.create_commit.return_value.oid = 'abc'
        seen = []
        def verifier(api, prefix, commit, hashes, downloader):
            seen.append((prefix, commit))
        result = m.upload(api, cp, self.recipe, lambda **kwargs: kwargs, Mock(), verifier)
        self.assertEqual(result['status'], 'uploaded')
        self.assertEqual(seen, [('runs/remediation-20261022/checkpoint-5/', 'abc')])
        self.assertEqual(len(api.create_commit.call_args.kwargs['operations']), 3)

    def test_lock_prevents_second_watcher(self):
        with m.singleton(self.root/'watch.lock'):
            with self.assertRaises(BlockingIOError):
                with m.singleton(self.root/'watch.lock'): pass

    def test_atomic_json_and_finished(self):
        path = self.root/'state.json'
        m.atomic_json(path, {'hello': True})
        self.assertEqual(json.loads(path.read_text()), {'hello': True})
        (self.output/'progress.json').write_text('{"state":"completed"}')
        self.assertTrue(m.training_finished(self.output))
        (self.output/'progress.json').write_text('{"state":"failed"}')
        self.assertFalse(m.training_finished(self.output))

    def args(self):
        recipe = self.root/'recipe.json'
        recipe.write_text(json.dumps(self.recipe))
        return types.SimpleNamespace(root=self.root, recipe_json=recipe, max_hours=1, poll_seconds=30)

    def test_run_waiting_auth_is_bounded_and_never_loads_sdk(self):
        args = self.args()
        with patch.dict(os.environ, {'HF_TOKEN': 'never-use'}), \
             patch.object(m.time, 'monotonic', side_effect=[0, 1, 4000]), \
             patch.object(m.time, 'sleep'):
            m.run(args)
        report = json.loads((self.root/'reports/hf-upload-progress.json').read_text())
        self.assertEqual(report['phase'], 'deadline_reached')
        self.assertNotIn('never-use', json.dumps(report))

    def test_run_uploads_then_restart_skips_verified_unchanged(self):
        cp = self.checkpoint()
        args = self.args()
        home = self.root/'hf_upload_auth'
        home.mkdir()
        (home/'token').write_text('private')
        (home/'token').chmod(0o600)
        (self.output/'progress.json').write_text('{"state":"completed"}')
        api = Mock()
        api.whoami.return_value = {'name': 'eexian'}
        fake = types.SimpleNamespace(HfApi=Mock(return_value=api), CommitOperationAdd=Mock(), hf_hub_download=Mock())
        result = {'status': 'uploaded', 'commit': 'abc', 'sha256': m.payload(cp, self.recipe)[1]}
        with patch.dict('sys.modules', {'huggingface_hub': fake}), \
             patch.dict(os.environ), patch.object(m, 'upload', return_value=result) as upload:
            m.run(args)
            m.run(args)
            self.assertEqual(upload.call_count, 1)
        self.assertEqual(json.loads((self.root/'reports/hf-upload-progress.json').read_text())['phase'], 'completed')

    def test_run_upload_error_redacts_and_retries(self):
        self.checkpoint()
        args = self.args()
        fake = types.SimpleNamespace(HfApi=Mock(), CommitOperationAdd=Mock(), hf_hub_download=Mock())
        with patch.dict('sys.modules', {'huggingface_hub': fake}), patch.dict(os.environ), \
             patch.object(m, 'auth_ready', return_value=True), \
             patch.object(m, 'check_account'), \
             patch.object(m, 'upload', side_effect=RuntimeError('SECRET')), \
             patch.object(m.time, 'monotonic', side_effect=[0, 1, 4000]), patch.object(m.time, 'sleep'):
            m.run(args)
        raw = (self.root/'reports/hf-upload-progress.json').read_text()
        self.assertNotIn('SECRET', raw)
        self.assertEqual(json.loads(raw)['error_type'], 'RuntimeError')
        self.assertEqual(json.loads((self.root/'reports/checkpoint-upload-state.json').read_text())['checkpoint-5']['status'], 'uploading')

    def test_expected_checkpoint_set_includes_non_multiple_final(self):
        self.assertEqual(m.expected_names({'training_recipe': {'total_steps': 12, 'save_every': 5}}),
                         {'checkpoint-5', 'checkpoint-10', 'checkpoint-12'})
        with self.assertRaises(ValueError): m.expected_names({'training_recipe': {}})

    def test_upload_detects_changed_checkpoint(self):
        cp = self.checkpoint()
        api = Mock()
        def change(*args): (cp/'adapter/adapter_config.json').write_text('changed')
        with self.assertRaises(ValueError):
            m.upload(api, cp, self.recipe, Mock(), Mock(), change)

if __name__ == '__main__': unittest.main()
