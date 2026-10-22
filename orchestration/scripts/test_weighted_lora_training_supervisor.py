import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import weighted_lora_training_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def test_gpu_gate_maps_only_idle_pro_cards(self):
        inventory = '\n'.join(f'{i}, GPU-{i}, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 98304, 95000' for i in (0, 2, 3))
        inventory += '\n1, GPU-1, NVIDIA H100 PCIe, 81559, 80000'
        self.assertEqual(supervisor.select_gpus(inventory, 'GPU-1, 123'), ['GPU-0', 'GPU-2', 'GPU-3'])
        for bad_inventory, apps in ((inventory, 'GPU-2, 234'), (inventory.replace('95000', '90000'), ''),
                                    (inventory.replace('PRO 6000', 'PRO 5000'), '')):
            with self.assertRaises(ValueError):
                supervisor.select_gpus(bad_inventory, apps)

    def test_scope_commands_are_fresh_and_fixed(self):
        args = SimpleNamespace(root=Path('/r'), source=Path('/r/source'), data=Path('/r/data.jsonl'),
                               output=Path('/r/output'), scope='full', run_name='test')
        command = supervisor.build_command(args, 'entity')
        self.assertIn('--nproc_per_node=3', command)
        self.assertEqual(command[command.index('--steps') + 1], '0')
        self.assertEqual(command[command.index('--save-every') + 1], '50')
        self.assertNotIn('--resume', command)
        smoke = supervisor.build_command(SimpleNamespace(**{**vars(args), 'scope': 'smoke'}), 'entity')
        self.assertIn('--smoke', smoke)
        self.assertEqual(smoke[smoke.index('--save-every') + 1], '1')
        with self.assertRaises(ValueError):
            supervisor.build_command(SimpleNamespace(**{**vars(args), 'scope': 'other'}), 'entity')

    def test_credential_requires_owned_private_regular_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'key'
            path.write_text('test-key-do-not-print\n')
            path.chmod(0o600)
            self.assertEqual(supervisor.read_credential(path), 'test-key-do-not-print')
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                supervisor.read_credential(path)
            link = Path(td)/'link'
            link.symlink_to(path)
            with self.assertRaises((ValueError, OSError)):
                supervisor.read_credential(link)

    def test_secret_stays_only_in_environment(self):
        args = SimpleNamespace(hf_home=Path('/r/hf'), run_name='new-run', output=Path('/r/output'))
        env = supervisor.training_environment(args, ['GPU-0', 'GPU-2', 'GPU-3'], 'private-key', 'entity',
            {'PATH': '/bin', 'WANDB_RUN_ID': 'old-id', 'WANDB_RESUME': 'must'})
        self.assertEqual(env['WANDB_API_KEY'], 'private-key')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-0,GPU-2,GPU-3')
        self.assertNotIn('WANDB_RUN_ID', env)
        self.assertNotIn('WANDB_RESUME', env)
        self.assertEqual(env['WANDB_MODE'], 'online')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')

    def test_output_scope_and_freshness(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ('source', 'hf'):
                (root/name).mkdir()
            (root/'source/train.py').write_text('')
            (root/'data.jsonl').write_text('{}\n')
            args = SimpleNamespace(root=root, source=root/'source', data=root/'data.jsonl', hf_home=root/'hf',
                output=root/'new', credential_file=Path('/workspace/.credentials/key'), manifest=None, run_name='run', scope='smoke')
            supervisor.validate_paths(args)
            (root/'new').mkdir()
            with self.assertRaises(ValueError):
                supervisor.validate_paths(args)
            with self.assertRaises(ValueError):
                supervisor.validate_paths(SimpleNamespace(**{**vars(args), 'output': Path('/tmp/outside')}))

    def test_completion_requires_all_ranks_and_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            cp = output/'checkpoint-1'
            cp.mkdir()
            for rank in range(3):
                (output/f'metrics.rank{rank}.jsonl').write_text(json.dumps({'step': 1, 'loss': .5,
                    'gradient_norm': .2, 'adapter_delta_max': .01})+'\n')
                (cp/f'rng.rank{rank}.pt').write_bytes(b'rng')
            for path, value in ((cp/'COMPLETE.json', {'step': 1, 'world_size': 3}),
                                (cp/'adapter_audit.json', {'passed': True}),
                                (output/'base_audit.json', {'passed': True}),
                                (output/'base_audit_after.json', {'passed': True}),
                                (output/'SMOKE_PASS.json', {'passed': True, 'steps': 1, 'world_size': 3,
                                    'reload_max_abs_logit_difference': 0})):
                path.write_text(json.dumps(value))
            self.assertTrue(supervisor.verify_completion(output, 'smoke')['passed'])
            (output/'base_audit_after.json').write_text('{"passed":true,"different":true}')
            with self.assertRaises(ValueError):
                supervisor.verify_completion(output, 'smoke')
            (output/'base_audit_after.json').write_text('{"passed":true}')
            (cp/'rng.rank2.pt').unlink()
            with self.assertRaises(ValueError):
                supervisor.verify_completion(output, 'smoke')

    def test_full_recipe_rejects_neighboring_dataset_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)/'data.jsonl'
            for count in (2950, 2951, 2952):
                data.write_text('{}\n'*count + '\n')
                if count == 2951:
                    self.assertEqual(supervisor.validate_recipe_data(data, 'full'), 2951)
                else:
                    with self.assertRaises(ValueError):
                        supervisor.validate_recipe_data(data, 'full')

    def test_manifest_binds_data_and_exact_python_source_subset(self):
        actual = {'data_sha256': 'data-hash', 'source_sha256': {'train.py': 'source-hash'}}
        expected = {'training_data_sha256': 'data-hash', 'source_sha256':
                    {'train.py': 'source-hash', 'README.md': 'ignored'}}
        supervisor.verify_manifest(actual, expected)
        for changed in ({**expected, 'training_data_sha256': 'other'},
                        {**expected, 'source_sha256': {'train.py': 'other'}},
                        {**expected, 'source_sha256': {'train.py': 'source-hash', 'extra.py': 'other'}}, {}):
            with self.assertRaises(ValueError):
                supervisor.verify_manifest(actual, changed)

    def test_host_memory_gate_requires_24_gib(self):
        self.assertEqual(supervisor.validate_host_memory(24*2**30), 24*2**30)
        with self.assertRaises(ValueError):
            supervisor.validate_host_memory(24*2**30-1)

    def test_metrics_ignore_incomplete_trailing_writes_and_hide_sample_ids(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            (output/'metrics.rank0.jsonl').write_text('{"step":1,"sample_ids":["private"]}\n{"step":2')
            self.assertEqual(supervisor.last_metrics(output), {'0': {'step': 1}})

    def test_bounded_lifecycle_records_outcomes_without_secrets(self):
        for outcome in ('completed', 'child_failed', 'timeout', 'interrupt', 'manifest_drift', 'host_low'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root/'source').mkdir(); (root/'hf').mkdir(); (root/'work').mkdir()
                (root/'source/train.py').write_text('# frozen source\n')
                (root/'data.jsonl').write_text('{}\n')
                (root/'manifest.json').write_text(json.dumps({'training_data_sha256': supervisor.fingerprint(root/'data.jsonl'),
                    'source_sha256': {'train.py': supervisor.fingerprint(root/'source/train.py')}}))
                if outcome == 'manifest_drift':
                    (root/'source/train.py').write_text('# unapproved change\n')
                (root/'work/wandb-connectivity.json').write_text('{"passed":true,"entity":"tested-account"}')
                args = SimpleNamespace(root=root, source=root/'source', data=root/'data.jsonl', hf_home=root/'hf',
                    output=root/'output', credential_file=Path('/workspace/.credentials/key'),
                    manifest=root/'manifest.json', run_name='new-run', scope='smoke')
                inventory = '\n'.join(f'{i}, GPU-{i}, {supervisor.GPU_NAME}, 98304, 95000' for i in (0, 2, 3))
                process = Mock(pid=123, returncode=7 if outcome == 'child_failed' else 0)
                process.poll.side_effect = [None, 0]
                fake_psutil = Mock()
                fake_psutil.Process.return_value.create_time.return_value = 123.0
                fake_psutil.virtual_memory.return_value.available = (16 if outcome == 'host_low' else 32)*2**30
                with patch.dict('sys.modules', {'psutil': fake_psutil}), \
                        patch.object(supervisor.subprocess, 'check_output', side_effect=[inventory, '']), \
                        patch.object(supervisor.subprocess, 'Popen', return_value=process) as popen, \
                        patch.object(supervisor, 'read_credential', return_value='do-not-expose-key'), \
                        patch.object(supervisor, 'verify_completion', return_value={'passed': True}), \
                        patch.object(supervisor, 'release_owned_model', return_value={'terminated': [123]}) as cleanup, \
                        patch.object(supervisor.time, 'CLOCK_BOOTTIME', 7, create=True), \
                        patch.object(supervisor.time, 'clock_gettime', return_value=100.0), \
                        patch.object(supervisor.time, 'sleep', side_effect=KeyboardInterrupt if outcome == 'interrupt' else None), \
                        patch.dict(supervisor.TIMEOUT_SECONDS, {'smoke': -1 if outcome == 'timeout' else 7200}):
                    code = supervisor.run(args)
                exit_record = json.loads((args.output/'exit.json').read_text())
                self.assertEqual(code, 0 if outcome == 'completed' else 130 if outcome == 'interrupt' else 1)
                self.assertEqual(exit_record['state'], 'completed' if outcome == 'completed' else 'failed')
                preflight_failed = outcome in ('manifest_drift', 'host_low')
                self.assertEqual(cleanup.call_count, 0 if outcome == 'completed' or preflight_failed else 1)
                if preflight_failed:
                    popen.assert_not_called()
                else:
                    self.assertTrue(popen.call_args.kwargs['start_new_session'])
                    self.assertEqual(popen.call_args.kwargs['env']['WANDB_API_KEY'], 'do-not-expose-key')
                for artifact in args.output.iterdir():
                    self.assertNotIn('do-not-expose-key', artifact.read_text())
                if outcome != 'manifest_drift':
                    self.assertEqual((args.output/'input-manifest.json').read_text(), (root/'manifest.json').read_text())


if __name__ == '__main__':
    unittest.main()
