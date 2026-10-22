"""Safety tests for the validation-only rootless Podman manager."""

import json
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from gemmacodeagent_validation20_podman import PodmanManager, EXEC_SCRIPT, PREPARE_SCRIPT


IMAGE = ('docker.io/namanjain12/coveragepy_final:'
         '05fbe9c95352ad7536c1f8ebd53a6739714a8af9@sha256:'
         '097c8b6810a09759918456bff850ce558748f48ade329e942dc4ff6e3c5554ee')
RUNTIME = {'image': IMAGE, 'repo_cwd': '/testbed',
           'resolved_base_commit': '4651411b26a4c5b2e15cd8066b264c9022f3010a',
           'python_executable': '/testbed/.venv/bin/python'}
CID = 'a' * 64


class GitlinkAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.vendor = self.repo/'vendor/llhttp'
        self.vendor.mkdir(parents=True)
        (self.vendor/'parser.c').write_text('unchanged source')
        (self.vendor/'compiled.so').write_bytes(b'preserved compiled bytes')
        (self.vendor/'.git').write_text('gitdir: ../../.git/modules/vendor/llhttp')
        functions = [node for node in ast.parse(PREPARE_SCRIPT).body
                     if isinstance(node, ast.FunctionDef)
                     and node.name in ('source_digest', 'submodule_audit')]
        self.namespace = dict(os=os, hashlib=hashlib, json=json, repo=str(self.repo))
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<audit-functions>', 'exec'), self.namespace)

    def tearDown(self):
        self.temp.cleanup()

    def test_gitlink_directory_audited_without_flattening_or_following_symlink(self):
        (self.vendor/'external').symlink_to('/definitely-not-a-readable-host-file')
        audit = self.namespace['submodule_audit'](str(self.vendor))
        self.assertEqual(audit['kind'], 'directory')
        self.assertEqual(audit['files'], 2)
        self.assertEqual(audit['symlinks'], 1)
        before = self.namespace['source_digest'](['vendor/llhttp'], {'vendor/llhttp': 'b'*40})
        (self.vendor/'.git').unlink()
        self.assertEqual(before, self.namespace['source_digest'](
            ['vendor/llhttp'], {'vendor/llhttp': 'b'*40}))
        (self.vendor/'compiled.so').write_bytes(b'changed compiled bytes')
        self.assertNotEqual(before, self.namespace['source_digest'](
            ['vendor/llhttp'], {'vendor/llhttp': 'b'*40}))

    def test_gitlink_identity_and_absent_worktree_are_distinct(self):
        first = self.namespace['source_digest'](['vendor/llhttp'], {'vendor/llhttp': 'a'*40})
        second = self.namespace['source_digest'](['vendor/llhttp'], {'vendor/llhttp': 'b'*40})
        self.assertNotEqual(first, second)
        self.assertEqual(self.namespace['submodule_audit'](str(self.repo/'absent'))['kind'], 'absent')


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.manager = PodmanManager(RUNTIME, self.output)
        self.manager._owned[CID] = self.manager.owner

    def tearDown(self):
        self.tmp.cleanup()

    def test_reject_mutable_image_or_bad_commit(self):
        for changes in ({'image': 'alpine:latest'}, {'resolved_base_commit': 'main'},
                        {'repo_cwd': '/'}, {'repo_cwd': '/proc/x'},
                        {'repo_cwd': '/repo/../testbed'}, {'python_executable': 'python'},
                        {'forbidden_commits': ['not-a-sha']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PodmanManager(dict(RUNTIME, **changes), self.output)

    def test_pid_override_is_bounded_and_sqlglot_only(self):
        for limit in (255, 1025, True, 512.5, '1024'):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                PodmanManager(dict(RUNTIME, task_id='tobymao__sqlglot-899',
                                   container_pids_limit=limit), self.output)
        with self.assertRaises(ValueError):
            PodmanManager(dict(RUNTIME, task_id='another-task', container_pids_limit=1024), self.output)
        manager = PodmanManager(dict(RUNTIME, task_id='tobymao__sqlglot-899',
                                    container_pids_limit=1024), self.output)
        self.assertEqual(manager.pids_limit, 1024)
        self.assertEqual(self.manager.pids_limit, 256)

    def test_foreign_label_prevents_exec_copy_and_stop(self):
        with patch.object(self.manager, '_podman', return_value=json.dumps(
                [{'Id': CID, 'Config': {'Labels': {self.manager.label: 'foreign'}}}]).encode()) as cli:
            for fn in (lambda: self.manager.exec(CID, 'true'),
                       lambda: self.manager.stop(CID),
                       lambda: self.manager.copy_from(CID, '/etc/passwd', self.output/'tmp'/'x')):
                with self.assertRaises(PermissionError):
                    fn()
            self.assertTrue(all(call.args[0][0] == 'inspect' for call in cli.call_args_list))

    def test_unknown_id_never_inspected(self):
        with patch.object(self.manager, '_podman') as cli:
            with self.assertRaises(PermissionError):
                self.manager.stop('b' * 64)
            cli.assert_not_called()

    def test_start_security_flags_and_owned_receipt(self):
        self.manager._owned.clear()
        proof = {'passed': True, 'head_before': RUNTIME['resolved_base_commit'], 'history_count': 1,
                 'forbidden_objects_absent': True, 'artifacts_remaining': [], 'other_git': []}
        def fake(args, **kwargs):
            if args[0] == 'run':
                self.launch_args = args
                return (CID+'\n').encode()
            if args[0] == 'inspect':
                return json.dumps([{'Id': CID, 'Config': {'Labels': {
                    self.manager.label: self.manager.owner}}}]).encode()
            return json.dumps(proof).encode()
        with patch.object(self.manager, '_podman', side_effect=fake):
            self.assertEqual(self.manager.start(), CID)
        flags = self.launch_args
        for flag in ('--network=none', '--cpus=2', '--memory=4g', '--pids-limit=256',
                     '--cap-drop=all', '--security-opt=no-new-privileges', '--pull=never'):
            self.assertIn(flag, flags)
        self.assertFalse(any('volume' in flag or flag in ('-v', '--device') for flag in flags))
        receipt = json.loads((self.output/'owned-containers.json').read_text())
        self.assertEqual(receipt['containers'][CID], self.manager.owner)
        self.assertTrue((self.output/f'sandbox-proof-{CID}.json').exists())

    def test_copy_to_reads_only_safe_regular_file(self):
        secret = self.output/'secret'
        secret.write_text('host-private')
        link = self.output/'tmp'/'link'
        link.symlink_to(secret)
        with patch.object(self.manager, '_verify_owned'), patch.object(self.manager, '_podman') as cli:
            for source in (secret, link):
                with self.assertRaises((PermissionError, OSError)):
                    self.manager.copy_to(CID, source, '/testbed/out')
            cli.assert_not_called()

    def test_copy_from_cannot_follow_host_parent_or_final_symlink(self):
        outside = self.output/'outside'
        outside.mkdir()
        (self.output/'tmp'/'escape').symlink_to(outside, target_is_directory=True)
        (outside/'secret').write_text('unchanged')
        (self.output/'tmp'/'final').symlink_to(outside/'secret')
        with patch.object(self.manager, '_verify_owned'), patch.object(
                self.manager, '_podman', return_value=b'ZGF0YQ=='):
            for target in (self.output/'outside'/'secret', self.output/'tmp'/'escape'/'secret',
                           self.output/'tmp'/'final', self.output/'tmp'/'..'/'outside'/'secret'):
                with self.assertRaises((PermissionError, OSError, ValueError)):
                    self.manager.copy_from(CID, '/etc/passwd', target)
        self.assertEqual((outside/'secret').read_text(), 'unchanged')

    def test_copy_roundtrip_uses_exec_bytes_never_podman_cp(self):
        source = self.output/'tmp'/'source'
        source.write_bytes(b'\x00hello\xff')
        with patch.object(self.manager, '_verify_owned'), patch.object(
                self.manager, '_podman', return_value=b'AGhlbGxv/w==') as cli:
            self.manager.copy_to(CID, source, '/workspace/file')
            self.manager.copy_from(CID, '/workspace/file', self.output/'tmp'/'dest')
        self.assertEqual((self.output/'tmp'/'dest').read_bytes(), source.read_bytes())
        self.assertTrue(all(call.args[0][0] == 'exec' for call in cli.call_args_list))

    def test_outer_exec_timeout_stops_only_owned_container(self):
        with patch.object(self.manager, '_verify_owned'), patch.object(
                self.manager, '_podman', side_effect=subprocess.TimeoutExpired('podman', 1)), patch.object(
                self.manager, 'stop') as stop:
            result = self.manager.exec(CID, 'sleep 1000', timeout=1)
        self.assertTrue(result.timed_out)
        stop.assert_called_once_with(CID)

    def test_command_timeout_clamped_and_official_result(self):
        payload = {'exit_code': 0, 'stdout': 'ok', 'stderr': '', 'duration_seconds': .1,
                   'timed_out': False}
        with patch.object(self.manager, '_verify_owned'), patch.object(
                self.manager, '_podman', return_value=json.dumps(payload).encode()) as cli:
            result = self.manager.exec(CID, 'echo ok', timeout=99999)
        self.assertEqual(result.stdout, 'ok')
        submitted = json.loads(cli.call_args.kwargs['input_data'])
        self.assertEqual(submitted['timeout'], 240)

    def test_nonfinite_timeout_rejected(self):
        with patch.object(self.manager, '_verify_owned'), patch.object(self.manager, '_podman') as cli:
            for timeout in (float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    self.manager.exec(CID, 'sleep 1', timeout=timeout)
            cli.assert_not_called()

    def test_copy_outer_timeout_removes_only_owned_container(self):
        with patch.object(self.manager, '_verify_owned'), patch.object(
                self.manager, '_podman', side_effect=subprocess.TimeoutExpired('podman', 30)), patch.object(
                self.manager, 'stop') as stop:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.manager.copy_from(CID, '/testbed/a', self.output/'tmp'/'download')
        stop.assert_called_once_with(CID)

    def test_exec_wrapper_real_timeout_and_output_bound(self):
        import sys
        for command, seconds, timed in [('printf ok', 1, False), ('sleep 5', .15, True),
                                       ('yes x', 2, False)]:
            result = subprocess.run([sys.executable, '-c', EXEC_SCRIPT], input=json.dumps({
                'command': command, 'timeout': seconds, 'limit': 1024,
                'cwd': self.tmp.name}).encode(), capture_output=True, timeout=5, check=True)
            value = json.loads(result.stdout)
            self.assertEqual(value['timed_out'], timed)
            self.assertLessEqual(len(value['stdout'].encode()), 1100)

    def test_prepare_script_contains_rootfs_and_history_gates(self):
        self.assertIn('os.walk', PREPARE_SCRIPT)
        self.assertIn('cat-file', PREPARE_SCRIPT)
        self.assertIn('r2e_tests', PREPARE_SCRIPT)
        self.assertIn('rev-list', PREPARE_SCRIPT)

    def test_prepare_never_receives_extra_runtime_payload(self):
        self.manager.runtime['gold_patch'] = 'PRIVATE_PATCH_NOT_FOR_AGENT'
        self.manager._owned.clear()
        calls = []
        def fake(args, **kwargs):
            calls.append((args, kwargs))
            if args[0] == 'run':
                return CID.encode()
            if args[0] == 'inspect':
                return json.dumps([{'Id': CID, 'Config': {'Labels': {
                    self.manager.label: self.manager.owner}}}]).encode()
            return b'{"passed":true,"history_count":1}'
        with patch.object(self.manager, '_podman', side_effect=fake):
            self.manager.start()
        self.assertFalse(any(b'PRIVATE_PATCH_NOT_FOR_AGENT' in k.get('input_data', b'')
                             for _, k in calls))


@unittest.skipUnless(os.getenv('VALIDATION20_PODMAN_LIVE') == '1', 'explicit rootless live smoke only')
class LiveContainerTests(unittest.TestCase):
    def test_real_image_history_tools_timeout_and_file_safety(self):
        output = Path(os.environ['VALIDATION20_PODMAN_TEST_ROOT']) / ('live-'+os.urandom(5).hex())
        runtime = dict(RUNTIME, forbidden_commits=['05fbe9c95352ad7536c1f8ebd53a6739714a8af9'])
        manager = PodmanManager(runtime, output)
        old_tempdir = tempfile.tempdir
        tempfile.tempdir = str(output/'tmp')
        try:
            cid = manager.start()
            proof = json.loads((output/f'sandbox-proof-{cid}.json').read_text())
            self.assertTrue(proof['passed'])
            self.assertEqual(proof['other_git'], [])
            self.assertEqual(proof['artifacts_remaining'], [])
            self.assertTrue(proof['forbidden_objects_absent'])
            self.assertIn('/testbed/.git', proof['removed_git'])
            self.assertIn('/r2e_tests', proof['removed_artifacts'])
            self.assertIn(runtime['resolved_base_commit'], proof['original_objects_checked'])
            manager.initialize({'manager-smoke.txt': 'safe roundtrip'})
            self.assertEqual(manager.read_file('manager-smoke.txt'), 'safe roundtrip')
            manager.write_file('second.txt', b'second')
            self.assertEqual(manager.run_command('cat second.txt').stdout, 'second')
            target_import = manager.exec(cid, '/testbed/.venv/bin/python -c '
                                        "'import coverage; print(coverage.__file__)'")
            self.assertEqual(target_import.exit_code, 0, target_import.stderr)
            self.assertIn('/testbed/coverage/', target_import.stdout)
            self.assertTrue(manager.exec(cid, 'sleep 30', timeout=1).timed_out)
            self.assertEqual(manager.exec(cid, 'printf alive').stdout, 'alive')
            # Simulate a forged receipt: live Podman label must still refuse it.
            owner = manager.owner
            manager.owner = 'f'*32
            manager._owned[cid] = manager.owner
            with self.assertRaises(PermissionError):
                manager.stop(cid)
            manager.owner = owner
            manager._owned[cid] = owner
            self.assertEqual(manager.exec(cid, 'printf owned').stdout, 'owned')
            manager.exec(cid, 'ln -s /etc/passwd /workspace/evil-copy')
            with self.assertRaises(RuntimeError):
                manager.copy_from(cid, '/workspace/evil-copy', output/'tmp'/'evil')
            with self.assertRaises(RuntimeError):
                manager.reset()
            (output/'SMOKE_PASS.json').write_text(json.dumps({'passed': True, 'container_id': cid}))
        finally:
            manager.close()
            tempfile.tempdir = old_tempdir


if __name__ == '__main__':
    unittest.main()
