"""Tests for the explicit non-root gemmacodeagent smoke-test boundary."""
import json
from pathlib import Path
import tempfile
import unittest
import os
import sys
import types
from unittest.mock import patch

import gemmacodeagent_mario_adapter as adapter


class AdapterBoundaryTests(unittest.TestCase):
    def test_strict_gate_accepts_only_proven_reference_and_real_negative_failure(self):
        gold = {'resolved': True, 'test_exit_code': 0, 'test_output': '5 passed in 0.1s'}
        empty = {'resolved': False, 'test_exit_code': 1, 'test_output': '1 failed, 4 passed in 0.1s'}
        self.assertTrue(adapter.strict_calibration(gold, empty))
        for negative in (
            {**empty, 'test_output': "ModuleNotFoundError: No module named 'external_dependency'\n1 error"},
            {**empty, 'test_exit_code': 2, 'test_output': 'ERROR collecting tests/test_a.py\n1 error'},
            {**empty, 'test_output': 'Failed to apply test_patch'},
            {**empty, 'test_exit_code': 5, 'test_output': 'no tests ran'},
            {**empty, 'test_exit_code': -1, 'test_output': 'Command timed out'},
            {**empty, 'test_exit_code': 0, 'test_output': '3 skipped'},
            gold,
        ):
            with self.subTest(negative=negative):
                self.assertFalse(adapter.strict_calibration(gold, negative))
        self.assertFalse(adapter.strict_calibration({**gold, 'resolved': False}, empty))

    def test_changed_verification_launcher_fails_closed(self):
        adapter.require_supported_verification('python3 -s -m pytest tests/a.py --junitxml=/tmp/_swegemma_junit_abc.xml')
        adapter.require_supported_verification('cat /tmp/_swegemma_junit_abc.xml')
        adapter.require_supported_verification('pytest tests/a.py -q')
        with self.assertRaises(RuntimeError):
            adapter.require_supported_verification('python3 -m pytest tests/a.py --junitxml=/tmp/_swegemma_junit_abc.xml')

    def test_only_single_task_and_relative_run_are_accepted(self):
        self.assertEqual(adapter.runner_arguments(['--mode', 'gold', '--ids', 'rich_3506', '--run', 'smoke/a']), ('rich_3506', 'smoke/a'))
        for args in (['--ids', 'a', 'b', '--run', 'x'], ['--ids', 'a', '--run', '../escape'], ['--ids', 'a', '--run', '/tmp/x'], ['--run', 'x']):
            with self.assertRaises(ValueError):
                adapter.runner_arguments(args)

    def test_gate_requires_eligible_single_profile_and_exact_freeze(self):
        gate = {'eligible_ids': ['a'], 'task_bindings': {'a': {'profile': {'environment': '/env', 'actual_freeze': 'pytest==8\n'}}}}
        self.assertEqual(adapter.validated_profile(gate, 'a', 'pytest==8\n')['environment'], '/env')
        for task, freeze in [('b', 'pytest==8\n'), ('a', 'pytest==9\n')]:
            with self.assertRaises(ValueError):
                adapter.validated_profile(gate, task, freeze)
        gate['task_bindings']['a']['required_profile_ids'] = ['p1', 'p2']
        with self.assertRaises(ValueError):
            adapter.validated_profile(gate, 'a', 'pytest==8\n')

    def test_confinement_includes_both_runtimes_but_not_home_or_full_proc(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shard = root / 'job'
            workspace = shard / 'tmp/swegemma_sandbox_123/workspace'
            workspace.mkdir(parents=True)
            for name in ('base/runtime', 'base/data/wheels', 'base/wheelhouse', 'new-runtime', 'profile'):
                (root / name).mkdir(parents=True)
            (root / 'base/landlock_guard.py').touch()
            command, environment = adapter.confined_invocation('echo ok', workspace, {'PATH': '/usr/bin', 'SECRET': 'private', 'HOME': '/real-home'}, root / 'base', shard, root / 'new-runtime', root / 'profile')
            self.assertIn(str(root / 'base/runtime'), command)
            self.assertIn(str(root / 'new-runtime'), command)
            self.assertNotIn('/proc', command)
            self.assertNotIn('SECRET', environment)
            self.assertEqual(environment['HOME'], str(workspace.parent / 'tmp'))
            self.assertEqual(command[-3:], ['/bin/bash', '-c', 'echo ok'])

    def test_confinement_refuses_external_and_symlink_workspaces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'outside').mkdir()
            target = root / 'job/tmp/swegemma_sandbox_fake'
            target.parent.mkdir(parents=True)
            target.symlink_to(root / 'outside', target_is_directory=True)
            for workspace in (root / 'outside', target / 'workspace'):
                with self.assertRaises(RuntimeError):
                    adapter.confined_invocation('true', workspace, {}, root, root / 'job', root, root)

    def test_install_confinement_routes_through_guard_without_shell_or_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = root / 'job/tmp/swegemma_sandbox_123/workspace'
            workspace.mkdir(parents=True)
            received = []
            backend = types.SimpleNamespace(execute_subprocess_command=lambda **kwargs: received.append(kwargs) or 'ok')
            adapter.install_confinement(backend, root, root / 'job', root, root)
            self.assertEqual(backend.execute_subprocess_command(command='true', cwd=workspace, env={'SECRET': 'key'}, timeout=12), 'ok')
            self.assertFalse(received[0]['shell'])
            self.assertEqual(received[0]['timeout'], 12)
            self.assertNotIn('SECRET', received[0]['env'])
            self.assertEqual(received[0]['command'][1], '-I')

    def test_native_file_reads_reject_workspace_replaced_with_host_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shard = root / 'job'
            sandbox = shard / 'tmp/swegemma_sandbox_123'
            paths = {name: sandbox / name for name in ('workspace', 'tmp', 'wheels', 'venv')}
            paths['root'] = sandbox
            for value in paths.values():
                value.mkdir(parents=True, exist_ok=True)
            host = root / 'fake-host'
            host.mkdir()
            (host / 'sentinel').write_text('host-only')
            manager = types.SimpleNamespace(sandboxes={'id': paths}, _default_sandbox_id='id',
                start=lambda: 'id', exec=lambda *args, **kwargs: None,
                stop=lambda identifier: None, read_file=lambda path: (paths['workspace'].resolve() / path).read_text())
            adapter.install_native_guards(manager, shard)
            paths['workspace'].rename(sandbox / 'old-workspace')
            paths['workspace'].symlink_to(host, target_is_directory=True)
            with self.assertRaises((RuntimeError, OSError)):
                manager.read_file('sentinel')
            with self.assertRaises((RuntimeError, OSError)):
                manager.exec('id', 'true')
            manager.stop('id')

    def test_factory_wires_real_profile_wrapper_and_writes_audit_records(self):
        import calibrate_all_tasks as calibration
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, repo, runtime, profile = [root / name for name in ('base', 'repo', 'runtime', 'profile')]
            (base / 'data').mkdir(parents=True)
            (profile / 'lib/python3.12/site-packages').mkdir(parents=True)
            dataset = base / 'data/tasks.jsonl'
            dataset.write_text(json.dumps({'instance_id': 'rich_3506', 'repo': 'Textualize/rich'}) + '\n')
            gate = root / 'gate.json'
            gate.write_text(json.dumps({'eligible_ids': ['rich_3506'], 'dataset_sha256': calibration.sha(dataset),
                'task_bindings': {'rich_3506': {'profile': {'environment': str(profile), 'actual_freeze': 'pytest==7.4.4\n'}}}}))
            calls = []

            class Manager:
                def __init__(self, **kwargs):
                    self.options = kwargs
                sandboxes = {}
                def start(self):
                    return 'sandbox'
                def stop(self, identifier):
                    pass
                def exec(self, identifier, command, **kwargs):
                    calls.append(command)
                    return types.SimpleNamespace(exit_code=0, stdout='done', stderr='')

            package = types.ModuleType('swegemma')
            sandbox = types.ModuleType('swegemma.sandbox')
            backend = types.ModuleType('swegemma.sandbox.subprocess')
            package.sandbox = sandbox
            sandbox.subprocess = backend
            backend.SubprocessManager = Manager
            modules = {'swegemma': package, 'swegemma.sandbox': sandbox, 'swegemma.sandbox.subprocess': backend}
            environment = {'GEMMA_BASE': str(base), 'GEMMA_REPO': str(repo), 'GEMMA_EVAL_RUNTIME': str(runtime),
                           'GEMMA_PREFLIGHT_GATE': str(gate)}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, environment), \
                    patch.object(sys, 'argv', ['runner', '--ids', 'rich_3506', '--run', 'smoke']), \
                    patch.object(sys, 'prefix', str(runtime)), patch.object(calibration, 'freeze', return_value='pytest==7.4.4\n'), \
                    patch.object(adapter, 'install_confinement') as confinement, \
                    patch.object(adapter, 'install_native_guards', side_effect=lambda manager, shard: manager):
                manager = adapter.ChrootManager(timeout_seconds=300)
                manager.exec('sandbox', 'echo ok')
            self.assertFalse(manager.options['system_site_packages'])
            self.assertEqual(manager.timeout_seconds, 300)
            self.assertEqual(confinement.call_args.args[-2:], (runtime.resolve(), profile.resolve()))
            output = repo / 'runs/smoke/mario-adapter'
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertIn('not-upstream-chroot', manifest['environment'])
            self.assertEqual(manifest['task_id'], 'rich_3506')
            self.assertEqual(json.loads((output / 'commands.json').read_text())[0]['command'], 'echo ok')
            self.assertEqual(calls, ['echo ok'])


if __name__ == '__main__':
    unittest.main()
