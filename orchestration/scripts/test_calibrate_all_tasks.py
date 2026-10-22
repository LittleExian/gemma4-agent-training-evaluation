"""Pure preflight classification tests; no model, network or server required."""
import importlib.util
from pathlib import Path
import tempfile
import json
import shutil
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch
import unittest

spec = importlib.util.spec_from_file_location('calibrate', Path(__file__).with_name('calibrate_all_tasks.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(resolved=False, code=1, output='1 failed, 2 passed in 0.1s', error=None):
    return {'resolved': resolved, 'test_exit_code': code, 'test_output': output, 'error': error}


class CalibrationTests(unittest.TestCase):
    def test_shards_cover_once(self):
        shards = [module.select_shard(list(range(129)), i, 8) for i in range(8)]
        self.assertEqual(sorted(x for shard in shards for x in shard), list(range(129)))
        with self.assertRaises(ValueError):
            module.select_shard([], 8, 8)

    def test_true_test_failure_calibrates(self):
        self.assertEqual(module.classify_pair(record(True, 0, '3 passed'), record())['status'], 'calibrated')

    def test_empty_pass_is_nondiscriminating(self):
        self.assertEqual(module.classify_pair(record(True, 0, '3 passed'), record(True, 0, '3 passed'))['status'], 'nondiscriminating')

    def test_gold_fail_uncalibrated(self):
        self.assertEqual(module.classify_pair(record(), record())['status'], 'uncalibrated')

    def test_collection_failure_not_calibration(self):
        result = record(output='ERROR collecting tests/test_x.py\nModuleNotFoundError: No module named \'inline_snapshot\'\n1 error')
        verdict = module.classify_pair(record(True, 0, '3 passed'), result)
        self.assertEqual(verdict['status'], 'uncalibrated')
        self.assertIn('missing_dependency_or_import', verdict['empty']['categories'])
        self.assertEqual(verdict['empty']['missing_modules'], ['inline_snapshot'])

    def test_landlock_success_is_not_permission_error(self):
        self.assertEqual(module.diagnose(record(output='Landlock filesystem confinement enabled\n1 failed'))['categories'], [])
        self.assertIn('permission_error', module.diagnose(record(output='PermissionError: denied'))['categories'])

    def test_timeout_patch_and_no_pytest_rejected(self):
        for result in (record(code=-1, error='Timeout expired'), record(error='Failed to apply test_patch'), record(output='No module named pytest'), record(output='nothing happened')):
            self.assertEqual(module.classify_pair(record(True, 0, '3 passed'), result)['status'], 'uncalibrated')

    def test_checkpoint_requires_identical_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            module.atomic_json(path, {**record(), 'fingerprint': 'a', 'instance_id': '../external'})
            self.assertTrue(module.checkpoint_valid(path, 'a', '../external'))
            self.assertFalse(module.checkpoint_valid(path, 'b', '../external'))
            self.assertEqual(module.task_directory(directory, '../external').parent, Path(directory))

    def test_zero_or_all_skip_is_not_gold_pass(self):
        for output in ('5 skipped in 1.0s', '0 passed in 1.0s', 'nothing happened'):
            self.assertEqual(module.classify_pair(record(True, 0, output), record())['status'], 'uncalibrated')

    def test_counts_parse_only_last_summary(self):
        result = module.pytest_counts('trace mentions 99 passed\n1 failed, 4 passed, 2 skipped, 3 errors in 1.2s')
        self.assertEqual(result, {'passed': 4, 'failed': 1, 'skipped': 2, 'errors': 3})

    def test_profile_map_requires_complete_valid_task_mapping(self):
        mapping = {'schema_version': 1, 'profiles': {'p': {'environment': '/env/p', 'requirements': ['pytest==8.4.1']}},
                   'tasks': {'task': 'p'}}
        profile = module.resolve_profile(mapping, 'task')
        self.assertEqual(profile['profile_id'], 'p')
        self.assertEqual(profile['environment'], '/env/p')
        with self.assertRaises(ValueError):
            module.resolve_profile(mapping, 'unknown')

    def test_isolated_manager_uses_only_profile_and_checks_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / 'profile'
            (environment / 'lib/python3.12/site-packages').mkdir(parents=True)
            sandbox = root / 'sandbox'
            workspace = sandbox / 'workspace'
            workspace.mkdir(parents=True)
            target = workspace / 'rich/__init__.py'

            class Manager:
                def __init__(self, **kwargs):
                    self.options = kwargs
                    self.sandboxes = {}
                    self.origin = str(target)

                def start(self):
                    venv = sandbox / 'venv'
                    (venv / 'lib/python3.12/site-packages').mkdir(parents=True)
                    self.sandboxes['id'] = {'venv': venv, 'workspace': workspace}
                    return 'id'

                def exec(self, identifier, command, **kwargs):
                    output = json.dumps({'package': 'rich', 'origin': self.origin}) if 'find_spec' in command else '1 passed'
                    return SimpleNamespace(exit_code=0, stdout=output, stderr='')

            transcript = []
            manager = module.isolated_manager(Manager, root, environment, transcript, 'rich')
            identifier = manager.start()
            self.assertFalse(manager.options['system_site_packages'])
            pth = sandbox / 'venv/lib/python3.12/site-packages/_preflight_test_env.pth'
            self.assertEqual(pth.read_text().strip(), str(environment / 'lib/python3.12/site-packages'))
            manager.exec(identifier, 'python3 -s -m pytest tests/test_rich.py')
            self.assertEqual(transcript[0]['phase'], 'target_package_origin')
            manager.origin = str(environment / 'lib/python3.12/site-packages/rich/__init__.py')
            with self.assertRaisesRegex(RuntimeError, 'origin outside workspace'):
                manager.exec(identifier, 'python3 -s -m pytest tests/test_rich.py')

    def test_probe_records_interpreter_paths_and_dependency_versions(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(module.environment_probe_code('json'), {})
        result = json.loads(capture.getvalue())
        self.assertEqual(result['package'], 'json')
        self.assertTrue(result['executable'])
        self.assertTrue(result['python_version'])
        self.assertIsInstance(result['site_packages'], list)
        self.assertIsInstance(result['sys_path'], list)
        self.assertEqual(set(result['dependencies']), {'pytest', 'pydantic', 'starlette', 'Pygments',
                                                     'inline-snapshot', 'pytest-httpbin'})

    def test_python310_profile_rebuilds_then_injects_matching_pth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / 'profile'
            profile_site = environment / 'lib/python3.10/site-packages'
            profile_site.mkdir(parents=True)
            sandbox_venv = root / 'sandbox/venv'

            class Manager:
                def __init__(self, **kwargs):
                    self.sandboxes = {}

                def start(self):
                    (sandbox_venv / 'lib/python3.12/site-packages').mkdir(parents=True)
                    self.sandboxes['id'] = {'venv': sandbox_venv, 'workspace': root / 'sandbox/workspace'}
                    return 'id'

                def exec(self, *args, **kwargs):
                    raise AssertionError('No test command should execute in this startup test')

            def rebuild(arguments, **kwargs):
                self.assertEqual(arguments, [str(environment / 'bin/python'), '-m', 'venv', '--clear', str(sandbox_venv)])
                self.assertTrue(kwargs['check'])
                shutil.rmtree(sandbox_venv)
                (sandbox_venv / 'lib/python3.10/site-packages').mkdir(parents=True)
                return SimpleNamespace(returncode=0)

            with patch.object(module.subprocess, 'run', side_effect=rebuild) as mocked:
                manager = module.isolated_manager(Manager, root, environment, [], 'requests')
                self.assertEqual(manager.start(), 'id')
                mocked.assert_called_once()
            pth = sandbox_venv / 'lib/python3.10/site-packages/_preflight_test_env.pth'
            self.assertEqual(pth.read_text().strip(), str(profile_site))
            self.assertFalse((sandbox_venv / 'lib/python3.12').exists())

    def test_counts_keep_denominator(self):
        counts = module.summarize([{'status': 'calibrated'}, {'status': 'nondiscriminating'}], 129)
        self.assertEqual(counts['dataset_count'], 129)
        self.assertEqual(counts['uncalibrated'], 0)
        self.assertEqual(counts['pending'], 127)


if __name__ == '__main__':
    unittest.main()
