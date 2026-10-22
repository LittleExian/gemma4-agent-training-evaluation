"""Installer safety tests; subprocesses are mocked, no package installation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('installer', Path(__file__).with_name('install_test_profiles.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(module, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        for name in ('test-envs', 'requirements', 'installation'):
            (self.root / name).mkdir()
        self.env = self.root / 'test-envs/p'
        self.profile = {'environment': str(self.env), 'python': '3.12',
                        'fingerprint': 'same', 'requirements': ['pytest[testing]==8.4.2']}
        self.frozen = 'pytest==8.4.2\n'
        self.snapshot = {'python_version': '3.12.4', 'requirement_errors': [],
                         'actual_freeze': self.frozen}

    def cached(self):
        (self.env / 'bin').mkdir(parents=True)
        (self.env / 'bin/python').touch()
        (self.env / 'requirements.freeze.txt').write_text(self.frozen)
        marker = {'success': True, 'fingerprint': 'same', 'environment': str(self.env),
                  'requirements_sha256': hashlib.sha256(('\n'.join(self.profile['requirements'])+'\n').encode()).hexdigest()}
        (self.root / 'installation/p.json').write_text(json.dumps(marker))
        return marker

    def test_cached_environment_revalidated_and_sha_recorded(self):
        self.cached()
        with patch.object(module, 'inspect_environment', return_value=self.snapshot) as inspect, patch.object(module.subprocess, 'run') as run:
            result = module.install(('p', self.profile))
        inspect.assert_called_once()
        run.assert_not_called()
        self.assertEqual(result['actual_freeze_sha256'], hashlib.sha256(self.frozen.encode()).hexdigest())

    def test_drift_requires_new_path_without_installing(self):
        self.cached()
        for snapshot in ({**self.snapshot, 'actual_freeze': 'pytest==9.0.3\n'},
                         {**self.snapshot, 'requirement_errors': ['pytest==8.4.2 not satisfied']},
                         {**self.snapshot, 'python_version': '3.10.12'}):
            with self.subTest(snapshot=snapshot), patch.object(module, 'inspect_environment', return_value=snapshot), patch.object(module.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'new environment path'):
                    module.install(('p', self.profile))
                run.assert_not_called()

    def test_missing_cached_interpreter_rejected(self):
        self.cached()
        (self.env / 'bin/python').unlink()
        with patch.object(module.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'new environment path'):
                module.install(('p', self.profile))
        run.assert_not_called()

    def test_existing_environment_without_success_marker_not_repaired(self):
        self.env.mkdir()
        with patch.object(module.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'new environment path'):
                module.install(('p', self.profile))
        run.assert_not_called()

    def test_failed_pip_check_rejects_cache(self):
        self.cached()
        with patch.object(module, 'inspect_environment', side_effect=subprocess.CalledProcessError(1, ['pip','check'])):
            with self.assertRaisesRegex(RuntimeError, 'new environment path'):
                module.install(('p', self.profile))

    def test_declared_requirements_change_cannot_reuse_marker(self):
        self.cached()
        changed = {**self.profile, 'requirements': ['pytest==9.0.3']}
        with patch.object(module.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'new environment path'):
                module.install(('p', changed))
        run.assert_not_called()

    def test_inspector_reads_runtime_and_runs_pip_check(self):
        probe = {'python_version': '3.12.4', 'requirement_errors': []}
        with patch.object(module.subprocess, 'check_output', side_effect=[json.dumps(probe), self.frozen]), patch.object(module.subprocess, 'run') as run:
            actual = module.inspect_environment(self.env, self.profile, {})
        self.assertEqual(actual['actual_freeze'], self.frozen)
        self.assertEqual(run.call_args.args[0][-2:], ['pip', 'check'])

    def test_new_environment_records_verified_freeze(self):
        def emulate(command, **kwargs):
            if 'venv' in command:
                self.env.mkdir()
        with patch.object(module.subprocess, 'run', side_effect=emulate) as run, patch.object(module, 'inspect_environment', return_value=self.snapshot):
            result = module.install(('p', self.profile))
        self.assertTrue(result['success'])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(result['actual_freeze_sha256'], hashlib.sha256(self.frozen.encode()).hexdigest())

    def test_marker_freeze_hash_mismatch_rejected(self):
        marker = self.cached()
        (self.root/'installation/p.json').write_text(json.dumps({**marker, 'actual_freeze_sha256': 'wrong'}))
        with patch.object(module, 'inspect_environment', return_value=self.snapshot):
            with self.assertRaisesRegex(RuntimeError, 'new environment path'):
                module.install(('p', self.profile))

    def test_new_install_failure_records_failure(self):
        with patch.object(module.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['venv'])):
            result = module.install(('p', self.profile))
        self.assertFalse(result['success'])
        self.assertIsNone(result['actual_freeze_sha256'])

    def test_outside_environment_rejected(self):
        with self.assertRaisesRegex(ValueError, 'inside'):
            module.install(('p', {**self.profile, 'environment': str(self.root/'outside')}))

    def test_real_probe_reads_metadata_and_requirements_without_installing(self):
        declarations = ['pip[optional_extra]>=0', 'pip<0', 'nonexistent-preflight-distribution==1',
                        'nonexistent-marker-distribution==1; python_version < "2"']
        output = subprocess.check_output([sys.executable, '-I', '-B', '-c', module.PROBE, json.dumps(declarations)], text=True)
        value = json.loads(output)
        self.assertEqual(len(value['requirement_errors']), 2)
        self.assertTrue(value['requirement_errors'][0].startswith('pip<0: installed'))
        self.assertTrue(value['requirement_errors'][1].endswith('missing distribution'))


if __name__ == '__main__':
    unittest.main()
