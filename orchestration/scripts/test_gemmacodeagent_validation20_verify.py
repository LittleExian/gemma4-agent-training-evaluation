"""Contracts for one sanitized verifier shared by R2E and SWE tasks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gemmacodeagent_validation20_verify import (
    assess_result, hidden_plan, execute_test, validate_patch,
)
from swegemma.sandbox.base import ExecResult


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = self.root/'payload'
        (self.payload/'r2e_tests').mkdir(parents=True)
        (self.payload/'r2e_tests/__init__.py').write_text('')
        (self.payload/'r2e_tests/test_1.py').write_text('def test_one(): assert True\n')
        (self.payload/'gold.patch').write_text('PRIVATE_GOLD')
        (self.payload/'old-files.json').write_text('PRIVATE_OLD_SOURCE')
        self.runtime = {'task_id': 'r2e_demo-abcdef', 'passed': True,
                        'image': 'docker.io/example@sha256:'+'a'*64,
                        'repo_cwd': '/testbed', 'resolved_base_commit': 'b'*40,
                        'python_executable': '/usr/bin/python3',
                        'payload_dir': str(self.payload), 'test_files': ['test_1.py'],
                        'test_command': 'python -m pytest r2e_tests --junitxml=/out/junit.xml',
                        'results': {'gold': {'test_outcomes': {'r2e_tests.test_1::test_one': 'passed'}}}}

    def tearDown(self):
        self.temp.cleanup()

    def test_hidden_plan_never_copies_reference_payload(self):
        plan = hidden_plan(self.runtime)
        names = {str(item['source'].relative_to(self.payload)) for item in plan}
        self.assertEqual(names, {'r2e_tests/__init__.py', 'r2e_tests/test_1.py'})

    def test_tornado_wrapper_is_only_additional_allowed_payload(self):
        self.runtime['test_command'] = 'python /probe/run_unittest_junit.py'
        (self.payload/'run_unittest_junit.py').write_text('print("runner")')
        plan = hidden_plan(self.runtime)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[-1]['destination'], '/probe/run_unittest_junit.py')

    def test_hidden_symlinks_and_payload_escape_rejected(self):
        (self.payload/'r2e_tests/test_1.py').unlink()
        (self.payload/'r2e_tests/test_1.py').symlink_to(self.payload/'gold.patch')
        with self.assertRaises((ValueError, PermissionError)):
            hidden_plan(self.runtime)
        self.runtime['test_files'] = ['../gold.patch']
        with self.assertRaises(ValueError):
            hidden_plan(self.runtime)

    def test_candidate_cannot_change_hidden_or_metadata(self):
        for name in ('r2e_tests/test_1.py', '.git/config', 'tests/test_x.py',
                     '../outside.py', 'gold.patch'):
            text = f'--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-x\n+y\n'
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_patch(text, ['/testbed/r2e_tests/test_1.py'], '/testbed')

    def test_legitimate_skips_must_match_gold(self):
        runtime = dict(self.runtime, results={'gold': {'test_outcomes': {'a': 'passed', 'b': 'skipped'}}})
        result = assess_result(runtime, 0, {'a': 'passed', 'b': 'skipped'})
        self.assertTrue(result['resolved'])
        self.assertEqual(result['passed'], 1)
        self.assertFalse(assess_result(runtime, 0, {'a': 'passed', 'b': 'passed'})['resolved'])
        self.assertFalse(assess_result(runtime, 0, {'a': 'passed'})['expected_tests_covered'])

    def test_candidate_collection_error_timeout_invalid_are_not_infra(self):
        for exit_code, outcomes, kind in [(2, {'a': 'error'}, 'candidate_test_failure'),
                                          (124, {}, 'candidate_test_timeout'),
                                          (1, {}, 'candidate_invalid_patch')]:
            result = assess_result(self.runtime, exit_code, outcomes, failure_kind=kind)
            self.assertFalse(result['resolved'])
            self.assertFalse(result['infra_error'])

    def test_invalid_candidate_encoding_does_not_start_container_or_mark_infra(self):
        candidate = self.root/'invalid.patch'
        candidate.write_bytes(b'\xff\xfe')
        with patch('gemmacodeagent_validation20_verify.PodmanManager') as manager:
            result = execute_test(self.runtime, candidate, self.root/'invalid-result')
        manager.assert_not_called()
        self.assertFalse(result['infra_error'])
        self.assertEqual(result['failure_kind'], 'candidate_invalid_patch')

    def test_manager_control_plane_failure_is_infrastructure_error(self):
        manager = unittest.mock.Mock()
        manager.tmp = self.root/'manager-tmp'
        manager.tmp.mkdir()
        manager.start.return_value = 'a'*64
        manager.exec.side_effect = RuntimeError('podman inspect: container does not exist')
        candidate = self.root/'empty.patch'
        candidate.write_text('')
        with patch('gemmacodeagent_validation20_verify.PodmanManager', return_value=manager), patch(
                'gemmacodeagent_validation20_verify.install_hidden', return_value=['/testbed/r2e_tests/test_1.py']), patch(
                'gemmacodeagent_validation20_verify.snapshot', return_value={'stable': 'hash'}):
            result = execute_test(self.runtime, candidate, self.root/'control-plane-failure')
        self.assertTrue(result['infra_error'])
        self.assertFalse(result['resolved'])

    def test_fresh_flow_removes_staged_candidate_and_preserves_command(self):
        runtime = self.runtime
        instances = []
        class FakeManager:
            def __init__(self, runtime, output):
                self.output, self.tmp = output, output/'tmp'
                self.tmp.mkdir(parents=True)
                self.commands, self.copied = [], []
                instances.append(self)
            def start(self):
                (self.output/'sandbox-proof-fake.json').write_text('{"passed":true}')
                return 'a'*64
            def exec(self, cid, command, timeout=None):
                self.commands.append(command)
                if 'VALIDATION_HIDDEN_HASH' in command:
                    return ExecResult(stdout=json.dumps({'test': {'sha256': 'stable'}}))
                return ExecResult(stdout='test log')
            def copy_to(self, cid, source, destination):
                self.copied.append((Path(source).read_bytes(), destination))
            def copy_from(self, cid, source, destination):
                Path(destination).write_text('<testsuite><testcase classname="r2e_tests.test_1" '
                                             'name="test_one"/></testsuite>')
            def cleanup_all(self):
                self.cleaned = True
        candidate = self.root/'gold.patch'
        candidate.write_text('--- a/demo.py\n+++ b/demo.py\n@@ -1 +1 @@\n-old\n+new\n')
        with patch('gemmacodeagent_validation20_verify.PodmanManager', FakeManager):
            result = execute_test(runtime, candidate, self.root/'verify')
        manager = instances[0]
        self.assertTrue(result['resolved'])
        self.assertTrue(manager.cleaned)
        self.assertFalse(any(b'PRIVATE_' in data for data, _ in manager.copied))
        test_index = next(i for i, cmd in enumerate(manager.commands) if runtime['test_command'] in cmd)
        self.assertTrue(any('rm -f' in cmd and 'candidate.patch' in cmd
                            for cmd in manager.commands[:test_index]))
        self.assertIn(runtime['test_command'], manager.commands[test_index])


if __name__ == '__main__':
    unittest.main()
