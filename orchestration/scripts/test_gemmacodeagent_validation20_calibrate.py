import tempfile
from pathlib import Path
import unittest

from gemmacodeagent_validation20_calibrate import (select_command, parse_junit,
    assess_controls, assess_patch, image_reference, validate_candidate, patch_paths, repository_cwd)


class CalibrationTests(unittest.TestCase):
    def test_image_root_workdir_is_not_the_repository(self):
        calls = []
        def probe(path):
            calls.append(path)
            return 'a'*40 if path == '/testbed' else None
        self.assertEqual(repository_cwd('/', 'a'*40, probe), '/testbed')
        self.assertEqual(calls, ['/testbed'])
        with self.assertRaises(ValueError):
            repository_cwd('/', 'b'*40, probe)

    def test_command_uses_official_pytest_files_and_junit(self):
        source = {'install_config': {'test_cmd': 'pytest --tb=line tests/a.py'},
                  'FAIL_TO_PASS': ['tests/a.py::test_x'], 'PASS_TO_PASS': []}
        command = select_command(source, '/usr/bin/python')
        self.assertIn('/usr/bin/python -m pytest', command)
        self.assertIn('tests/a.py', command)
        self.assertIn('--junitxml=/tmp/validation-result.xml', command)

    def test_missing_command_derives_files_without_parameter_splitting(self):
        source = {'FAIL_TO_PASS': ['tests/a.py::test_x[a b]'],
                  'PASS_TO_PASS': ['tests/b.py::C::test_y']}
        self.assertIn('tests/a.py tests/b.py', select_command(source, '/opt/python'))

    def test_official_bare_pytest_is_not_narrowed(self):
        source = {'install_config': {'test_cmd': 'pytest -ra'},
                  'FAIL_TO_PASS': ['tests/a.py::test_x'], 'PASS_TO_PASS': []}
        self.assertNotIn('tests/a.py', select_command(source, '/usr/bin/python'))

    def test_skip_does_not_satisfy_required_metadata_case(self):
        gold = {'exit_code': 0, 'test_outcomes': {'a': 'passed', 'b': 'skipped'}}
        empty = {'exit_code': 1, 'test_outcomes': {'a': 'failed', 'b': 'skipped'}}
        self.assertFalse(assess_controls(gold, empty, ['a'], ['b'])['passed'])

    def test_protected_candidate_paths_rejected(self):
        for path in ('tests/a.py', 'pkg/test_a.py', '.git/config', '../x.py', 'conftest.py'):
            patch = '--- a/' + path + '\n+++ b/' + path + '\n@@ -1 +1 @@\n-a\n+b\n'
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_candidate(patch, [])
        good = '--- a/pkg/module.py\n+++ b/pkg/module.py\n@@ -1 +1 @@\n-a\n+b\n'
        self.assertEqual(validate_candidate(good, []), ['pkg/module.py'])
        with self.assertRaises(ValueError):
            validate_candidate(good, ['pkg/module.py'])

    def test_production_configuration_paths_are_not_blanket_rejected(self):
        for path in ('tox.ini', 'pyproject.toml', 'setup.py', 'setup.cfg', 'Makefile', 'requirements.txt', '.github/workflows/lint.yml'):
            patch = '--- a/' + path + '\n+++ b/' + path + '\n@@ -1 +1 @@\n-a\n+b\n'
            self.assertEqual(validate_candidate(patch, []), [path])

    def test_candidate_collection_error_is_model_failure(self):
        runtime = {'expected_output': {'test_cases': ['a'], 'passed_cases': ['a']}}
        result = assess_patch(runtime, {'exit_code': 2, 'test_outcomes': {}})
        self.assertFalse(result['resolved'])
        self.assertFalse(result['infra_failed'])
        self.assertFalse(result['expected_tests_covered'])
        self.assertEqual(result['test_exit_code'], 2)

    def test_reject_shell_command_and_unsafe_test_path(self):
        for source in ({'install_config': {'test_cmd': 'pytest a.py; env'}},
                       {'FAIL_TO_PASS': ['../../secret.py::test_x']}):
            with self.assertRaises(ValueError):
                select_command(source, '/usr/bin/python')

    def test_junit_duplicates_preserve_failure_over_pass(self):
        xml = '<testsuites><testsuite><testcase file="tests/a.py" classname="tests.a.C" name="test_x"><failure/></testcase><testcase file="tests/a.py" classname="tests.a.C" name="test_x"/></testsuite></testsuites>'
        self.assertEqual(parse_junit(xml, ['tests/a.py']), {'tests/a.py::C::test_x': 'failed'})

    def test_junit_parameter_names_and_classname_mapping(self):
        xml = '<testsuite><testcase classname="tests.test_a" name="test_x[a b]"/></testsuite>'
        self.assertEqual(parse_junit(xml, ['tests/test_a.py']), {'tests/test_a.py::test_x[a b]': 'passed'})

    def test_controls_refuse_zero_cases_collection_errors_missing_cases(self):
        for gold, empty in (({}, {}), ({'a': 'passed'}, {}),
                            ({'a': 'error'}, {'a': 'failed'}),
                            ({'a': 'passed'}, {'a': 'passed'})):
            self.assertFalse(assess_controls({'exit_code': 0, 'test_outcomes': gold},
                {'exit_code': 1, 'test_outcomes': empty}, [])['passed'])

    def test_controls_support_truncated_official_parameter_id(self):
        gold = {'exit_code': 0, 'test_outcomes': {'t::test_x[a b]': 'passed'}}
        empty = {'exit_code': 1, 'test_outcomes': {'t::test_x[a b]': 'failed'}}
        gate = assess_controls(gold, empty, ['t::test_x[a'])
        self.assertTrue(gate['passed'])
        self.assertFalse(assess_controls(gold, empty, ['t::missing'])['passed'])

    def test_truncated_prefix_requires_every_full_parameter_case_passed(self):
        good = {'t::x[cat]': 'passed', 't::x[cat 0]': 'passed'}
        bad = {'t::x[cat]': 'failed', 't::x[cat 0]': 'passed'}
        gold, empty = {'exit_code': 0, 'test_outcomes': good}, {'exit_code': 1, 'test_outcomes': bad}
        self.assertTrue(assess_controls(gold, empty, ['t::x[cat'])['passed'])
        self.assertFalse(assess_controls({'exit_code': 0, 'test_outcomes': {**good, 't::x[cat 0]': 'skipped'}}, empty, ['t::x[cat'])['passed'])

    def test_patch_requires_every_calibrated_case_and_expected_pass(self):
        runtime = {'expected_output': {'test_cases': ['a', 'b'], 'passed_cases': ['a']}}
        self.assertTrue(assess_patch(runtime, {'exit_code': 0, 'test_outcomes': {'a': 'passed', 'b': 'skipped'}})['resolved'])
        self.assertFalse(assess_patch(runtime, {'exit_code': 0, 'test_outcomes': {'a': 'passed'}})['resolved'])

    def test_image_uses_official_or_documented_gym_naming(self):
        self.assertEqual(image_reference({'instance_id': 'dask__dask-7894'}),
                         'docker.io/xingyaoww/sweb.eval.x86_64.dask_s_dask-7894:latest')
        self.assertEqual(image_reference({'image_name': 'swerebench/x:tag'}), 'docker.io/swerebench/x:tag')


if __name__ == '__main__':
    unittest.main()
