"""Reference-control gate regression tests; no model or server required."""
import importlib.util
from pathlib import Path
import tempfile
import json
import unittest

spec = importlib.util.spec_from_file_location('finalize', Path(__file__).with_name('finalize_preflight.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(variant, output=None, exit_code=None):
    root = '/run/generation-1/shard-0/tmp/swegemma_sandbox_x'
    profile = {'profile_id': 'rich-a', 'environment': '/run/test-envs/rich-a',
               'actual_freeze': 'pytest==8.4.2\n', 'interpreter_version': 'Python 3.12.4', 'python': '3.12',
               'requirements': ['pytest==8.4.2']}
    gold = variant == 'gold'
    return {'instance_id': 'task', 'variant': variant, 'fingerprint': 'same',
            'resolved': gold, 'test_exit_code': exit_code if exit_code is not None else (0 if gold else 1),
            'test_output': output if output is not None else ('3 passed in 1s' if gold else '1 failed, 2 passed in 1s'),
            'dependency_profile': profile, 'environment_probes': [{
                'package': 'rich', 'origin': root + '/workspace/rich/__init__.py',
                'executable': root + '/venv/bin/python3', 'python_version': '3.12.4 (build)',
                'site_packages': [root + '/venv/lib/python3.12/site-packages'],
                'sys_path': [root + '/workspace', root + '/venv/lib/python3.12/site-packages',
                             '/run/test-envs/rich-a/lib/python3.12/site-packages', '/usr/lib/python3.12'],
                'dependencies': {'pytest': '8.4.2'}}]}


def task(**kwargs):
    return {'instance_id': 'task', 'repo': 'Textualize/rich', 'patch': '--- a/rich/foo.py\n+++ b/rich/foo.py\n+def feature(): pass', **kwargs}


class FinalGateTests(unittest.TestCase):
    def gate(self, empty=None, gold=None, item=None):
        return module.classify_pair(item or task(), gold or record('gold'), empty or record('empty'))

    def test_real_negative(self):
        self.assertEqual(self.gate()['gate'], 'eligible')

    def test_external_dependency_is_quarantined(self):
        empty = record('empty', "ModuleNotFoundError: No module named 'inline_snapshot'\n1 error in 1s", 2)
        self.assertEqual(self.gate(empty)['gate'], 'quarantined')

    def test_repo_missing_module_is_implementation(self):
        empty = record('empty', "ModuleNotFoundError: No module named 'rich.foo'\n1 error in 1s", 2)
        result = self.gate(empty)
        self.assertEqual(result['gate'], 'eligible')
        self.assertIn('expected_missing_implementation', result['evidence'][0]['kind'])

    def test_script_missing_module_requires_patch_path(self):
        empty = record('empty', "ModuleNotFoundError: No module named 'scripts.prepare_release'\n1 error in 1s", 2)
        item = task(patch='--- a/scripts/prepare_release.py\n+++ b/scripts/prepare_release.py\n+def main(): pass')
        self.assertEqual(self.gate(empty, item=item)['gate'], 'eligible')
        self.assertEqual(self.gate(empty)['gate'], 'quarantined')

    def test_reference_added_docs_module_is_not_external_dependency(self):
        empty = record('empty', "ModuleNotFoundError: No module named 'docs_src.json_base64_bytes'\n1 error in 1s", 2)
        item = task(patch='--- a/docs_src/json_base64_bytes/tutorial001_py310.py\n+++ b/docs_src/json_base64_bytes/tutorial001_py310.py\n+implementation')
        self.assertEqual(self.gate(empty, item=item)['gate'], 'eligible')

    def test_missing_repo_api_reference_patch(self):
        empty = record('empty', "ImportError: cannot import name 'feature' from 'rich.foo' (/workspace/rich/foo.py)\n1 error in 1s", 2)
        self.assertEqual(self.gate(empty)['gate'], 'eligible')

    def test_empty_error_and_fixture_failures_can_be_negative(self):
        self.assertEqual(self.gate(record('empty', 'ERROR tests/test_x.py::test_y - ValueError: bad\n3 passed, 1 error in 1s', 1))['gate'], 'eligible')

    def test_patch_timeout_no_tests_rejected(self):
        for output in ('Failed to apply test_patch\n1 failed', 'Command timed out\n1 failed', 'no tests ran', '5 skipped in 1s'):
            self.assertEqual(self.gate(record('empty', output))['gate'], 'quarantined')

    def test_normal_landlock_is_not_error(self):
        self.assertEqual(self.gate(record('empty', '1 failed, 2 passed in 1s\nLandlock filesystem confinement active (ABI 4)'))['gate'], 'eligible')

    def test_permission_error_only_reviewed_rich4070_override(self):
        empty = record('empty', "test_input_password getpass.py PermissionError: Permission denied: '/dev/tty'\n1 failed, 123 passed in 1s")
        item = task(instance_id='rich_4070', patch='--- a/rich/console.py\n+++ b/rich/console.py\n-from getpass import getpass\n+        import getpass as _getpass_mod\n+        result = _getpass_mod.getpass("")')
        result = self.gate(empty, item=item)
        self.assertEqual(result['gate'], 'eligible')
        self.assertIn('reviewed_reference_failure', [entry['kind'] for entry in result['evidence']])
        self.assertEqual(self.gate(empty)['gate'], 'quarantined')

    def test_empty_pass_nondiscriminating_and_zero_gold_rejected(self):
        empty = record('empty', '3 passed in 1s', 0)
        empty['resolved'] = True
        self.assertEqual(self.gate(empty)['gate'], 'nondiscriminating')
        self.assertEqual(self.gate(gold=record('gold', '5 skipped in 1s', 0))['gate'], 'quarantined')

    def test_profile_or_probe_mismatch_quarantines(self):
        empty = record('empty')
        empty['dependency_profile'] = {**empty['dependency_profile'], 'actual_freeze': 'pytest==9.0.0'}
        self.assertEqual(self.gate(empty)['gate'], 'quarantined')
        gold = record('gold')
        gold['environment_probes'][0]['sys_path'].append('/ml/runtime/lib/python3.12/site-packages')
        self.assertEqual(self.gate(gold=gold)['gate'], 'quarantined')
        gold = record('gold')
        gold['environment_probes'][0]['dependencies']['pytest'] = '9.0.0'
        self.assertEqual(self.gate(gold=gold)['gate'], 'quarantined')

    def test_reviewed_infinite_loop_controls_preserve_exit124(self):
        cases = {
            'rich_4006': ('rich/cells.py', 'tests/test_cells.py',
                          ' while index < codepoint_count:\n-        elif spans:\n+                spans.append((index, index := index + 1, 0))',
                          '+def test_chop_cells_zero_width(): pass\n+"\\x1b"; "\\ufe0f"; "\\u200d"'),
            'rich_3772': ('rich/traceback.py', 'tests/test_traceback.py',
                          '+cause is not None and cause is not exc_value\n+_visited_exceptions\n+grouped_exceptions.add(exception)',
                          '+def test_recursive_exception():\n+    raise e from e'),
            'rich_3480': ('rich/text.py', 'tests/test_text.py',
                          '-for start, end, style in text._spans\n+for start, end, style in text._spans.copy()',
                          '+def test_append_loop_regression():\n+    a.append(a)\n+    b.append_text(b)'),
        }
        for instance_id, (source, test_path, patch, test_patch) in cases.items():
            item = task(instance_id=instance_id, patch='--- a/' + source + '\n+++ b/' + source + '\n' + patch,
                        test_patch='--- a/' + test_path + '\n+++ b/' + test_path + '\n' + test_patch)
            empty = record('empty', 'STDOUT:\n....\nSTDERR:\nLandlock filesystem confinement active (ABI 4)', 124)
            empty['commands'] = [{'command': 'python3 -s -m pytest ' + test_path, 'exit_code': 124}]
            result = self.gate(empty, item=item)
            self.assertEqual(result['gate'], 'eligible', instance_id)
            self.assertEqual(result['empty_raw_exit_code'], 124)
            self.assertEqual(empty['test_exit_code'], 124)
            self.assertIn('reviewed_reference_nontermination', [entry['kind'] for entry in result['evidence']])
            self.assertEqual(self.gate(empty, item={**item, 'patch': ''})['gate'], 'quarantined')
            self.assertEqual(self.gate(empty, item={**item, 'instance_id': 'unreviewed'})['gate'], 'quarantined')
            self.assertEqual(self.gate(empty, gold=record('gold', '1 failed in 1s', 1), item=item)['gate'], 'quarantined')

    def test_expired_snapshot_tls_fixture_stays_quarantined(self):
        item = task(instance_id='requests_6757', repo='psf/requests')
        gold = record('gold', 'SSLError: SSLV3_ALERT_CERTIFICATE_EXPIRED\n1 failed, 326 passed in 38s', 1)
        gold['resolved'] = False
        result = self.gate(gold=gold, item=item)
        self.assertEqual(result['gate'], 'quarantined')
        self.assertIn('snapshot_tls_fixture_expired', result['reasons'])
        entry = next(entry for entry in result['evidence'] if entry['kind'] == 'snapshot_tls_fixture_expired')
        self.assertEqual(entry['not_after'], '2026-03-13')
        self.assertFalse(entry['tls_verification_disabled'])

    def test_matrix_requires_union_of_required_nodes_and_negative_control(self):
        item = task(instance_id='rich_3486', test_patch='+def test_old(): pass\n+def test_new(): pass')
        pairs = []
        for index, (passed, skipped) in enumerate((('test_old', 'test_new'), ('test_new', 'test_old'))):
            gold, empty = record('gold', '1 passed, 1 skipped in 1s'), record('empty')
            gold['resolved'] = False
            gold['error_message'] = 'Required test node did not pass: ' + skipped
            for entry in (gold, empty):
                entry['dependency_profile'] = {**entry['dependency_profile'], 'profile_id': 'profile' + str(index)}
            gold['commands'] = [{'command': 'cat /tmp/_swegemma_junit_a.xml', 'stdout':
                '<testsuites><testsuite tests="2"><testcase name="' + passed + '"/><testcase name="' + skipped + '"><skipped/></testcase></testsuite></testsuites>'}]
            pairs.append({'gold': gold, 'empty': empty, 'generation': index + 1, 'files': {}})
        result = module.classify_matrix(item, pairs)
        self.assertEqual(result['gate'], 'eligible')
        self.assertEqual(result['calibration_status'], 'calibrated_matrix')
        self.assertEqual(result['required_profile_ids'], ['profile0', 'profile1'])
        self.assertFalse(result['matrix']['arms'][0]['original_gold_resolved'])
        self.assertEqual(module.classify_matrix(item, pairs[:1])['gate'], 'quarantined')
        pairs[1]['gold']['commands'] = []
        self.assertEqual(module.classify_matrix(item, pairs)['gate'], 'quarantined')

    def test_reports_keep_all129_tasks_and_do_not_modify_raw_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, run = root / 'base', root / 'run'
            (base / 'data').mkdir(parents=True)
            tasks = [task(instance_id='task' if i == 0 else f'task{i}') for i in range(129)]
            (base / 'data/tasks.jsonl').write_text('\n'.join(json.dumps(item) for item in tasks))
            pair = run / 'generation-1/shard-0/results/hash'
            module.atomic_json(pair / 'gold.json', record('gold'))
            module.atomic_json(pair / 'empty.json', record('empty'))
            original = (pair / 'gold.json').read_bytes()
            summary = module.finalize(base, run, [1, 2])
            self.assertEqual(summary['dataset_count'], 129)
            self.assertEqual(summary['eligible'], 1)
            self.assertEqual(summary['quarantined'], 128)
            gate = json.loads((run / 'final-gate.json').read_text())
            self.assertEqual(len(gate['quarantined_ids']), 128)
            self.assertTrue((run / 'calibration.csv').exists())
            self.assertEqual((pair / 'gold.json').read_bytes(), original)

    def test_declared_pins_are_checked_against_actual_freeze(self):
        gold = record('gold')
        gold['dependency_profile']['actual_freeze'] = 'pytest==9.0.3\n'
        gold['environment_probes'][0]['dependencies']['pytest'] = '9.0.3'
        issues = module.validate_probe(gold, 'rich')
        self.assertTrue(any('declared_requirement_version_mismatch' in issue for issue in issues), issues)
        gold = record('gold')
        gold['dependency_profile']['python'] = '3.10'
        self.assertIn('declared_python_version_mismatch', module.validate_probe(gold, 'rich'))

    def test_requirement_extras_markers_and_name_normalization(self):
        gold = record('gold')
        gold['dependency_profile']['requirements'] = [
            'PyTest[testing]>=8,<9', 'typing-extensions==4.15.0',
            'importlib-metadata>=6; python_version < "3.10"']
        gold['dependency_profile']['actual_freeze'] += 'typing_extensions==4.15.0\n'
        self.assertEqual(module.validate_probe(gold, 'rich'), [])
        gold['dependency_profile']['requirements'].append('missing-external>=1')
        self.assertTrue(any('declared_requirement_missing' in issue for issue in module.validate_probe(gold, 'rich')))

    def test_latest_complete_pair_preserves_older_if_new_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / 'generation-1/shard-0/results/hash'
            new = root / 'generation-2/shard-0/results/hash'
            for path in (old, new): path.mkdir(parents=True)
            module.atomic_json(old / 'gold.json', record('gold'))
            module.atomic_json(old / 'empty.json', record('empty'))
            module.atomic_json(new / 'gold.json', record('gold'))
            selected, provenance = module.select_pairs(root, [1, 2])
            self.assertEqual(selected['task']['generation'], 1)
            self.assertEqual(len(provenance['task']), 2)
            self.assertFalse(provenance['task'][-1]['complete'])


if __name__ == '__main__':
    unittest.main()
