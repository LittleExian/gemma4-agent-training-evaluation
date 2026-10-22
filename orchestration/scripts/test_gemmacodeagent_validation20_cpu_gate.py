import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gemmacodeagent_validation20_cpu_gate import build_gate, controls_ready, required_cases_passed, sanitized_proof_ready


class GateContractTests(unittest.TestCase):
    def fixture(self):
        runtime = {'task_id': 'example', 'results': {'gold': {'test_outcomes': {'a': 'passed', 'b': 'skipped'}}}}
        gold = {'task_id': 'example', 'test_exit_code': 0, 'resolved': True,
                'expected_tests_covered': True, 'infra_error': False,
                'hidden_payload_unchanged': True, 'test_outcomes': {'a': 'passed', 'b': 'skipped'}}
        empty = dict(gold, test_exit_code=1, resolved=False, test_outcomes={'a': 'failed', 'b': 'skipped'})
        return runtime, gold, empty

    def test_exact_nonempty_discriminating_controls(self):
        self.assertTrue(controls_ready(*self.fixture()))
        for field, value in [('test_exit_code', 0), ('infra_error', True),
                             ('expected_tests_covered', False), ('hidden_payload_unchanged', False)]:
            runtime, gold, empty = self.fixture()
            empty = dict(empty, **{field: value})
            self.assertFalse(controls_ready(runtime, gold, empty))
        runtime, gold, empty = self.fixture()
        self.assertFalse(controls_ready(runtime, gold, dict(empty, test_outcomes={'a': 'failed'})))
        self.assertFalse(controls_ready(runtime, gold, dict(empty, task_id='other')))
        self.assertFalse(controls_ready(runtime, gold, dict(empty, test_outcomes=gold['test_outcomes'])))

    def test_gold_skip_cannot_replace_required_pass(self):
        source = {'expected_output_json': '{"C.test_a": "PASSED"}'}
        self.assertTrue(required_cases_passed('r2e_task', source, {'r2e_tests.test_1.C::test_a': 'passed'}))
        self.assertFalse(required_cases_passed('r2e_task', source, {'r2e_tests.test_1.C::test_a': 'skipped'}))
        self.assertFalse(required_cases_passed('r2e_task', source, {'r2e_tests.test_1.C::test_b': 'passed'}))

    def test_truncated_metadata_requires_every_matching_case(self):
        source = {'FAIL_TO_PASS': ['tests/a.py::test_f[-length'], 'PASS_TO_PASS': []}
        good = {'tests/a.py::test_f[-length0]': 'passed', 'tests/a.py::test_f[-length1]': 'passed'}
        self.assertTrue(required_cases_passed('swe', source, good))
        self.assertFalse(required_cases_passed('swe', source, dict(good, **{'tests/a.py::test_f[-length1]': 'skipped'})))
        self.assertFalse(required_cases_passed('swe', source, {}))

    def test_proof_requires_source_identity_and_absent_objects(self):
        proof = {'forbidden_objects_absent': True, 'other_git': [], 'artifacts_remaining': [],
                 'source_tree_before': 'a'*64, 'source_tree_after': 'a'*64, 'history_count': 1,
                 'original_objects_checked': ['a'*40], 'removed_git': ['/testbed/.git']}
        self.assertTrue(sanitized_proof_ready(proof))
        for field, value in [('source_tree_after', 'b'*64), ('forbidden_objects_absent', False),
                             ('history_count', 2), ('other_git', ['/root/repo/.git'])]:
            self.assertFalse(sanitized_proof_ready(dict(proof, **{field: value})))

    def test_full_gate_binds_all20_and_rejects_payload_drift_without_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ('data/runtimes', 'source-records', 'reports', 'src'):
                (root/name).mkdir(parents=True)
            tasks = ['swe-'+str(number) for number in range(20)]
            (root/'data/selection.json').write_text(json.dumps({'tasks': tasks}))
            (root/'data/tasks.jsonl').write_text('\n'.join(json.dumps({'instance_id': name, 'patch': '', 'test_patch': ''}) for name in tasks))
            (root/'src/critical.py').write_text('frozen=True\n')
            fake = root/'reports/fake.json'
            fake.write_text(json.dumps({'passed': True, 'fake_backend': True,
                'counts_as_model_evaluation': False, 'requests': 5,
                'official_runner_keyword_abi': True, 'real_podman_sandbox': True,
                'fresh_hidden_tests': True, 'empty_patch_unresolved': True,
                'graph_embedding_unavailable_fallback_exercised': True}))
            index = {}
            for name in tasks:
                runtime, gold, empty = self.fixture()
                runtime['task_id'] = name
                source = root/'source-records'/(name+'.source.json')
                source.write_text(json.dumps({'selection': {'task_id': name}, 'source': {'FAIL_TO_PASS': ['a'], 'PASS_TO_PASS': []}}))
                payload = root/name/'payload'
                payload.mkdir(parents=True)
                (payload/'test.patch').write_text('trusted hidden test')
                (payload/'gold.patch').write_text('trusted gold')
                runtime.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), payload_dir=str(payload),
                               payload_sha256={'test.patch': hashlib.sha256((payload/'test.patch').read_bytes()).hexdigest()},
                               test_command='exact tests', repo_cwd='/testbed')
                (root/'data/runtimes'/(name+'.json')).write_text(json.dumps(runtime))
                index[name] = {}
                for kind, result in [('gold', gold), ('empty', empty)]:
                    control = root/name/kind
                    control.mkdir()
                    proof = control/'proof.json'
                    proof.write_text(json.dumps({'forbidden_objects_absent': True, 'other_git': [],
                        'artifacts_remaining': [], 'history_count': 1, 'removed_git': ['/testbed/.git'],
                        'original_objects_checked': ['a'*40], 'source_tree_before': 'a'*64, 'source_tree_after': 'a'*64}))
                    result = dict(result, task_id=name, manager_proofs=[str(proof)],
                                  patch_sha256=hashlib.sha256((payload/'gold.patch').read_bytes()).hexdigest())
                    (control/'verification.json').write_text(json.dumps(result))
                    for filename in ('junit.xml', 'test.log', 'hidden-test-hashes.json'):
                        (control/filename).write_text('evidence')
                    (control/'injected-hidden.json').write_text(json.dumps([{'payload_relative': 'test.patch',
                        'sha256': hashlib.sha256((payload/'test.patch').read_bytes()).hexdigest()}]))
                    (control/'executed-command.json').write_text(json.dumps({'command': 'exact tests', 'repo_cwd': '/testbed', 'trusted_control': kind == 'gold'}))
                    index[name][kind] = str(control)
            index_path = root/'reports/index.json'
            index_path.write_text(json.dumps(index))
            gate = build_gate(root, index_path, fake)
            self.assertEqual(len(gate['tasks']), 20)
            self.assertTrue(gate['model_evaluation_held_by_user'])
            self.assertTrue(gate['no_gpu_started'])
            self.assertIn(str(root/'src/critical.py'), gate['artifact_sha256'])
            (root/tasks[-1]/'payload/test.patch').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'payload drift'):
                build_gate(root, index_path, fake)
            self.assertFalse((root/'reports/cpu-gate.json').exists())


if __name__ == '__main__':
    unittest.main()
