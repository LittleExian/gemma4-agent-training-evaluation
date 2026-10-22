import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gemmacodeagent_validation20_task import (
    classify_agent_error, final_result, load_checked_gate, validate_endpoint,
    load_verifier,
)


class TaskContractTests(unittest.TestCase):
    def test_endpoint_local_only(self):
        self.assertEqual(validate_endpoint('http://127.0.0.1:8140/v1'),
                         'http://127.0.0.1:8140/v1')
        for value in ('https://example.com/v1', 'http://127.0.0.1:8140/v1?key=x',
                      'http://user@127.0.0.1:8140/v1'):
            with self.assertRaises(ValueError):
                validate_endpoint(value)

    def test_budget_and_context_errors_are_model_outcomes(self):
        for message in ('Agent exceeded session timeout (15 min)',
                        'Sandbox execution error: max tool calls budget exceeded',
                        'Agent exceeded tool call budget (56 calls)',
                        'Agent exceeded turn budget (160 turns)',
                        'ContextWindowExceededError: request too long',
                        'Sandbox execution error: context_length_exceeded'):
            self.assertFalse(classify_agent_error(message)['infra_error'])
        self.assertTrue(classify_agent_error('Connection refused')['infra_error'])

    def test_resolution_requires_expected_coverage(self):
        runtime = {'results': {'gold': {'test_outcomes': {'a': 'passed'}}}}
        verification = {'test_exit_code': 0, 'passed': 1, 'failed': 0, 'errors': 0,
                        'expected_tests_covered': False, 'test_outcomes': {'a': 'passed'},
                        'infra_error': False}
        result = final_result('task', None, verification, runtime)
        self.assertFalse(result['resolved'])
        self.assertEqual(result['status'], 'completed')
        result = final_result('task', 'max tool calls budget exceeded',
                              dict(verification, expected_tests_covered=True), runtime)
        self.assertTrue(result['resolved'])
        self.assertFalse(result['normal_output'])
        self.assertFalse(result['infra_error'])

    def test_candidate_collection_failure_is_completed(self):
        runtime = {'results': {'gold': {'test_outcomes': {'a': 'passed'}}}}
        verification = {'test_exit_code': 2, 'passed': 0, 'failed': 0, 'errors': 1,
                        'expected_tests_covered': False, 'test_outcomes': {},
                        'infra_error': False}
        result = final_result('task', None, verification, runtime)
        self.assertFalse(result['resolved'])
        self.assertEqual(result['status'], 'completed')

    def test_gate_detects_artifact_drift(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'reports').mkdir()
            artifact = root/'bound.json'
            artifact.write_text('{}')
            gate = {'passed': True, 'artifact_sha256': {
                str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()},
                'tasks': {'task': {'eligible': True, 'gold_passed': True,
                                   'empty_passed': False}}}
            (root/'reports/cpu-gate.json').write_text(json.dumps(gate))
            self.assertTrue(load_checked_gate(root, 'task')['passed'])
            artifact.write_text('drift')
            with self.assertRaises(ValueError):
                load_checked_gate(root, 'task')

    def test_all_sources_use_shared_sanitized_verifier(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'src').mkdir()
            module = root/'src/gemmacodeagent_validation20_verify.py'
            module.write_text('def verify_patch(*args): return {"shared": True}\n')
            for task_id in ('r2e_coveragepy-example', 'pallets__click-example'):
                self.assertEqual(load_verifier(root, task_id)(), {'shared': True})


if __name__ == '__main__':
    unittest.main()
