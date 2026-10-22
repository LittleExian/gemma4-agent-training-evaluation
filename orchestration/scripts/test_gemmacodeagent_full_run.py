import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import gemmacodeagent_full_run as runner


class RunnerTests(unittest.TestCase):
    def test_slots_validate_and_expand(self):
        self.assertEqual(runner.endpoint_slots(['http://127.0.0.1:8100/v1', 'http://127.0.0.1:8101/v1'], [2, 1]),
                         ['http://127.0.0.1:8100/v1'] * 2 + ['http://127.0.0.1:8101/v1'])
        for counts in ([0, 1], [1], [-1, 2], [9, 1]):
            with self.assertRaises(ValueError):
                runner.endpoint_slots(['a', 'b'], counts)
        self.assertEqual(len(runner.endpoint_slots(['a', 'b', 'c'], [4, 4, 4])), 12)

    def test_normal_means_terminal_without_infra_not_success_or_nonempty(self):
        good = {'patch_chars': 12, 'test_exit_code': 1, 'agent_timed_out': False}
        self.assertTrue(runner.normal_output(good))
        self.assertTrue(runner.normal_output({**good, 'patch_chars': 0}))
        self.assertTrue(runner.normal_output({**good, 'agent_timed_out': True,
                         'test_exit_code': -1, 'error_code': 'agent_budget_timeout'}))
        for bad in ({**good, 'test_exit_code': 124},
                    {**good, 'test_exit_code': -1}, {**good, 'infra_error': 'broken'}, {}):
            self.assertFalse(runner.normal_output(bad))

    def test_profile_matrix_and_missing_profile(self):
        gate = {'task_bindings': {'x': {'required_profile_ids': ['a', 'b']}}}
        self.assertEqual(runner.profiles_for(gate, 'x'), ['a', 'b'])
        with self.assertRaises(ValueError):
            runner.profiles_for({'task_bindings': {'x': {}}}, 'x')

    def test_checkpoint_immutable_and_fingerprint_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.bind_run(root, {'a': 1})
            runner.bind_run(root, {'a': 1})
            with self.assertRaises(ValueError):
                runner.bind_run(root, {'a': 2})
            runner.commit_result(root, 'task', {'ok': 1})
            runner.commit_result(root, 'task', {'ok': 1})
            with self.assertRaises(ValueError):
                runner.commit_result(root, 'task', {'ok': 2})

    def test_task_calls_controls_then_one_generation_then_matrix_verify(self):
        calls = []
        task = {'instance_id': 'x', 'repo': 'r/p'}
        def invoke(mode, profile, patch_file=None):
            calls.append((mode, profile))
            return {'patch_chars': 10, 'patch_file': '/tmp/p', 'test_exit_code': 0,
                    'resolved': True, 'agent_timed_out': False}
        with patch.object(runner, 'control_verdict', return_value={'gate': 'eligible'}), \
             patch.object(runner, 'score_patch', return_value=True):
            result = runner.evaluate_task(task, ['a', 'b'], invoke)
        self.assertEqual(calls, [('gold', 'a'), ('empty', 'a'), ('gold', 'b'), ('empty', 'b'),
                                 ('agent', 'a'), ('verify-patch', 'b')])
        self.assertTrue(result['normal_output'])
        self.assertTrue(result['resolved'])

    def test_control_classifier_dispatches_matrix_and_single(self):
        with patch.object(runner, 'classify_pair', return_value='pair') as single, \
             patch.object(runner, 'classify_matrix', return_value='matrix'):
            pair = {'gold': {}, 'empty': {}}
            self.assertEqual(runner.control_verdict({}, [pair]), 'pair')
            self.assertEqual(runner.control_verdict({}, [pair, pair]), 'matrix')
            single.assert_called_once()

    def test_single_score_preserves_false_and_timeout_is_terminal(self):
        self.assertFalse(runner.score_patch({}, [{'resolved': False}]))
        timeout = {'test_exit_code': -1, 'agent_timed_out': True,
                   'error_code': 'agent_budget_timeout', 'patch_chars': 0}
        with patch.object(runner, 'control_verdict', return_value={'gate': 'eligible'}):
            result = runner.evaluate_task({'instance_id': 'x'}, ['a', 'b'],
                                          lambda mode, profile: timeout if mode == 'agent' else {})
        self.assertTrue(result['normal_output'])
        self.assertFalse(result['resolved'])
        self.assertEqual(result['status'], 'agent_timeout')

    def test_agent_infrastructure_result_not_labeled_algorithm_completion(self):
        broken = {'test_exit_code': 1, 'infra_error': True, 'resolved': True}
        with patch.object(runner, 'control_verdict', return_value={'gate': 'eligible'}):
            result = runner.evaluate_task({'instance_id': 'x'}, ['a'],
                                          lambda mode, profile: broken if mode == 'agent' else {})
        self.assertEqual(result['status'], 'infra_error')
        self.assertFalse(result['normal_output'])
        self.assertFalse(result['resolved'])

    def test_bad_control_never_generates_patch(self):
        calls = []
        def invoke(mode, profile, patch_file=None):
            calls.append(mode)
            return {}
        with patch.object(runner, 'control_verdict', return_value={'gate': 'quarantined'}):
            result = runner.evaluate_task({'instance_id': 'x'}, ['a'], invoke)
        self.assertEqual(calls, ['gold', 'empty'])
        self.assertEqual(result['status'], 'control_failed')

    def test_rolling_preserves_endpoint_capacity_and_all_tasks(self):
        import threading
        active, maximum = {}, {}
        lock = threading.Lock()
        def work(task, endpoint):
            with lock:
                active[endpoint] = active.get(endpoint, 0) + 1
                maximum[endpoint] = max(maximum.get(endpoint, 0), active[endpoint])
            import time
            time.sleep(.01)
            with lock:
                active[endpoint] -= 1
            return {'instance_id': str(task), 'normal_output': True}
        results = []
        runner.rolling(range(9), ['a', 'a', 'b'], work, results.append)
        self.assertEqual(len(results), 9)
        self.assertLessEqual(maximum['a'], 2)
        self.assertLessEqual(maximum['b'], 1)

    def test_matrix_scores_union_of_required_nodes_once_each(self):
        records = [{'test_exit_code': 0, 'test_output': '3 passed', 'node': name}
                   for name in ('test_a', 'test_b')]
        with patch.object(runner, 'required_test_names', return_value=(['test_a', 'test_b'], 'explicit')), \
             patch.object(runner, 'junit_outcomes', side_effect=lambda r: [
                 {'name': r['node'], 'status': 'passed'}]):
            self.assertTrue(runner.score_patch({}, records))
            self.assertFalse(runner.score_patch({}, [records[0], records[0]]))
            self.assertFalse(runner.score_patch({}, [records[0], {**records[1], 'test_exit_code': 1}]))

    def test_matrix_reads_junit_commands_from_real_normalized_result_shape(self):
        records = [{'verification': {'resolved': False, 'test_exit_code': 0,
                                     'test_output': '3 passed'},
                    'commands': [{'command': 'cat /tmp/_swegemma_junit_results.xml',
                                  'stdout': '<testsuites><testsuite><testcase name="' + name + '"/></testsuite></testsuites>'}]}
                   for name in ('test_a', 'test_b')]
        with patch.object(runner, 'required_test_names', return_value=(['test_a', 'test_b'], 'explicit')):
            self.assertTrue(runner.score_patch({}, records))
            self.assertFalse(runner.score_patch({}, [records[0], records[0]]))

    def test_fake_subprocess_end_to_end_milestone_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            tasks = [{'instance_id': 'task_' + str(i), 'repo': 'owner/p'} for i in range(12)]
            (root / 'data/tasks.jsonl').write_text('\n'.join(map(json.dumps, tasks)))
            gate = {'dataset_sha256': runner.digest(root / 'data/tasks.jsonl'),
                    'eligible_ids': [t['instance_id'] for t in tasks[:11]],
                    'quarantined_ids': ['task_11'],
                    'task_bindings': {t['instance_id']: {'profile': {'profile_id': 'profile'}} for t in tasks[:11]}}
            (root / 'gate.json').write_text(json.dumps(gate))
            script = root / 'fake_task.py'
            for name in runner.EXECUTION_DEPENDENCIES:
                source = Path(runner.__file__).parent/name
                (root/name).write_bytes(source.read_bytes() if source.exists() else b'fixture')
            script.write_text('''import argparse,json,pathlib,os
p=argparse.ArgumentParser()
for arg in ('base','repo','run-dir','task-id','endpoint','mode','profile-id','output-dir','gate'):
 p.add_argument('--'+arg)
a=p.parse_args()
assert int(os.environ['GEMMA_EVAL_SLOTS']) == {'a':2,'b':1}[a.endpoint]
assert '/endpoint-slots/' in os.environ['GEMMA_SLOT_DIR']
pathlib.Path(a.output_dir,'result.json').write_text(json.dumps({
 'instance_id':a.task_id,'patch_chars':0,'test_exit_code':1,
 'resolved':False,'agent_timed_out':False}))
''')
            args = Namespace(base=root, repo=root, run_dir=root/'run', gate=root/'gate.json',
                             task_runner=script, endpoints=['a', 'b'], workers=[2, 1],
                             ids=None, fingerprint=None, invocation_timeout=20)
            with patch.object(runner, 'control_verdict', return_value={'gate': 'eligible'}):
                runner.run(args)
                progress = json.loads((args.run_dir/'progress.json').read_text())
                self.assertEqual(progress['completed'], 12)
                self.assertEqual(progress['normal_outputs'], 11)
                self.assertEqual(progress['statuses'], {'completed': 11, 'preflight_excluded': 1})
                milestone = json.loads((args.run_dir/'ten-normal-outputs.json').read_text())
                self.assertEqual(len(milestone['task_ids']), 10)
                results_before = {path.name: path.read_bytes() for path in (args.run_dir/'results').glob('*.json')}
                runner.run(args)
                self.assertEqual(results_before, {path.name: path.read_bytes() for path in (args.run_dir/'results').glob('*.json')})
            # New run retains explicit infrastructure errors and never makes the milestone.
            args.run_dir = root/'error-run'
            with patch.object(runner, 'invocation', side_effect=RuntimeError('broken environment')):
                runner.run(args)
            progress = json.loads((args.run_dir/'progress.json').read_text())
            self.assertEqual(progress['statuses']['infra_error'], 11)
            self.assertFalse((args.run_dir/'ten-normal-outputs.json').exists())
            (root/'calibrate_all_tasks_v2.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                runner.run(args)

    def test_invocation_rejects_nonzero_and_wrong_identifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root/'bad.py'
            script.write_text('raise SystemExit(3)')
            args = Namespace(base=root, repo=root, run_dir=root, gate=root/'gate', task_runner=script,
                             endpoints=['a'], workers=[4], invocation_timeout=3)
            invoke = runner.invocation(args, 'expected', 'a', root/'tasks')
            with self.assertRaisesRegex(RuntimeError, 'invocation failed'):
                invoke('gold', 'a')
            script.write_text('''import json,pathlib,sys
d=sys.argv[sys.argv.index('--output-dir')+1]
pathlib.Path(d,'result.json').write_text(json.dumps({'instance_id':'wrong'}))
''')
            with self.assertRaisesRegex(RuntimeError, 'identifier mismatch'):
                invoke('gold', 'a')
            script.write_text('import time; time.sleep(30)')
            args.invocation_timeout = .03
            with self.assertRaisesRegex(RuntimeError, 'watchdog expired'):
                invoke('gold', 'a')

    def test_inputs_bind_dataset_and_reject_unknown_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'data').mkdir()
            data = root/'data/tasks.jsonl'
            data.write_text(json.dumps({'instance_id': 'x'}) + '\n')
            gate = root/'gate.json'
            gate.write_text(json.dumps({'dataset_sha256': runner.digest(data)}))
            args = Namespace(base=root, gate=gate, ids=['bad'])
            with self.assertRaisesRegex(ValueError, 'Unknown selected'):
                runner.load_inputs(args)
            args.ids = ['x']
            self.assertEqual(len(runner.load_inputs(args)[0]), 1)
            gate.write_text(json.dumps({'dataset_sha256': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                runner.load_inputs(args)
            data.write_text(data.read_text() * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                runner.load_inputs(args)

    def test_main_parses_explicit_worker_counts(self):
        arguments = ['runner', '--base', '.', '--repo', '.', '--run-dir', './run',
                     '--gate', './gate', '--task-runner', './task.py',
                     '--endpoints', 'a', 'b', 'c', '--workers', '4', '4', '4']
        with patch.object(runner.sys, 'argv', arguments), patch.object(runner, 'run') as run:
            runner.main()
        self.assertEqual(run.call_args[0][0].workers, [4, 4, 4])


if __name__ == '__main__':
    unittest.main()
