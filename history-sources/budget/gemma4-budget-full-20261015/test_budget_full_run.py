import importlib.util
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).with_name('budget_full_run.py')
SPEC = importlib.util.spec_from_file_location('budget_full_run', SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class BudgetRunTests(unittest.TestCase):
    def test_shards_cover_all_129_exactly_once(self):
        shards = [runner.select_shard(list(range(129)), i, 4) for i in range(4)]
        self.assertEqual(sorted(sum(shards, [])), list(range(129)))
        self.assertEqual([len(s) for s in shards], [33, 32, 32, 32])
        self.assertTrue(all(value % 4 == i for i, s in enumerate(shards) for value in s))

    def test_invalid_shard_rejected(self):
        for index, count in [(-1, 4), (4, 4), (0, 0)]:
            with self.assertRaises(ValueError):
                runner.select_shard([], index, count)

    def test_results_are_immutable_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = {'instance_id': 'org__repo-1', 'status': 'evaluated', 'resolved': False,
                      'agent_patch': '', 'test_output': '', 'test_exit_code': 1, 'duration_seconds': 1.0}
            runner.save_result(root, record)
            self.assertTrue(runner.has_result(root, record['instance_id']))
            with self.assertRaises(FileExistsError):
                runner.save_result(root, {**record, 'resolved': True})
            self.assertFalse(json.loads(runner.result_path(root, record['instance_id']).read_text())['resolved'])
            self.assertEqual(list(root.glob('*.tmp')), [])

    def test_corrupt_checkpoint_is_never_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner.result_path(root, 'task').write_text('{broken')
            with self.assertRaises(ValueError):
                runner.has_result(root, 'task')

    def test_external_task_id_cannot_escape_result_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            p = runner.result_path(Path(temp), '../../outside')
            self.assertEqual(p.parent, Path(temp))

    def test_src_layout_precedes_installed_package_and_stays_in_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / 'workspace'
            workspace.mkdir()
            self.assertEqual(runner.workspace_pythonpath(workspace), str(workspace))
            (workspace / 'src').mkdir()
            self.assertEqual(runner.workspace_pythonpath(workspace),
                             str(workspace / 'src') + ':' + str(workspace))
            (workspace / 'src').rmdir()
            (workspace / 'src').symlink_to('/usr')
            with self.assertRaises(ValueError):
                runner.workspace_pythonpath(workspace)

    def test_valid_json_with_missing_fields_is_not_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner.result_path(root, 'task').write_text(json.dumps(
                {'instance_id': 'task', 'status': 'evaluated'}))
            with self.assertRaises(ValueError):
                runner.has_result(root, 'task')

    def test_verification_collection_is_candidate_not_automatic_infra(self):
        data = {'test_exit_code': 2, 'test_output': 'ERROR collecting test_x.py\nModuleNotFoundError: No module named xyz'}
        diagnostic = runner.verification_diagnostic(data)
        self.assertTrue(diagnostic['suspected_environment'])
        self.assertIn('collection_error', diagnostic['categories'])
        self.assertEqual(runner.classify_result(data), 'evaluated')
        data['test_output'] = "python: No module named pytest"
        self.assertEqual(runner.classify_result(data), 'infrastructure_error')

    def test_resumption_fingerprint_and_completed_provenance(self):
        state = {'fingerprint': {'dataset': 'abc'}, 'assigned_ids': ['one'], 'status': 'started'}
        previous = {**state, 'gpu_uuid': 'GPU-one', 'started_at': 'yesterday'}
        merged = runner.resume_state(state, previous)
        self.assertEqual(merged['gpu_uuid'], 'GPU-one')
        self.assertEqual(merged['started_at'], 'yesterday')
        with self.assertRaises(ValueError):
            runner.resume_state({**state, 'fingerprint': {'dataset': 'changed'}}, previous)
        records = [{'status': 'evaluated', 'resolved': True, 'verification_diagnostic': {}},
                   {'status': 'infrastructure_error', 'resolved': False,
                    'verification_diagnostic': {'suspected_environment': True}}]
        totals = runner.result_counts(records, total_dataset=129)
        self.assertEqual(totals['dataset_count'], 129)
        self.assertEqual(totals['resolved'], 1)
        self.assertEqual(totals['suspected_environment_verifications'], 1)

    def test_cleanup_only_acts_on_captured_child_identity(self):
        events = []
        class Child:
            pid = 101
            def create_time(self): return 1.0
            def is_running(self): return True
            def terminate(self): events.append('terminate child')
            def kill(self): events.append('kill child')
        child = Child()
        class API:
            @staticmethod
            def Process(pid):
                return SimpleNamespace(children=lambda recursive: [child])
            @staticmethod
            def wait_procs(processes, timeout): return [], processes
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / 'temporary.log'
            log.write_text('model log')
            server = SimpleNamespace(log_path=str(log), process=SimpleNamespace(pid=100),
                                     stop=lambda: events.append('stop parent'))
            result = runner.stop_owned_server(server, root, API)
            self.assertEqual(events, ['stop parent', 'terminate child', 'kill child'])
            self.assertEqual((root / 'server.log').read_text(), 'model log')
            self.assertEqual(result['captured_child_pids'], [101])

    def test_infrastructure_failures_not_counted_as_unsolved(self):
        self.assertEqual(runner.classify_result({'error_message': 'Snapshot file not found: /x', 'test_exit_code': -1}), 'infrastructure_error')
        self.assertEqual(runner.classify_result({'error_message': 'Agent exceeded time budget', 'test_exit_code': -1}), 'evaluated')
        self.assertEqual(runner.classify_result({'error_message': None, 'test_exit_code': 1}), 'evaluated')

    def test_evaluator_results_and_exceptions_both_persist(self):
        class Evaluator:
            async def evaluate_task(self, task, **kwargs):
                if task.instance_id == 'broken':
                    raise RuntimeError('connection refused')
                data = {'resolved': True, 'agent_patch': 'diff', 'test_output': 'PASS',
                        'test_exit_code': 0, 'trace_path': '/trace/task.json'}
                return SimpleNamespace(model_dump=lambda **kw: data, trace=None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = [SimpleNamespace(instance_id=name, repo='org/repo') for name in ['ok', 'broken']]
            asyncio.run(runner.evaluate_all(Evaluator(), tasks, root, {}, root / 'summary.json'))
            first = json.loads(runner.result_path(root, 'ok').read_text())
            second = json.loads(runner.result_path(root, 'broken').read_text())
            self.assertEqual(first['agent_patch'], 'diff')
            self.assertEqual(first['trace_path'], '/trace/task.json')
            self.assertTrue(first['resolved'])
            self.assertEqual(second['status'], 'infrastructure_error')
            self.assertIsNone(second['resolved'])
            self.assertIn('RuntimeError', second['exception_traceback'])


if __name__ == '__main__':
    unittest.main()
