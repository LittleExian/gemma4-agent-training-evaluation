import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gemmacodeagent_mm_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_all_pilots_required_before_full(self):
        statuses = {arm: {'passed': True} for arm in pipeline.ARMS[:-1]}
        with self.assertRaises(RuntimeError):
            pipeline.require_pilots(statuses)
        pipeline.require_pilots({arm: {'passed': True} for arm in pipeline.ARMS})

    def test_terminal_records_allow_failed_patch_but_not_infrastructure(self):
        rows = [{'instance_id': 't1', 'status': 'completed', 'resolved': False,
                 'validations': [{'infra_error': False, 'test_exit_code': 2}]}]
        pipeline.validate_records(rows, ['t1'])
        for status in ['infra_error', 'control_failed', 'preflight_excluded']:
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                pipeline.validate_records([{**rows[0], 'status': status}], ['t1'])

    def test_missing_or_duplicate_records_block(self):
        for rows in [[], [{'instance_id': 'a'}, {'instance_id': 'a'}]]:
            with self.assertRaises(RuntimeError):
                pipeline.validate_records(rows, ['a'])

    def test_identity_checked_before_cleanup(self):
        class FakeProcess:
            def create_time(self): return 11
            def uids(self): return type('Uid', (), {'real': 123})()
            def status(self): return 'running'
        class FakePsutil:
            STATUS_ZOMBIE = 'zombie'
            def Process(self, pid): return FakeProcess()
        with self.assertRaises(RuntimeError):
            pipeline.assert_identity({'pid': 1, 'create_time': 12, 'uid': 123}, FakePsutil())

    def test_protocol_reuse_refuses_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'protocol.json'
            pipeline.bind_json(path, {'value': 1})
            pipeline.bind_json(path, {'value': 1})
            with self.assertRaises(RuntimeError): pipeline.bind_json(path, {'value': 2})

    def run_mock_pipeline(self, root, fail_arm=None):
        experiment = root/'experiments/distributed_model_comparison'
        experiment.mkdir(parents=True)
        identifiers = ['task'+str(i) for i in range(129)]
        for filename, ids in [('pilot16.json', identifiers[:16]), ('full129.json', identifiers)]:
            (experiment/filename).write_text(json.dumps({'tasks': ids}))
        args = SimpleNamespace(root=root, run_dir=root/'runs/mock', base=root/'base', gate=root/'gate.json')
        calls = []
        def evaluate(*values):
            arm, scope = values[3:5]
            calls.append((scope, arm))
            if arm == fail_arm:
                raise RuntimeError('Infrastructure failure')
            return {'passed': True, 'resolved': 0, 'total': 16 if scope == 'pilot16' else 129}
        with patch.object(pipeline, 'build_protocol', return_value={'fixed': True}), \
                patch.object(pipeline.subprocess, 'run'), \
                patch.object(pipeline, 'wait_downloads', return_value={}), \
                patch.object(pipeline, 'start_models', return_value=([], 1)), \
                patch.object(pipeline, 'evaluate', side_effect=evaluate):
            if fail_arm:
                with self.assertRaisesRegex(RuntimeError, 'Infrastructure failure'):
                    pipeline.run(args)
            else:
                pipeline.run(args)
        return calls, args.run_dir

    def test_pipeline_e2e_all_seven_pilots_precede_any_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, run = self.run_mock_pipeline(Path(directory))
            self.assertEqual(calls, [(scope, arm) for scope in ('pilot16', 'full') for arm in pipeline.ARMS])
            self.assertEqual(json.loads((run/'pipeline-status.json').read_text())['status'], 'completed')
            self.assertEqual(set(json.loads((run/'all-seven-pilots-passed.json').read_text())), set(pipeline.ARMS))

    def test_pipeline_e2e_pilot_infrastructure_failure_blocks_all_full_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, run = self.run_mock_pipeline(Path(directory), fail_arm=pipeline.ARMS[2])
            self.assertEqual(calls, [('pilot16', arm) for arm in pipeline.ARMS[:3]])
            self.assertFalse((run/'all-seven-pilots-passed.json').exists())
            self.assertEqual(json.loads((run/'pipeline-status.json').read_text())['status'], 'blocked')

    def test_health_failure_releases_only_recorded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            own_record = {'pid': 123, 'uid': 456, 'create_time': 1}
            def checked(command, *args, **kwargs):
                if 'launch' in command:
                    state = Path(command[command.index('--state')+1])
                    (state/'process.json').write_text(json.dumps(own_record))
                else:
                    raise RuntimeError('Health probe failed')
            with patch.object(pipeline, 'checked_process', side_effect=checked), \
                    patch.object(pipeline, 'release_owned_model', return_value={}) as release:
                with self.assertRaisesRegex(RuntimeError, 'Health probe failed'):
                    pipeline.start_models(root, root, root, ('gemma4_kaggle',), {}, root/'services', {})
            self.assertEqual(release.call_count, 1)
            self.assertEqual(release.call_args.args[0], own_record)

    def test_checked_process_timeout_or_identity_failure_cleans_only_child(self):
        child = {'pid': 123, 'create_time': 5, 'uid': 456}
        model = {'pid': 999, 'create_time': 4, 'uid': 456}
        with tempfile.TemporaryDirectory() as directory:
            for failure in ('timeout', 'identity'):
                with self.subTest(failure=failure):
                    process = Mock(returncode=None)
                    process.poll.return_value = None
                    with patch.object(pipeline.subprocess, 'Popen', return_value=process), \
                            patch.object(pipeline, 'process_record', return_value=child), \
                            patch.object(pipeline, 'assert_identity', side_effect=RuntimeError('identity') if failure == 'identity' else None), \
                            patch.object(pipeline.time, 'time', side_effect=[10, 100]), \
                            patch.object(pipeline, 'release_owned_model') as release:
                        with self.assertRaises(TimeoutError if failure == 'timeout' else RuntimeError):
                            pipeline.checked_process(['owned-command'], Path(directory)/(failure+'.log'), {}, [model], timeout=1)
                        release.assert_called_once_with(child, 10)

    def test_checked_process_records_success_and_nonzero_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            for code in (0, 2):
                process = Mock(returncode=code)
                process.poll.return_value = code
                with patch.object(pipeline.subprocess, 'Popen', return_value=process), \
                        patch.object(pipeline, 'process_record', return_value={'pid': 123}), \
                        patch.object(pipeline, 'release_owned_model') as release:
                    log = Path(directory)/('exit'+str(code)+'.log')
                    if code:
                        with self.assertRaises(RuntimeError): pipeline.checked_process(['cmd'], log, {})
                        self.assertEqual(release.call_count, 1)
                    else:
                        pipeline.checked_process(['cmd'], log, {})
                        release.assert_not_called()
                    self.assertEqual(json.loads(log.with_suffix('.process.json').read_text())['command'], ['cmd'])

    def test_completed_download_still_invokes_pinned_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'models.json').write_text(json.dumps({'qwen38_base': {'directory': 'qwen38_base'}}))
            cached = root/'cache/qwen38_base'
            cached.mkdir(parents=True)
            (cached/'DOWNLOAD_COMPLETE.json').write_text('{}')
            with patch.object(pipeline, 'checked_process') as checked:
                pipeline.wait_downloads(root, root, root/'cache', ['qwen38_base'], {}, root)
            command = checked.call_args.args[0]
            self.assertIn('download', command)
            self.assertEqual(command[command.index('--arm')+1], 'qwen38_base')

    def test_original_weights_are_rehashed_and_corruption_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'models.json').write_text('{}')
            model = root/'models/gemma-4-31b-it-qat-w4a16-ct'
            model.mkdir(parents=True)
            (model/'config.json').write_text('{}')
            (model/'weights.safetensors').write_bytes(b'weights')
            expected = '1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20'
            with patch('gemmacodeagent_mm_server.digest', return_value=expected) as digest:
                pipeline.wait_downloads(root, root, root, ['gemma4_kaggle'], {}, root)
                digest.assert_called_once_with(model/'weights.safetensors')
            self.assertTrue((root/'original-model-verified.json').exists())
            with patch('gemmacodeagent_mm_server.digest', return_value='corrupt'):
                with self.assertRaisesRegex(RuntimeError, 'checkpoint hash changed'):
                    pipeline.wait_downloads(root, root, root, ['gemma4_kaggle'], {}, root)

    def test_evaluate_writes_compatibility_only_for_complete_noninfra_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {'instance_id': 'task1', 'status': 'completed', 'resolved': False,
                      'validations': [{'infra_error': False, 'test_exit_code': 2}]}
            def checked(command, *args):
                output = Path(command[command.index('--run-dir')+1])
                (output/'results').mkdir(exist_ok=True)
                (output/'results/one.json').write_text(json.dumps(result))
            with patch.object(pipeline, 'checked_process', side_effect=checked):
                report = pipeline.evaluate(root, root, root/'gate', 'arm', 'pilot16', ['task1'], root, [], {}, root/'protocol')
                self.assertTrue(report['passed'])
                self.assertEqual(report['resolved'], 0)
                result = {**result, 'validations': [{'infra_error': True}]}
                with self.assertRaisesRegex(RuntimeError, 'infrastructure failure'):
                    pipeline.evaluate(root, root, root/'gate', 'badarm', 'pilot16', ['task1'], root, [], {}, root/'protocol')
                self.assertFalse((root/'pilot16/badarm/compatibility.json').exists())

    def test_protocol_binds_runtime_submission_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ['src/runner.py', 'src/test_runner.py', 'submissions/mm15_6000/eval_config.yaml',
                     'experiments/distributed_model_comparison/models.json', 'data/tasks.jsonl',
                     'gate.json', 'environment/installed_code_hashes.json',
                     'models/gemma-4-31b-it-qat-w4a16-ct/config.json']
            for name in paths:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(name)
            with patch.object(pipeline.subprocess, 'check_output', return_value='vllm==0.19.1\n'):
                before = pipeline.build_protocol(root, root, root/'gate.json', root/'experiments/distributed_model_comparison')
                (root/'src/runner.py').write_text('changed')
                after = pipeline.build_protocol(root, root, root/'gate.json', root/'experiments/distributed_model_comparison')
            self.assertEqual(before['runtime_freeze'], 'vllm==0.19.1\n')
            self.assertEqual(before['critical_source_manifest_sha256'], pipeline.digest(root/'environment/installed_code_hashes.json'))
            self.assertNotEqual(before['source_sha256']['src/runner.py'], after['source_sha256']['src/runner.py'])
            self.assertIn('submissions/mm15_6000/eval_config.yaml', before['source_sha256'])
            self.assertNotIn('src/test_runner.py', before['source_sha256'])


if __name__ == '__main__': unittest.main()
