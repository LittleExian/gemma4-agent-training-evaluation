import json
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gemmacodeagent_lora246_validation_pipeline as pipeline


class ValidationPipelineTests(unittest.TestCase):
    def fixture(self, root):
        ids = ['heldout-'+str(i) for i in range(20)]
        selection = root/'selection.json'; selection.write_text(json.dumps({'tasks': ids}))
        dataset = root/'tasks.jsonl'
        dataset.write_text('\n'.join(json.dumps({'instance_id': identifier}) for identifier in ids)+'\n')
        artifact = root/'snapshot'; artifact.write_text('frozen snapshot')
        gate = root/'gate.json'
        gate.write_text(json.dumps({'passed': True, 'selection_sha256': pipeline.digest(selection),
            'dataset_sha256': pipeline.digest(dataset), 'tasks': {identifier: {
                'eligible': True, 'gold_passed': True, 'empty_passed': False} for identifier in ids},
            'artifact_sha256': {str(artifact): pipeline.digest(artifact)}}))
        return selection, dataset, gate, ids

    def test_all_twenty_require_bound_successful_gold_and_failing_empty(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selection, dataset, gate, ids = self.fixture(root)
            self.assertEqual(pipeline.validate_inputs(selection, dataset, gate), ids)
            value = json.loads(gate.read_text()); value['tasks'][ids[0]]['eligible'] = False
            gate.write_text(json.dumps(value))
            with self.assertRaises(ValueError): pipeline.validate_inputs(selection, dataset, gate)

    def test_missing_tasks_corrupt_artifacts_or_dataset_block_all_dispatch(self):
        for fault in ('missing', 'artifact', 'dataset'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as td:
                root = Path(td); selection, dataset, gate, ids = self.fixture(root)
                if fault == 'missing':
                    selection.write_text(json.dumps({'tasks': ids[:-1]}))
                elif fault == 'artifact':
                    (root/'snapshot').write_text('drift')
                else:
                    dataset.write_text(dataset.read_text()+dataset.read_text().splitlines()[0]+'\n')
                with self.assertRaises(ValueError): pipeline.validate_inputs(selection, dataset, gate)

    def test_kaggle_blocked_never_acquires_or_restarts_a_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'pipeline-status.json'; path.write_text('{"status":"blocked","error":"control failed"}')
            with self.assertRaisesRegex(RuntimeError, 'Kaggle.*blocked'):
                pipeline.predecessor_ready(path)

    def test_predecessor_needs_completed_full_and_no_cleanup_residue(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'pipeline-status.json'
            state = {'status': 'running'}; path.write_text(json.dumps(state))
            self.assertFalse(pipeline.predecessor_ready(path))
            state = {'status': 'completed', 'models': [], 'reports': {
                'pilot16': {'passed': True, 'total': 16}, 'full': {'passed': True, 'total': 129}},
                'cleanup': [{'terminated': [], 'killed': []} for _ in range(2)]}
            path.write_text(json.dumps(state))
            with patch.object(pipeline, 'verify_predecessor_services', return_value=True):
                self.assertTrue(pipeline.predecessor_ready(path))
            path.write_text(json.dumps({**state, 'cleanup': [{'cleanup_error': 'owned pid remains'}]}))
            with self.assertRaises(RuntimeError): pipeline.predecessor_ready(path)

    def test_result_identity_verification_and_normal_output_are_enforced(self):
        row = {'instance_id': 't', 'status': 'completed', 'resolved': False, 'normal_output': True,
               'verification': {'test_exit_code': 1}}
        pipeline.validate_result(row, 't')
        pipeline.validate_result({**row, 'normal_output': False, 'agent_error_kind': 'tool_budget_exceeded'}, 't')
        pipeline.validate_result({**row, 'normal_output': False, 'status': 'agent_timeout'}, 't')
        pipeline.validate_result({**row, 'normal_output': False, 'verification': {
            'test_exit_code': 2, 'passed': 0, 'failed': 0, 'errors': 1,
            'expected_tests_covered': False, 'failure_reason': 'candidate_import_failure'}}, 't')
        pipeline.validate_result({**row, 'verification': {
            'test_exit_code': 124, 'passed': 0, 'failed': 0, 'errors': 0,
            'expected_tests_covered': False, 'failure_reason': 'candidate_test_timeout'}}, 't')
        for change in ({'instance_id': 'other'}, {'normal_output': None}, {'status': 'control_failed'},
                       {'resolved': True}, {'verification': {}}, {'infra_error': True},
                       {'verification': {'test_exit_code': False}},
                       {'verification': {'test_exit_code': 2, 'failed': -1}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                pipeline.validate_result({**row, **change}, 't')

    def test_only_eight_slots_on_cards_zero_and_two(self):
        self.assertEqual(pipeline.GPUS, (0, 2))
        self.assertEqual(pipeline.PORTS, (8140, 8142))
        self.assertEqual(pipeline.endpoint_slots(), ['http://127.0.0.1:8140/v1']*4+['http://127.0.0.1:8142/v1']*4)

    def test_invocation_has_watchdog_and_never_uses_shell(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def checked(command, log, environment, models, timeout):
                self.assertEqual(timeout, 1800)
                output = Path(command[command.index('--output')+1])
                pipeline.atomic_json(output/'result.json', {'instance_id': 'task', 'status': 'completed',
                    'normal_output': True, 'resolved': True, 'verification': {'test_exit_code': 0,
                        'passed': 3, 'failed': 0, 'errors': 0, 'expected_tests_covered': True}})
            args = SimpleNamespace(root=root, task_runner=root/'task.py', run_dir=root/'run')
            with patch.object(pipeline, 'checked_process', side_effect=checked):
                row = pipeline.invoke_task(args, 'task', 'http://127.0.0.1:8140/v1', [], {})
            self.assertTrue(row['resolved'])
            self.assertEqual(row['result_sha256'], pipeline.digest(Path(row['result_file'])))

    def test_real_child_creates_fresh_output_while_controller_logs_stay_outside(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); task = root/'task.py'
            task.write_text("import argparse,json\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser()\n"
                "for n in ('task-id','endpoint','output','root'):p.add_argument('--'+n,required=True)\n"
                "a=p.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=False)\n"
                "r={'instance_id':a.task_id,'status':'completed','normal_output':False,'resolved':False,"
                "'verification':{'test_exit_code':2,'passed':0,'failed':0,'errors':1,'expected_tests_covered':False}}\n"
                "(o/'result.json').write_text(json.dumps(r))\n")
            args = SimpleNamespace(root=root, task_runner=task, run_dir=root/'run')
            def checked(command, log, environment, models, timeout):
                output = Path(command[command.index('--output')+1])
                self.assertFalse(output.exists())
                self.assertNotEqual(log.parent, output)
                with log.open('w') as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=stream, timeout=timeout)
            with patch.object(pipeline, 'checked_process', side_effect=checked):
                row = pipeline.invoke_task(args, 'fresh-task', 'http://127.0.0.1:8140/v1', [], {})
                repeated = pipeline.invoke_task(args, 'fresh-task', 'http://127.0.0.1:8140/v1', [], {})
            self.assertEqual(row['status'], 'completed', row.get('error'))
            self.assertFalse(row['resolved'])
            self.assertEqual(repeated['status'], 'infra_error')
            self.assertTrue(Path(row['result_file']).exists())
            self.assertTrue(list((args.run_dir/'task-logs').glob('*/task.log')))

    def test_failed_invocation_becomes_committed_infrastructure_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); args = SimpleNamespace(root=root, task_runner=root/'task.py', run_dir=root/'run')
            with patch.object(pipeline, 'checked_process', side_effect=TimeoutError('watchdog')):
                row = pipeline.invoke_task(args, 'task', 'endpoint', [], {})
            self.assertEqual(row['status'], 'infra_error')
            self.assertIn('watchdog', row['error'])

    def test_trusted_success_requires_tests_and_declared_coverage(self):
        record = {'instance_id': 't', 'status': 'completed', 'normal_output': False, 'resolved': True,
            'verification': {'test_exit_code': 0, 'passed': 2, 'failed': 0, 'errors': 0, 'expected_tests_covered': True}}
        pipeline.validate_result(record, 't')
        for change in ({'passed': 0}, {'failed': 1}, {'errors': 1}, {'expected_tests_covered': False}):
            with self.assertRaises(ValueError):
                pipeline.validate_result({**record, 'verification': {**record['verification'], **change}}, 't')

    def test_scheduler_scores_legitimate_model_budget_failure_without_stopping(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(run_dir=Path(td)); ids = ['t'+str(i) for i in range(20)]
            def invoke(args, identifier, endpoint, models, environment):
                return {'instance_id': identifier, 'status': 'completed', 'normal_output': False,
                    'resolved': False, 'agent_error_kind': 'context_overflow', 'verification': {'test_exit_code': 1}}
            with patch.object(pipeline, 'invoke_task', side_effect=invoke):
                report = pipeline.evaluate(args, ids, [], {})
            self.assertEqual(report['completed'], 20)
            self.assertEqual(report['normal_outputs'], 0)
            self.assertEqual(report['infrastructure_errors'], 0)

    def test_gpu_probe_allows_verified_foreign_processes_with_capacity_without_killing(self):
        fake = Mock()
        fake.NoSuchProcess = ProcessLookupError
        fake.AccessDenied = PermissionError
        fake.Process.return_value.uids.return_value = SimpleNamespace(real=1006)
        fake.Process.return_value.create_time.return_value = 123.
        def output(command, **kwargs):
            if '--query-gpu=index,uuid,name,memory.total,memory.free' in command:
                return '0, GPU-a, NVIDIA RTX PRO 6000, 97278, 94000\n2, GPU-b, NVIDIA RTX PRO 6000, 97278, 90000\n'
            return 'GPU-a, 100, other-service, 3600\nGPU-b, 200, other-service, 7800\nGPU-unused, 300, no-touch, 100\n'
        report = pipeline.gpu_readiness(output, fake)
        self.assertTrue(report['ready'])
        self.assertEqual(len(report['existing_compute_processes']), 2)
        self.assertEqual(report['existing_compute_processes'][0]['uid'], 1006)
        fake.Process.return_value.terminate.assert_not_called()
        fake.Process.return_value.kill.assert_not_called()
        fake.Process.side_effect = PermissionError('other user')
        self.assertFalse(pipeline.gpu_readiness(output, fake)['ready'])
        fake.Process.side_effect = ProcessLookupError('exited')
        self.assertTrue(pipeline.gpu_readiness(output, fake)['ready'])
        def insufficient(command, **kwargs):
            return output(command, **kwargs).replace('90000', '60000')
        self.assertFalse(pipeline.gpu_readiness(insufficient, fake)['ready'])
        def wrong(command, **kwargs):
            return output(command, **kwargs).replace('2, GPU-b', '3, GPU-b')
        with self.assertRaises(RuntimeError): pipeline.gpu_readiness(wrong, fake)

    def test_cleanup_checks_only_recorded_kaggle_identities_and_never_kills(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = Mock(); fake.NoSuchProcess = ProcessLookupError; fake.STATUS_ZOMBIE = 'zombie'
            fake.Process.side_effect = ProcessLookupError('gone')
            for scope in ('pilot16', 'full'):
                services = root/scope/'services'
                pipeline.atomic_json(services/'cleanup-secondary.json', [{'terminated': [], 'killed': []}]*2)
                for gpu, port in zip(pipeline.GPUS, pipeline.PORTS):
                    pipeline.atomic_json(services/('gpu-'+str(gpu)+'-123')/'process.json', {
                        'physical_gpu_index_at_start': gpu, 'port': port, 'models': [pipeline.ARM],
                        'pid': gpu+1, 'uid': 1005, 'create_time': 123., 'launch_boottime': 100.})
            with patch.object(pipeline.os, 'getuid', return_value=1005):
                self.assertTrue(pipeline.verify_predecessor_services(root, fake))
                fake.Process.side_effect = None
                fake.Process.return_value.create_time.return_value = 123.
                fake.Process.return_value.uids.return_value = SimpleNamespace(real=1005)
                fake.Process.return_value.status.return_value = 'running'
                self.assertFalse(pipeline.verify_predecessor_services(root, fake))
                fake.Process.return_value.create_time.return_value = 124.
                self.assertTrue(pipeline.verify_predecessor_services(root, fake))
            fake.Process.return_value.terminate.assert_not_called()
            fake.Process.return_value.kill.assert_not_called()

    def test_host_memory_uses_available_ram_not_swap_and_scales_only_explicit_workers(self):
        memory = 'MemTotal: 197132288 kB\nMemAvailable: 42991616 kB\nSwapTotal: 33554432 kB\nSwapFree: 23068672 kB\n'
        report = pipeline.host_memory_readiness(4, memory)
        self.assertFalse(report['ready'])
        self.assertEqual(report['required_gib'], 48)
        self.assertEqual(report['swap_used_gib'], 10)
        self.assertTrue(pipeline.host_memory_readiness(2, memory)['ready'])
        self.assertEqual(pipeline.host_memory_readiness(2, memory)['required_gib'], 32)
        self.assertEqual(len(pipeline.endpoint_slots(2)), 4)
        self.assertEqual(len(pipeline.endpoint_slots(1)), 2)
        with self.assertRaises(ValueError): pipeline.host_memory_readiness(3, memory)

    def test_readiness_waits_for_inputs_predecessor_then_gpu_capacity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selection, dataset, gate, _ = self.fixture(root)
            files = [root/'manifest', root/'deployment', root/'task.py']
            for path in files: path.write_text('ready')
            args = SimpleNamespace(selection=selection, dataset=dataset, gate=gate,
                manifest=files[0], deployment_gate=files[1], task_runner=files[2],
                wait_for=root/'prior', wait_timeout=100.)
            save = Mock()
            with patch.object(pipeline, 'predecessor_ready', side_effect=[False, True, True]), \
                 patch.object(pipeline, 'gpu_readiness', side_effect=[{'ready': False}, {'ready': True}]), \
                 patch.object(pipeline.time, 'sleep'), \
                 patch.object(pipeline, 'host_memory_readiness', return_value={'ready': True}):
                self.assertTrue(pipeline.wait_until_ready(args, save)['ready'])
            self.assertEqual([call.kwargs['phase'] for call in save.call_args_list],
                             ['waiting_kaggle_cleanup', 'memory_ready', 'waiting_gpu', 'memory_ready', 'gpu_ready'])
            files[0].unlink(); save.reset_mock()
            with patch.object(pipeline, 'predecessor_ready', return_value=False), \
                 patch.object(pipeline.time, 'monotonic', side_effect=[0., 2.]):
                args.wait_timeout = 1.
                with self.assertRaises(TimeoutError): pipeline.wait_until_ready(args, save)
            self.assertEqual(save.call_args.kwargs['phase'], 'waiting_inputs')

    def test_protocol_binds_completed_source_and_all_validation_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selection, dataset, gate, _ = self.fixture(root)
            source = root/'src/task.py'; source.parent.mkdir(); source.write_text('trusted task')
            service = root/'main'; serving_source = service/'src/loader.py'
            serving_source.parent.mkdir(parents=True); serving_source.write_text('loader')
            prior = service/'runs/main/pipeline-status.json'
            pipeline.atomic_json(prior, {'status': 'completed'})
            model = service/'models'/pipeline.serving.BASE_ALIAS
            model.mkdir(parents=True); (model/'model.safetensors').write_text('base'); (model/'config.json').write_text('config')
            pipeline.atomic_json(prior.parent/'protocol.json', {
                'source_sha256': {'src/loader.py': pipeline.digest(serving_source)},
                'base_weight_sha256': pipeline.serving.BASE_SHA,
                'base_metadata_sha256': {'config.json': pipeline.digest(model/'config.json')}})
            for name in ('manifest', 'deployment'): (root/name).write_text(name)
            args = SimpleNamespace(root=root, service_root=service, wait_for=prior, selection=selection,
                dataset=dataset, gate=gate, manifest=root/'manifest', deployment_gate=root/'deployment', task_runner=source)
            def streaming(path):
                return pipeline.serving.BASE_SHA if path.name == 'model.safetensors' else pipeline.digest(path)
            with patch.object(pipeline.subprocess, 'check_output', return_value='frozen'), \
                 patch.object(pipeline, 'streaming_digest', side_effect=streaming):
                report = pipeline.build_protocol(args)
            self.assertEqual(report['total_slots'], 8)
            self.assertEqual(report['gpus'], [0, 2])
            self.assertEqual(report['cpu_gate_sha256'], pipeline.digest(gate))
            serving_source.write_text('changed')
            with self.assertRaises(ValueError): pipeline.build_protocol(args)

    def test_run_ready_always_cleans_only_new_models_and_preserves_primary_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selection, dataset, gate, ids = self.fixture(root)
            manifest = root/'manifest'; manifest.write_text('{"adapter":{"source_path":"/checkpoint246/adapter"}}')
            args = SimpleNamespace(root=root, service_root=root/'main', selection=selection, dataset=dataset,
                gate=gate, manifest=manifest, deployment_gate=root/'deploy', run_dir=root/'runs/new')
            save = Mock()
            with patch.object(pipeline.serving, 'validate_package'), \
                 patch.object(pipeline.serving, 'validate_deployment_gate'), \
                 patch.object(pipeline.serving, 'verify_runtime'), \
                 patch.object(pipeline, 'build_protocol', return_value={'bound': True}), \
                 patch.object(pipeline.serving, 'start_models', return_value=([{'pid': 1}], 123.)) as start, \
                 patch.object(pipeline, 'evaluate', side_effect=RuntimeError('agent infrastructure')) as evaluate, \
                 patch.object(pipeline, 'cleanup_secondary', return_value=[{'cleanup_error': 'secondary'}]) as cleanup, \
                 patch.object(pipeline, 'host_memory_readiness', return_value={'ready': True}):
                with self.assertRaisesRegex(RuntimeError, 'agent infrastructure'):
                    pipeline.run_ready(args, save)
            self.assertEqual(start.call_args.args[1], Path('/checkpoint246/adapter'))
            self.assertEqual(cleanup.call_args.args[0], [{'pid': 1}])
            environment = evaluate.call_args.args[3]
            self.assertEqual(environment['G4_VALIDATION_ROOT'], str(root))
            self.assertEqual(environment['OMP_NUM_THREADS'], '1')

    def test_final_launch_memory_drop_requeues_and_rechecks_protocol_before_start(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selection, dataset, gate, _ = self.fixture(root)
            manifest = root/'manifest'; manifest.write_text('{"adapter":{"source_path":"/checkpoint246/adapter"}}')
            args = SimpleNamespace(root=root, service_root=root/'main', selection=selection, dataset=dataset,
                gate=gate, manifest=manifest, deployment_gate=root/'deploy', run_dir=root/'runs/new')
            save = Mock(); events = []
            def ready(*args): events.append('wait'); return {'ready': True}
            def start(*args): events.append('start'); return ([{'pid': 1}], 123.)
            with patch.object(pipeline.serving, 'validate_package') as package, \
                 patch.object(pipeline.serving, 'validate_deployment_gate'), \
                 patch.object(pipeline.serving, 'verify_runtime'), \
                 patch.object(pipeline, 'build_protocol', return_value={'bound': True}) as protocol, \
                 patch.object(pipeline, 'host_memory_readiness', return_value={'ready': False}), \
                 patch.object(pipeline, 'wait_until_ready', side_effect=ready), \
                 patch.object(pipeline.serving, 'start_models', side_effect=start), \
                 patch.object(pipeline, 'evaluate', return_value={'completed': 20}), \
                 patch.object(pipeline, 'cleanup_secondary', return_value=[]):
                pipeline.run_ready(args, save)
            self.assertEqual(events, ['wait', 'start'])
            self.assertEqual(package.call_count, 2)
            self.assertEqual(protocol.call_count, 2)
            self.assertIn('waiting_host_memory', [call.kwargs.get('phase') for call in save.call_args_list])

    def test_recorded_kaggle_child_residue_waits_without_terminating_anything(self):
        fake = Mock(); fake.NoSuchProcess = ProcessLookupError; fake.STATUS_ZOMBIE = 'zombie'
        fake.Process.return_value.uids.return_value = SimpleNamespace(real=1005)
        fake.Process.return_value.create_time.return_value = 110.
        fake.Process.return_value.status.return_value = 'running'
        cleanup = [{'terminated': [99], 'killed': []}]
        with patch.object(pipeline.os, 'getuid', return_value=1005):
            self.assertTrue(pipeline._cleanup_targets_alive(cleanup, [{'create_time': 100.}], fake))
            fake.Process.return_value.uids.return_value = SimpleNamespace(real=1006)
            self.assertFalse(pipeline._cleanup_targets_alive(cleanup, [{'create_time': 100.}], fake))
            fake.Process.side_effect = ProcessLookupError('gone')
            self.assertFalse(pipeline._cleanup_targets_alive(cleanup, [{'create_time': 100.}], fake))
        fake.Process.return_value.kill.assert_not_called()
        fake.Process.return_value.terminate.assert_not_called()

    def test_waiting_host_memory_is_explicit_and_queue_deadline_is_not_reset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); selection, dataset, gate, _ = self.fixture(root)
            args = SimpleNamespace(selection=selection, dataset=dataset, gate=gate, manifest=root/'manifest',
                deployment_gate=root/'deployment', task_runner=root/'task.py', wait_for=root/'prior',
                wait_timeout=1., run_dir=root/'run')
            for path in (args.manifest, args.deployment_gate, args.task_runner): path.write_text('ready')
            save = Mock()
            with patch.object(pipeline, 'predecessor_ready', return_value=True), \
                 patch.object(pipeline, 'host_memory_readiness', return_value={'ready': False}), \
                 patch.object(pipeline.time, 'monotonic', side_effect=[0., 2.]):
                with self.assertRaises(TimeoutError): pipeline.wait_until_ready(args, save)
            self.assertEqual(save.call_args.kwargs['phase'], 'waiting_host_memory')
            pipeline.atomic_json(args.run_dir/'pipeline-status.json', {'started_unix': 10.})
            with patch.object(pipeline.time, 'time', return_value=20.):
                with self.assertRaises(TimeoutError): pipeline.wait_until_ready(args, save)

    def test_queue_lifecycle_and_failure_are_atomically_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for fail in (False, True):
                args = SimpleNamespace(root=root, service_root=root/'main', run_dir=root/'runs'/str(fail), wait_timeout=10.)
                with patch.object(pipeline, 'wait_until_ready', return_value={'ready': True}), \
                     patch.object(pipeline, 'run_ready', side_effect=RuntimeError('gate failed') if fail else None):
                    if fail:
                        with self.assertRaises(RuntimeError): pipeline.run(args)
                    else:
                        pipeline.run(args)
                state = json.loads((args.run_dir/'pipeline-status.json').read_text())
                self.assertEqual(state['status'], 'blocked' if fail else 'completed')
                with self.assertRaises(ValueError): pipeline.run(args)
            args.run_dir = root/'outside'
            with self.assertRaises(ValueError): pipeline.run(args)

    def test_scheduler_reports_twenty_and_stops_after_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); args = SimpleNamespace(run_dir=root)
            ids = ['t'+str(i) for i in range(20)]
            def invoke(args, identifier, endpoint, models, environment):
                return {'instance_id': identifier, 'status': 'completed', 'normal_output': True, 'resolved': False}
            with patch.object(pipeline, 'invoke_task', side_effect=invoke):
                report = pipeline.evaluate(args, ids, [], {})
            self.assertEqual(report['completed'], 20)
            self.assertEqual(report['normal_outputs'], 20)
            self.assertEqual(len(list((root/'results').glob('*.json'))), 20)
            def failed(args, identifier, endpoint, models, environment):
                return {'instance_id': identifier, 'status': 'infra_error', 'normal_output': False, 'resolved': False}
            args = SimpleNamespace(run_dir=root/'failed')
            args.run_dir.mkdir()
            with patch.object(pipeline, 'invoke_task', side_effect=failed):
                with self.assertRaises(RuntimeError): pipeline.evaluate(args, ids, [], {})
            progress = json.loads((args.run_dir/'progress.json').read_text())
            self.assertEqual(progress['status'], 'blocked')
            self.assertLessEqual(progress['completed'], 8)


if __name__ == '__main__':
    unittest.main()
