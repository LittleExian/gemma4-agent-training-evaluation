import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import gemmacodeagent_full_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def config(self, root):
        return {'models': [{'gpu': gpu, 'port': 8200+i, 'slots': 4, 'memory_utilization': .65,
                            'state_dir': str(root/f'model-{gpu}')} for i, gpu in enumerate([0, 2, 3])],
                'runner': '/repo/full_run.py', 'runner_args': ['--config', '/run/config.json']}

    def test_config_requires_three_distinct_pros_and_unique_ports_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            supervisor.validate_config(config, root)
            for key in ['gpu', 'port', 'state_dir']:
                changed = {**config, 'models': [*config['models'][:2], {**config['models'][2], key: config['models'][0][key]}]}
                with self.assertRaises(ValueError):
                    supervisor.validate_config(changed, root)
            with self.assertRaises(ValueError):
                supervisor.validate_config({**config, 'models': config['models'][:2]}, root)
            with self.assertRaises(ValueError):
                supervisor.validate_config({**config, 'runner': 'relative.py'}, root)

    def test_config_fingerprint_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor.bind_config(root, {'x': 1})
            supervisor.bind_config(root, {'x': 1})
            with self.assertRaises(RuntimeError):
                supervisor.bind_config(root, {'x': 2})

    def test_process_identity_check_rejects_reuse_and_foreign_uid(self):
        fake_psutil = Mock(NoSuchProcess=ProcessLookupError, STATUS_ZOMBIE='zombie')
        process = fake_psutil.Process.return_value
        process.create_time.return_value = 10
        process.uids.return_value.real = 501
        with patch.dict('sys.modules', {'psutil': fake_psutil}), patch.object(supervisor.os, 'getuid', return_value=501):
            supervisor.assert_live({'pid': 1, 'create_time': 10})
            process.create_time.return_value = 11
            with self.assertRaises(RuntimeError):
                supervisor.assert_live({'pid': 1, 'create_time': 10})
            process.create_time.return_value = 10
            process.uids.return_value.real = 999
            with self.assertRaises(RuntimeError):
                supervisor.assert_live({'pid': 1, 'create_time': 10})

    def test_mock_health_checks_process_on_every_retry(self):
        response = Mock(status_code=200)
        response.json.return_value = {'data': [{'id': 'gemma'}]}
        requests = Mock()
        requests.RequestException = OSError
        requests.get.side_effect = [OSError(), response]
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('sys.modules', {'requests': requests}), \
                patch.object(supervisor, 'assert_live') as identity, \
                patch.object(supervisor.time, 'sleep'):
            supervisor.wait_healthy({'pid': 1, 'create_time': 10}, 8200, Path(directory))
            self.assertEqual(identity.call_count, 2)
            self.assertTrue((Path(directory)/'health.json').exists())

    def test_health_timeout_does_not_accept_unhealthy_port(self):
        requests = Mock(RequestException=OSError)
        requests.get.return_value.status_code = 503
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('sys.modules', {'requests': requests}), \
                patch.object(supervisor, 'assert_live'), patch.object(supervisor.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                supervisor.wait_healthy({'pid': 1, 'create_time': 10}, 8200, Path(directory))

    def test_cli_success_and_live_service_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root/'config.json'
            config_path.write_text(json.dumps(self.config(root)))
            fake_psutil = Mock()
            args = ['supervisor', '--repo', str(root), '--run-dir', str(root), '--config', str(config_path)]
            with patch('sys.argv', args), patch.dict('sys.modules', {'psutil': fake_psutil}), \
                    patch('gemmacodeagent_full_server.refuse_live_record') as refuse, \
                    patch.object(supervisor, 'orchestrate') as execute:
                supervisor.main()
                self.assertEqual(refuse.call_count, 3)
                execute.assert_called_once()
                refuse.side_effect = RuntimeError('already alive')
                with self.assertRaises(RuntimeError):
                    supervisor.main()

    def test_success_keeps_models_until_runner_returns_and_records_cleanup_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            calls = []
            def fake_subprocess(command, **kwargs):
                calls.append(command)
                if '--state-dir' in command:
                    state = Path(command[command.index('--state-dir')+1])
                    state.mkdir()
                    (state/'process.json').write_text(json.dumps({'pid': 55, 'create_time': 101}))
                if command[1] == '/repo/full_run.py':
                    self.assertEqual(cleanup.call_count, 0)
            with patch.object(supervisor, 'wait_healthy'), \
                    patch.object(supervisor, 'run_evaluation', side_effect=lambda *a: self.assertEqual(cleanup.call_count, 0)), \
                    patch.object(supervisor.subprocess, 'run', side_effect=fake_subprocess), \
                    patch.object(supervisor, 'release_owned_model', side_effect=[RuntimeError('changed'), {}, {}]) as cleanup, \
                    patch.object(supervisor.time, 'time', return_value=100):
                supervisor.orchestrate(root, root, config)
            status = json.loads((root/'supervisor-status.json').read_text())
            self.assertEqual(status['status'], 'completed')
            self.assertIn('changed', status['cleanup'][0]['error'])
            self.assertEqual(len(calls), 5)

    def test_orchestration_cleanup_runs_after_runner_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            def fake_subprocess(command, **kwargs):
                if '--state-dir' in command:
                    state = Path(command[command.index('--state-dir')+1])
                    state.mkdir()
                    (state/'process.json').write_text(json.dumps({'pid': len(command), 'create_time': 101}))
                if command[1] == '/repo/full_run.py':
                    raise RuntimeError('runner failed')
            with patch.object(supervisor, 'assert_live'), patch.object(supervisor, 'wait_healthy'), \
                    patch.object(supervisor, 'run_evaluation', side_effect=RuntimeError('runner failed')), \
                    patch.object(supervisor.subprocess, 'run', side_effect=fake_subprocess), \
                    patch.object(supervisor, 'release_owned_model', return_value={'terminated': []}) as cleanup, \
                    patch.object(supervisor.time, 'time', return_value=100):
                with self.assertRaises(RuntimeError):
                    supervisor.orchestrate(root, root, config)
            self.assertEqual(cleanup.call_count, 3)
            status = json.loads((root/'supervisor-status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(len(status['cleanup']), 3)

    def test_monitor_model_death_fails_without_waiting_for_runner(self):
        process = Mock()
        process.poll.return_value = None
        requests = Mock(RequestException=OSError)
        with patch.dict('sys.modules', {'requests': requests}), \
                patch.object(supervisor, 'assert_live', side_effect=RuntimeError('identity changed')):
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                supervisor.monitor_runner(process, [{'pid': 1}], [{'port': 8200}])
        requests.get.assert_not_called()

    def test_monitor_runner_finished_does_not_need_health_or_cleanup(self):
        process = Mock(returncode=0)
        process.poll.return_value = 0
        with patch.dict('sys.modules', {'requests': Mock()}), \
                patch.object(supervisor, 'assert_live') as identity:
            supervisor.monitor_runner(process, [{'pid': 1}], [{'port': 8200}])
        identity.assert_not_called()

    def test_runner_nonzero_exit_is_reported(self):
        process = Mock(returncode=2, args=['runner'])
        process.poll.return_value = 2
        with patch.dict('sys.modules', {'requests': Mock()}):
            with self.assertRaises(supervisor.subprocess.CalledProcessError):
                supervisor.monitor_runner(process, [], [])

    def test_monitor_transient_failure_reset_and_three_failures_failfast(self):
        requests = Mock(RequestException=OSError)
        requests.get.side_effect = [OSError(), Mock(status_code=503), Mock(status_code=200),
                                    OSError(), OSError(), OSError()]
        process = Mock()
        process.poll.return_value = None
        with patch.dict('sys.modules', {'requests': requests}), \
                patch.object(supervisor, 'assert_live'), patch.object(supervisor.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'three consecutive'):
                supervisor.monitor_runner(process, [{'pid': 1}], [{'port': 8200}])
        self.assertEqual(requests.get.call_count, 6)
        self.assertEqual(sleep.call_count, 5)

    def test_model_failure_cleans_only_recorded_runner(self):
        process = Mock(pid=55, args=['python', 'runner'])
        process.poll.return_value = None
        psutil = Mock()
        psutil.Process.return_value.create_time.return_value = 101
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('sys.modules', {'psutil': psutil}), \
                patch.object(supervisor.subprocess, 'Popen', return_value=process) as popen, \
                patch.object(supervisor, 'monitor_runner', side_effect=RuntimeError('model gone')), \
                patch.object(supervisor, 'release_owned_model', return_value={'terminated': [55]}) as cleanup:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'model gone'):
                supervisor.run_evaluation('python', self.config(root), root, {'models': []}, {}, 100)
            record = json.loads((root/'runner-process.json').read_text())
            cleanup.assert_called_once_with(record, 100)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            process.wait.assert_called_once_with(timeout=30)


if __name__ == '__main__':
    unittest.main()
