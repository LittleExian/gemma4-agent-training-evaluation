import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import gemmacodeagent_full_server as server


class FullServerTests(unittest.TestCase):
    def test_limits_and_device_allowlist(self):
        for gpu, concurrency, memory, port in [(1, 4, .65, 8100), (0, 5, .65, 8100),
                                               (2, 0, .65, 8100), (3, 4, .71, 8100),
                                               (3, 4, float('nan'), 8100), (0, 4, .65, 0)]:
            with self.subTest(gpu=gpu, concurrency=concurrency, memory=memory):
                with self.assertRaises(ValueError):
                    server.validate_options(gpu, concurrency, memory, port)
        server.validate_options(0, 4, .65, 8106)

    def test_gpu_budget_uses_requested_utilization(self):
        self.assertEqual(server.validate_gpu('GPU-123, NVIDIA RTX PRO 6000 Blackwell, 98304, 80000', .65)['uuid'], 'GPU-123')
        for line in ['GPU-x, NVIDIA H100, 98304, 98000',
                     'GPU-x, NVIDIA RTX PRO 6000 Blackwell, 98304, 70000']:
            with self.assertRaises(RuntimeError):
                server.validate_gpu(line, .65)

    def test_build_config_and_command(self):
        config = server.config_kwargs(Path('/repo'), 8107, 4, .65)
        self.assertEqual(config['gpu_memory_utilization'], .65)
        self.assertEqual(config['tensor_parallel_size'], 1)
        self.assertFalse(config['enable_lora'])
        self.assertEqual(config['extra_args'][-1], '--enforce-eager')
        self.assertEqual(config['extra_args'][3], '4')
        self.assertEqual(config['max_model_len'], 32768)

    def test_cache_paths_are_independent(self):
        original = {'HOME': '/home/user', 'HF_HOME': '/shared'}
        result = server.launch_environment(original, Path("/run/" + "long" * 40), 2, Path("/tmp/g4-private"))
        self.assertEqual(original['HF_HOME'], '/shared')
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], '2')
        self.assertEqual(result['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
        self.assertEqual(result['TMPDIR'], '/tmp/g4-private')
        self.assertLess(len(result['TMPDIR'].encode()) + 1 + 36, 107)
        self.assertTrue(result['TRITON_CACHE_DIR'].startswith('/run/'))

    def test_verification_requires_both_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'runs').mkdir()
            (root/'runs/environment-verified.json').write_text(json.dumps({'official_six_source_hashes_match': True}))
            (root/'runs/model-sha256.json').write_text(json.dumps({'matches': False}))
            with self.assertRaises(RuntimeError):
                server.verify_records(root)
            (root/'runs/model-sha256.json').write_text(json.dumps({'matches': True}))
            server.verify_records(root)

    def test_live_record_refused_and_dead_record_allowed(self):
        fake = Mock()
        fake.NoSuchProcess = ProcessLookupError
        fake.STATUS_ZOMBIE = 'zombie'
        fake.Process.return_value.create_time.return_value = 123
        fake.Process.return_value.status.return_value = 'running'
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state/'process.json').write_text(json.dumps({'pid': 55, 'create_time': 123}))
            with self.assertRaises(RuntimeError):
                server.refuse_live_record(state, fake)
            fake.Process.return_value.create_time.return_value = 124
            server.refuse_live_record(state, fake)
            fake.Process.side_effect = ProcessLookupError
            server.refuse_live_record(state, fake)

    def test_uuid_normalization(self):
        server.validate_uuid('GPU-AABB', 'aabb')
        with self.assertRaises(RuntimeError):
            server.validate_uuid('GPU-AABB', 'ccdd')

    def test_mock_launch_records_only_owned_process_and_gpu_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'runs').mkdir()
            for name, content in [('environment-verified.json', {'official_six_source_hashes_match': True}),
                                  ('model-sha256.json', {'matches': True})]:
                (root/'runs'/name).write_text(json.dumps(content))
            fake_psutil = Mock(NoSuchProcess=ProcessLookupError, STATUS_ZOMBIE='zombie')
            fake_psutil.Process.return_value.create_time.return_value = 123
            fake_torch = Mock()
            fake_torch.cuda.get_device_properties.return_value.uuid = '123'
            fake_adk = Mock()
            fake_adk.VllmServer.return_value.build_cmd.return_value = ['python', 'serve']
            fake_adk.VllmServer.return_value.build_env.return_value = {'CUDA_VISIBLE_DEVICES': '2'}
            args = types.SimpleNamespace(repo=root, state_dir=root/'state', gpu=2,
                                         concurrency=4, memory_utilization=.65, port=8107)
            with patch.dict('sys.modules', {'psutil': fake_psutil, 'torch': fake_torch, 'adk_submission': fake_adk}), \
                    patch.object(server.socket, 'socket'), \
                    patch.object(server.subprocess, 'check_output', side_effect=[
                        'GPU-123, NVIDIA RTX PRO 6000 Blackwell, 98304, 80000', 'GPU-other, 900, foreign, 1000']), \
                    patch.object(server.subprocess, 'Popen', return_value=Mock(pid=55)), \
                    patch.dict(server.os.environ, {}, clear=True):
                record = server.launch(args)
            self.assertEqual(record['pid'], 55)
            self.assertEqual(record['create_time'], 123)
            self.assertIn('foreign', record['foreign_gpu_processes_before'])
            self.assertEqual(json.loads((root/'state/process.json').read_text())['gpu_uuid'], 'GPU-123')
            args.state_dir = Path('relative')
            with self.assertRaises(ValueError):
                server.launch(args)


if __name__ == '__main__':
    unittest.main()
