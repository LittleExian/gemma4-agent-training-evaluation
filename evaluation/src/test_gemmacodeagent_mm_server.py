import json
from pathlib import Path
import tempfile
import types
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import Mock, patch
import gemmacodeagent_mm_server as mm


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.identity_patch = patch.object(mm, 'verify_service_identity')
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)

    def test_stale_service_identity_refused(self):
        self.identity_patch.stop()
        fake = Mock(NoSuchProcess=ProcessLookupError)
        fake.Process.return_value.create_time.return_value = 200
        with patch.dict('sys.modules', {'psutil': fake}):
            with self.assertRaises(RuntimeError):
                mm.verify_service_identity({'pid': 1, 'create_time': 100, 'uid': 5})

    def test_exact_four_and_safe_devices(self):
        for gpu, slots in [(0, 8), (2, 1), (1, 4), (3, 0)]:
            with self.assertRaises(ValueError):
                mm.validate_options(gpu, slots, .65, 8120)
        mm.validate_options(2, 4, .65, 8122)

    def test_quantization_all_arms(self):
        for arm, quant in [('qwen35_base', 'fp8'), ('qwen38_base', 'published_fp8'),
                           ('gemma4_teich_agent', 'fp8'), ('gemma4_kaggle', 'original_kaggle_w4a16')]:
            cmd = mm.build_command(Path('/r'), Path('/model'), arm, {'quantization': quant}, 8120, .65)
            self.assertEqual(cmd[cmd.index('--max-num-seqs') + 1], '4')
            self.assertEqual('--quantization' in cmd, quant == 'fp8')
            self.assertIn('--enforce-eager', cmd)
            self.assertEqual('--generation-config' in cmd, arm != 'gemma4_kaggle')
            self.assertEqual('--language-model-only' in cmd, arm.startswith('qwen'))

    def test_capybara_prefix_override_is_only_applied_to_capybara(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model, state = root/'model', root/'state'
            model.mkdir(); state.mkdir()
            source = "{% if add_generation_prompt %}{{ '<|turn>model\\n' }}{% endif %}"
            (model/'chat_template.jinja').write_text(source)
            self.assertIsNone(mm.prepare_template(model, 'gemma4_kaggle', state))
            path = mm.prepare_template(model, 'gemma4_capybara', state)
            self.assertTrue(path.read_text().startswith(source))
            evidence = json.loads((state/'serving-adaptation.json').read_text())
            self.assertEqual(evidence['source_sha256'], mm.digest(model/'chat_template.jinja'))
            self.assertEqual(evidence['override_sha256'], mm.digest(path))

    def test_lora_remapping_requires_expected_prefix(self):
        self.assertEqual(mm.remap_key('base_model.model.model.layers.0.x'),
                         'base_model.model.model.language_model.layers.0.x')
        with self.assertRaises(ValueError):
            mm.remap_key('unexpected.0.x')

    def test_prefilled_capybara_uses_its_parser_plugin_only(self):
        cmd = mm.build_command(Path('/r'), Path('/m'), 'gemma4_capybara', {'quantization': 'fp8'}, 8150, .65)
        self.assertEqual(cmd[cmd.index('--reasoning-parser')+1], 'capybara_budget')
        self.assertEqual(cmd[cmd.index('--reasoning-parser-plugin')+1], '/r/src/gemmacodeagent_capybara_parser.py')
        base = mm.build_command(Path('/r'), Path('/m'), 'gemma4_teich_agent', {'quantization': 'fp8'}, 8150, .65)
        self.assertNotIn('--reasoning-parser-plugin', base)

    def test_tool_validation_is_strict(self):
        valid = {'choices': [{'message': {'tool_calls': [{'type': 'function', 'function': {
            'name': 'run_command', 'arguments': '{"command":"pwd"}'}}]}}]}
        mm.validate_tool(valid)
        for args in ['{}', '{"command":2}', '{"command":"pwd","evil":true}', 'not json']:
            candidate = json.loads(json.dumps(valid))
            candidate['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = args
            with self.assertRaises((ValueError, RuntimeError)):
                mm.validate_tool(candidate)

    def test_identity_and_budget(self):
        mm.check_identity({'data': [{'id': 'a'}, {'id': 'b'}]}, ['a', 'b'])
        with self.assertRaises(RuntimeError):
            mm.check_identity({'data': [{'id': 'wrong'}]}, ['a'])
        tokenizer = Mock()
        tokenizer.encode.return_value = [1] * 16
        data = {'choices': [{'message': {'reasoning': 'thinking', 'content': 'answer'}}]}
        self.assertEqual(mm.validate_thinking(data, tokenizer), 16)
        for length in [0, 25]:
            tokenizer.encode.return_value = [1] * length
            with self.assertRaises(RuntimeError):
                mm.validate_thinking(data, tokenizer)

    def test_health_four_concurrent_and_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            (state/'process.json').write_text(json.dumps({'port': 8120, 'models': ['a'], 'model_path': '/model'}))
            tokenizer = Mock()
            tokenizer.encode.return_value = [1] * 16
            def request(url, body=None, timeout=300):
                if body is None:
                    return {'data': [{'id': 'a'}]}
                if body.get('tools'):
                    message = {'tool_calls': [{'type': 'function', 'function': {'name': 'run_command', 'arguments': '{"command":"pwd"}'}}]}
                else:
                    message = {'reasoning': 'reason', 'content': 'answer'}
                return {'choices': [{'message': message}], 'usage': {'completion_tokens': 25}}
            with patch.object(mm, 'request_json', side_effect=request):
                result = mm.health(state, tokenizer=tokenizer, timeout=1)
            self.assertTrue(result['passed'])
            self.assertEqual(result['concurrency'], 4)
            self.assertEqual(len(result['rows']), 6)
            self.assertTrue((state/'health.json').exists())

    def test_adapter_mapping_records_bitwise_identity(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            source = state/'source'
            source.mkdir()
            (source/'adapter_config.json').write_text('{"r":32}')
            (source/'adapter_model.safetensors').write_bytes(b'original')
            tensor = Mock(dtype='torch.bfloat16')
            tensor.equal.return_value = True
            original = 'base_model.model.model.layers.0.weight'
            mapped = mm.remap_key(original)
            source_file, mapped_file = Mock(), Mock()
            source_file.__enter__ = Mock(return_value=source_file)
            source_file.__exit__ = Mock(return_value=False)
            mapped_file.__enter__ = Mock(return_value=mapped_file)
            mapped_file.__exit__ = Mock(return_value=False)
            source_file.keys.return_value = [original]
            mapped_file.keys.return_value = [mapped]
            source_file.get_tensor.return_value = mapped_file.get_tensor.return_value = tensor
            safe = Mock()
            safe.safe_open.side_effect = [source_file, mapped_file]
            safe_torch = Mock()
            def save(tensors, path):
                Path(path).write_bytes(b'remapped')
            safe_torch.save_file.side_effect = save
            with patch.dict('sys.modules', {'safetensors': safe, 'safetensors.torch': safe_torch}):
                destination, rank = mm.stage_adapter(source, state)
            self.assertEqual(rank, 32)
            self.assertTrue((destination/'adapter_model.safetensors').exists())
            self.assertTrue(json.loads((state/'adapter_mapping.json').read_text())['bitwise_equal'])

    def test_shared_lora_both_names_health(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            models = ['qwen35_base', 'qwen35_swegym']
            (state/'process.json').write_text(json.dumps({'port': 8120, 'models': models, 'model_path': '/m'}))
            tokenizer = Mock()
            tokenizer.encode.return_value = [1]*16
            def request(url, body=None, timeout=300):
                if body is None:
                    return {'data': [{'id': name} for name in models]}
                msg = {'reasoning': 'reason', 'content': 'answer'}
                if body.get('tools'):
                    msg = {'tool_calls': [{'type': 'function', 'function': {'name': 'run_command', 'arguments': '{"command":"pwd"}'}}]}
                return {'choices': [{'message': msg, 'logprobs': {'model': body['model']}}], 'usage': {'completion_tokens': 20}}
            with patch.object(mm, 'request_json', side_effect=request):
                result = mm.health(state, tokenizer, timeout=1)
            self.assertTrue(result['lora_logits_differ'])
            self.assertEqual(set(result['thinking_tokens']), set(models))

    def test_cli_health(self):
        with patch('sys.argv', ['mm', 'health', '--state', '/run']), patch.object(mm, 'health', return_value={}) as health, patch('builtins.print'):
            mm.main()
        health.assert_called_once_with(Path('/run'), timeout=1200)

    def test_launch_with_mock_gpu_process_and_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root/'model'
            model.mkdir()
            (model/'config.json').write_text('{}')
            specs = root/'models.json'
            specs.write_text(json.dumps({'qwen35_base': {'quantization': 'fp8'}}))
            fake_psutil = Mock(NoSuchProcess=ProcessLookupError, STATUS_ZOMBIE='zombie')
            fake_psutil.Process.return_value.create_time.return_value = 123
            fake_torch = Mock()
            fake_torch.cuda.get_device_properties.return_value.uuid = '123'
            args = types.SimpleNamespace(root=root, model=model, state=root/'state', models_json=specs,
                arm='qwen35_base', adapter=root/'adapter', gpu=2, slots=4, memory_utilization=.65, port=8122)
            with patch.dict('sys.modules', {'psutil': fake_psutil, 'torch': fake_torch}), \
                    patch.object(mm.socket, 'socket'), \
                    patch.object(mm.subprocess, 'check_output', side_effect=[
                        'GPU-123, NVIDIA RTX PRO 6000 Blackwell, 98304, 80000', 'foreign preserved']), \
                    patch.object(mm.subprocess, 'Popen', return_value=Mock(pid=55)), \
                    patch.object(mm, 'stage_adapter', return_value=(root/'adapter-staged', 32)), \
                    patch.dict(mm.os.environ, {}, clear=True):
                record = mm.launch(args)
            self.assertEqual(record['models'], ['qwen35_base', 'qwen35_swegym'])
            self.assertEqual(record['pid'], 55)
            self.assertEqual(record['gpu_uuid'], 'GPU-123')
            self.assertIn('--lora-dtype', record['command'])
            if hasattr(mm.time, 'CLOCK_BOOTTIME'):
                self.assertIsInstance(record['launch_boottime'], float)
            else:
                self.assertIsNone(record['launch_boottime'])
            self.assertEqual(json.loads((args.state/'routes.json').read_text())['qwen35_swegym']['port'], 8122)

    def test_actual_http_four_concurrent_requests(self):
        counter = {'active': 0, 'maximum': 0}
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.reply({'data': [{'id': 'a'}]})
            def reply(self, data):
                encoded = json.dumps(data).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body.get('tools'):
                    msg = {'tool_calls': [{'type': 'function', 'function': {'name': 'run_command', 'arguments': '{"command":"pwd"}'}}]}
                else:
                    msg = {'content': 'answer', 'reasoning': 'thinking'}
                if body['max_tokens'] == 128:
                    import time
                    with lock:
                        counter['active'] += 1
                        counter['maximum'] = max(counter['maximum'], counter['active'])
                    time.sleep(.1)
                    with lock:
                        counter['active'] -= 1
                self.reply({'choices': [{'message': msg}], 'usage': {'completion_tokens': 20}})
        service = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=service.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                state = Path(td)
                (state/'process.json').write_text(json.dumps({'port': service.server_port, 'models': ['a'], 'model_path': '/m'}))
                tokenizer = Mock()
                tokenizer.encode.return_value = [1]*16
                result = mm.health(state, tokenizer, timeout=1)
                self.assertTrue(result['passed'])
                self.assertEqual(counter['maximum'], 4)
        finally:
            service.shutdown()
            service.server_close()
            worker.join()

    def test_health_failure_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            (state/'process.json').write_text(json.dumps({'port': 8120, 'models': ['a'], 'model_path': '/model'}))
            with patch.object(mm, 'request_json', return_value={'data': [{'id': 'wrong'}]}):
                with self.assertRaises(RuntimeError):
                    mm.health(state, tokenizer=Mock(), timeout=0)
            self.assertFalse(json.loads((state/'health.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
