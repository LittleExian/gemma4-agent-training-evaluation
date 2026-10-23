import json
from pathlib import Path
import unittest
from unittest.mock import patch
import gemmacodeagent_step25_retest_server as module

class ConfigurationTests(unittest.TestCase):
 def test_builder_requests_graph_auto_unrestricted_topk_and_seed(self):
  command=module.build_command(Path('/r'),Path('/m'),Path('/a'),18826,.65)
  self.assertNotIn('--enforce-eager',command)
  self.assertEqual(command[command.index('--seed')+1],'42')
  self.assertEqual(command[command.index('--chat-template-content-format')+1],'auto')
  self.assertEqual(json.loads(command[command.index('--override-generation-config')+1]),{'top_k':-1,'seed':42})
  self.assertEqual(json.loads(command[command.index('--compilation-config')+1])['cudagraph_mode'],'FULL_AND_PIECEWISE')
  self.assertEqual(command.count('--override-generation-config'),1)
  route=json.loads(command[command.index('--lora-modules')+1]);self.assertEqual(route['name'],module.ARM)
  module.validate_launch_command(command)
 def test_wrong_runtime_flags_are_rejected(self):
  command=module.build_command(Path('/r'),Path('/m'),Path('/a'),18826,.65)
  for wrong in [command+['--enforce-eager'],[x if x!='42' else '43' for x in command],command+['--seed','43']]:
   with self.assertRaises(ValueError):module.validate_launch_command(wrong)
 def test_auto_is_not_reported_as_openai_without_runtime_evidence(self):
  with patch.object(module,'TOKENIZER_SHA256',{}):
   proof=module.validate_tokenizer()
  self.assertEqual(proof['requested_content_format'],'auto')
  self.assertIsNone(proof['resolved_content_format'])



import json
from pathlib import Path
import struct
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gemmacodeagent_step25_retest_server as wrapper
import gemmacodeagent_mm_server as mm


class Morning25Tests(unittest.TestCase):
    def setUp(self):
        for target in ('validate_resources', 'validate_tokenizer'):
            patcher = patch.object(wrapper, target, return_value={})
            patcher.start(); self.addCleanup(patcher.stop)

    def fixture(self, root):
        model, adapter = root/'model', root/'checkpoint-25/adapter'
        model.mkdir(); adapter.mkdir(parents=True)
        quant = {'quant_method': 'compressed-tensors', 'format': 'pack-quantized', 'config_groups': {'group_0': {
            'weights': {'num_bits': 4, 'type': 'int', 'group_size': 32, 'symmetric': True, 'strategy': 'group'}}}}
        (model/'config.json').write_text(json.dumps({'quantization_config': quant}))
        (model/'generation_config.json').write_text('{"top_k":64,"eos_token_id":[1,106,50]}')
        # Literal identities copied from the real checkpoint, independently of implementation constants.
        config = {'base_model_name_or_path': 'google/gemma-4-31B-it-qat-w4a16-ct',
            'revision': '52f3f65bc7a02d555763bc923bd1d9094898219d',
            'r': 16, 'lora_alpha': 16, 'bias': 'none', 'modules_to_save': None,
            'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'inference_mode': True,
            'use_dora': False, 'use_rslora': False, 'lora_dropout': .05}
        (adapter/'adapter_config.json').write_text(json.dumps(config))
        (adapter.parent/'COMPLETE.json').write_text('{"step":25,"world_size":2}')
        (adapter.parent/'base_identity.json').write_text(json.dumps({'model': 'google/gemma-4-31B-it-qat-w4a16-ct',
            'revision': '52f3f65bc7a02d555763bc923bd1d9094898219d',
            'weight_sha256': '1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20',
            'base_loading': 'dequantize_w4_to_bf16_v1'}))
        (adapter.parent/'adapter_audit.json').write_text(json.dumps({'passed': True, 'tensor_count': 820,
            'rank': 16, 'base_model': 'google/gemma-4-31B-it-qat-w4a16-ct',
            'revision': '52f3f65bc7a02d555763bc923bd1d9094898219d'}))
        header, offset = {}, 0
        for layer in range(410):
            prefix = f'base_model.model.model.language_model.layers.{layer}.mlp.down_proj'
            for letter, shape in [('A', [16, 1]), ('B', [1, 16])]:
                header[prefix+f'.lora_{letter}.weight'] = {'dtype': 'F32', 'shape': shape,
                    'data_offsets': [offset, offset+64]}
                offset += 64
        raw = json.dumps(header).encode()
        (adapter/'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(raw))+raw+b'\x00'*offset)
        return model, adapter

    def test_checkpoint_hf_identity_is_distinct_from_lowercase_serving_alias(self):
        self.assertEqual(wrapper.BASE_NAME, 'google/gemma-4-31B-it-qat-w4a16-ct')
        self.assertEqual(wrapper.BASE_ALIAS, 'gemma-4-31b-it-qat-w4a16-ct')

    def test_native_route_preserves_base_config_and_topk(self):
        args = (Path('/r'), Path('/m'), Path('/checkpoint/adapter'), 8120, .65)
        command = wrapper.build_command(*args)
        self.assertEqual(command[command.index('--served-model-name')+1], wrapper.BASE_ALIAS)
        self.assertNotIn('--generation-config', command)
        self.assertNotIn('--quantization', command)
        self.assertEqual(command[command.index('--lora-dtype')+1], 'auto')
        route = json.loads(command[command.index('--lora-modules')+1])
        self.assertEqual(route, {'name': wrapper.ARM, 'path': '/checkpoint/adapter', 'base_model_name': wrapper.BASE_ALIAS})
        self.assertEqual(json.loads(command[command.index('--override-generation-config')+1]), {'top_k': -1, 'seed': 42})
        self.assertNotIn('qwen35_swegym', ' '.join(command))

    def test_native_fp32_adapter_is_verified_without_remapping(self):
        with tempfile.TemporaryDirectory() as td:
            model, adapter = self.fixture(Path(td))
            proof = wrapper.validate_inputs(model, adapter)
            self.assertEqual(proof['adapter_tensor_count'], 820)
            self.assertEqual(proof['adapter_dtype'], 'F32')
            config = json.loads((adapter/'adapter_config.json').read_text())
            (adapter/'adapter_config.json').write_text(json.dumps({**config, 'r': 32}))
            with self.assertRaises(ValueError):
                wrapper.validate_inputs(model, adapter)

    def test_tensor_header_rejects_wrong_dtype_and_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            _, adapter = self.fixture(Path(td))
            path = adapter/'adapter_model.safetensors'
            raw = path.read_bytes()
            size = struct.unpack('<Q', raw[:8])[0]
            header = json.loads(raw[8:8+size]); first = next(iter(header))
            header[first]['dtype'] = 'BF16'
            replacement = json.dumps(header).encode()
            path.write_bytes(struct.pack('<Q', len(replacement))+replacement+raw[8+size:])
            with self.assertRaises(ValueError):
                wrapper.inspect_adapter(path)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                wrapper.inspect_adapter(path)

    def test_launch_hooks_are_scoped_and_record_native_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); model, adapter = self.fixture(root)
            args = SimpleNamespace(root=root, model=model, adapter=adapter, state=root/'state',
                gpu=0, port=8120, slots=4, memory_utilization=.65)
            original_build, original_save = mm.build_command, mm.save_json
            def launch(bound):
                self.assertEqual(bound.arm, wrapper.ARM)
                command = mm.build_command(root, model, bound.arm, {'quantization': 'original_kaggle_w4a16'},
                    bound.port, .65, None, None, top_k=None)
                record = {'pid': 12, 'create_time': 123., 'uid': 5, 'models': [bound.arm],
                    'arm': bound.arm, 'command': command}
                mm.save_json(bound.state/'launch-inputs.json', {'arm': bound.arm})
                mm.save_json(bound.state/'process.json', record)
                return record
            with patch.object(mm, 'launch', side_effect=launch), patch.object(mm, 'stage_adapter') as stage:
                record = wrapper.launch(args)
            self.assertIs(mm.build_command, original_build)
            self.assertIs(mm.save_json, original_save)
            stage.assert_not_called()
            self.assertEqual(record['adapter_source'], str(adapter.resolve()))
            self.assertEqual(json.loads((args.state/'process.json').read_text())['adapter_dtype'], 'F32')
            with self.assertRaises(FileExistsError), patch.object(mm, 'launch') as restart:
                wrapper.launch(args)
            restart.assert_not_called()

    def test_failed_launch_restores_hooks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); model, adapter = self.fixture(root)
            args = SimpleNamespace(root=root, model=model, adapter=adapter, state=root/'state',
                gpu=0, port=8120, slots=4, memory_utilization=.65)
            original_build, original_save = mm.build_command, mm.save_json
            with patch.object(mm, 'launch', side_effect=RuntimeError('test')):
                with self.assertRaises(RuntimeError):
                    wrapper.launch(args)
            self.assertIs(mm.build_command, original_build)
            self.assertIs(mm.save_json, original_save)

    def test_foreign_or_finished_state_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); model, adapter = self.fixture(root)
            state = root/'state'; state.mkdir()
            (state/'process.json').write_text('{"arm":"previous-run"}')
            args = SimpleNamespace(root=root, model=model, adapter=adapter, state=state,
                gpu=0, port=8120, slots=4, memory_utilization=.65)
            with self.assertRaises(FileExistsError), patch.object(mm, 'launch') as launch:
                wrapper.launch(args)
            launch.assert_not_called()
            self.assertEqual((state/'process.json').read_text(), '{"arm":"previous-run"}')

    def test_packaged_adapter_is_bound_to_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); model, adapter = self.fixture(root)
            exported = root/'submission/adapters'/wrapper.ARM
            shutil.copytree(adapter, exported)
            proof = wrapper.validate_inputs(model, exported, checkpoint_source=adapter.parent)
            self.assertEqual(proof['checkpoint_source'], str(adapter.parent))
            (exported/'adapter_model.safetensors').write_bytes((exported/'adapter_model.safetensors').read_bytes()[:-1]+b'x')
            with self.assertRaises(ValueError):
                wrapper.validate_inputs(model, exported, checkpoint_source=adapter.parent)

    def test_builder_supports_the_frozen_legacy_core(self):
        original = mm.build_command
        def legacy(root, model, arm, spec, port, memory):
            return original(root, model, arm, spec, port, memory)
        with patch.object(wrapper, '_BASE_BUILD', side_effect=legacy):
            command = wrapper.build_command(Path('/r'), Path('/m'), Path('/adapter'), 8120, .65)
        self.assertEqual(command.count('--override-generation-config'), 1)

    def test_health_cli_and_owned_cleanup(self):
        with patch('sys.argv', ['wrapper', 'health', '--state', '/state']), \
                patch.object(mm, 'health', return_value={}) as health, patch('builtins.print'):
            wrapper.main()
        health.assert_called_once_with(Path('/state'), timeout=1200)
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            (state/'process.json').write_text(json.dumps({'arm': wrapper.ARM,
                'wrapper': 'gemmacodeagent_step25_retest_server_v1', 'wrapper_started_unix': 123.}))
            with patch.object(wrapper, 'release_owned_model', return_value={'terminated': [9]}) as release:
                result = wrapper.cleanup(state)
            self.assertEqual(result, {'terminated': [9]})
            self.assertEqual(release.call_args.args[1], 123.)
            (state/'process.json').write_text('{"arm":"foreign"}')
            with self.assertRaises(ValueError), patch.object(wrapper, 'release_owned_model') as release:
                wrapper.cleanup(state)
            release.assert_not_called()

    def test_failure_after_process_record_cleans_only_that_record(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); model, adapter = self.fixture(root)
            args = SimpleNamespace(root=root, model=model, adapter=adapter, state=root/'state',
                gpu=0, port=8120, slots=4, memory_utilization=.65)
            def failed_launch(bound):
                mm.save_json(bound.state/'process.json', {'pid': 123, 'arm': wrapper.ARM})
                raise RuntimeError('after launch')
            with patch.object(mm, 'launch', side_effect=failed_launch), \
                    patch.object(wrapper, 'release_owned_model', return_value={}) as cleanup:
                with self.assertRaisesRegex(RuntimeError, 'after launch'):
                    wrapper.launch(args)
            self.assertEqual(cleanup.call_count, 1)
            self.assertEqual(cleanup.call_args.args[0]['pid'], 123)


class ResourceAndTokenizerTests(unittest.TestCase):
    def test_only_gpu0_bounded_memory_and_available_host_memory(self):
        args = SimpleNamespace(gpu=0, slots=4, memory_utilization=.65)
        with patch.dict('sys.modules', {'psutil':SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=25*2**30))}):
            self.assertEqual(wrapper.validate_resources(args)['host_available_bytes'],25*2**30)
            self.assertEqual(wrapper.validate_resources(SimpleNamespace(**{**vars(args),'gpu':2}))['host_available_bytes'],25*2**30)
            for changes in ({'gpu':1},{'memory_utilization':.66},{'slots':2}):
                with self.assertRaises(ValueError):
                    wrapper.validate_resources(SimpleNamespace(**{**vars(args),**changes}))
        with patch.dict('sys.modules', {'psutil':SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=23*2**30))}):
            with self.assertRaises(ValueError):wrapper.validate_resources(args)

    def test_tokenizer_is_pinned_and_mismatch_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); hashes={}
            for name in ('tokenizer.json','chat_template.jinja','tokenizer_config.json'):
                data=name.encode();(p/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
            with patch.object(wrapper,'TOKENIZER',p),patch.object(wrapper,'TOKENIZER_SHA256',hashes):
                self.assertEqual(wrapper.validate_tokenizer()['tokenizer_sha256'],hashes)
                (p/'chat_template.jinja').write_text('wrong')
                with self.assertRaises(ValueError):wrapper.validate_tokenizer()

    def test_command_pins_deployment_tokenizer_and_content_format(self):
        command=wrapper.build_command(Path('/r'),Path('/m'),Path('/a'),18825,.65)
        self.assertEqual(command[command.index('--tokenizer')+1],str(wrapper.TOKENIZER))
        self.assertEqual(command[command.index('--chat-template')+1],str(wrapper.TOKENIZER/'chat_template.jinja'))
        self.assertEqual(command[command.index('--chat-template-content-format')+1],'auto')

if __name__ == '__main__':
    unittest.main()
