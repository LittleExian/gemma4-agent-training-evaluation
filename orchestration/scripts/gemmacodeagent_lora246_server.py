"""Native final Gemma W4 LoRA route, using the established isolated service launcher."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import struct
import threading
import time
from types import SimpleNamespace

import gemmacodeagent_mm_server as mm
from gemmacodeagent_owned_cleanup import release_owned_model

ARM = 'gemma4_w4_lora246'
BASE_ALIAS = 'gemma-4-31b-it-qat-w4a16-ct'
BASE_NAME = 'google/gemma-4-31B-it-qat-w4a16-ct'
REVISION = '52f3f65bc7a02d555763bc923bd1d9094898219d'
BASE_SHA = '1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20'
SPEC = {'quantization': 'original_kaggle_w4a16'}
_BASE_BUILD = mm.build_command
_HOOK_LOCK = threading.RLock()
_KEY = re.compile(r'base_model\.model\.model\.language_model\.layers\.\d+\.'
    r'(?:mlp\.(?:gate_proj|up_proj|down_proj)|self_attn\.(?:q_proj|k_proj|v_proj|o_proj))\.lora_([AB])\.weight')


def inspect_adapter(path):
    """Inspect the safetensors header without importing torch before CUDA selection."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        encoded = stream.read(8)
        if len(encoded) != 8:
            raise ValueError('Truncated adapter header')
        length = struct.unpack('<Q', encoded)[0]
        if not 0 < length <= min(2**20, size-8):
            raise ValueError('Invalid adapter header length')
        header = json.loads(stream.read(length))
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    if len(tensors) != 820 or size >= 3*2**30:
        raise ValueError('Expected the 820-tensor final LoRA export')
    spans, pairs = [], {}
    for key, value in tensors.items():
        match = _KEY.fullmatch(key)
        shape, offsets = value.get('shape', []), value.get('data_offsets', [])
        if (not match or value.get('dtype') != 'F32' or len(shape) != 2
                or any(type(d) is not int or d <= 0 for d in shape)
                or shape[0 if match[1] == 'A' else 1] != 16
                or len(offsets) != 2 or any(type(d) is not int for d in offsets)
                or offsets[0] < 0 or offsets[1]-offsets[0] != shape[0]*shape[1]*4):
            raise ValueError('Unexpected adapter tensor key, dtype, rank or storage')
        spans.append(tuple(offsets))
        prefix = key.rsplit('.lora_', 1)[0]
        pairs.setdefault(prefix, set()).add(match[1])
    ordered = sorted(spans)
    if (ordered[0][0] != 0 or ordered[-1][1] != size-8-length
            or any(a[1] != b[0] for a, b in zip(ordered, ordered[1:]))
            or any(pair != {'A', 'B'} for pair in pairs.values())):
        raise ValueError('Incomplete, overlapping or unpaired adapter storage')
    return {'adapter_tensor_count': len(tensors), 'adapter_dtype': 'F32'}


def validate_inputs(model, adapter, checkpoint_source=None):
    config_path, weights = adapter/'adapter_config.json', adapter/'adapter_model.safetensors'
    config = json.loads(config_path.read_text())
    required = {'base_model_name_or_path': BASE_NAME, 'revision': REVISION, 'r': 16,
                'lora_alpha': 16, 'bias': 'none', 'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'inference_mode': True}
    if (any(config.get(k) != v for k, v in required.items())
            or any(config.get(k) for k in ('modules_to_save', 'use_dora', 'use_rslora', 'rank_pattern', 'alpha_pattern', 'lora_bias'))):
        raise ValueError('Adapter config differs from the pinned rank-16 final export')
    checkpoint = checkpoint_source if checkpoint_source is not None else adapter.parent
    if json.loads((checkpoint/'COMPLETE.json').read_text()) != {'step': 246, 'world_size': 3}:
        raise ValueError('Checkpoint 246 is not complete')
    identity = json.loads((checkpoint/'base_identity.json').read_text())
    if identity != {'model': BASE_NAME, 'revision': REVISION, 'weight_sha256': BASE_SHA,
                    'base_loading': 'dequantize_w4_to_bf16_v1'}:
        raise ValueError('Checkpoint base identity differs from the original W4 model')
    audit = json.loads((checkpoint/'adapter_audit.json').read_text())
    if (audit.get('passed') is not True or audit.get('tensor_count') != 820 or audit.get('rank') != 16
            or audit.get('base_model') != BASE_NAME or audit.get('revision') != REVISION):
        raise ValueError('Final adapter export audit is missing or mismatched')
    model_config = json.loads((model/'config.json').read_text())
    quant = model_config.get('quantization_config', {})
    groups = quant.get('config_groups', {})
    if quant.get('quant_method') != 'compressed-tensors' or quant.get('format') != 'pack-quantized' or len(groups) != 1:
        raise ValueError('Expected the original packed W4 checkpoint configuration')
    group = next(iter(groups.values()))
    expected = {'num_bits': 4, 'type': 'int', 'group_size': 32, 'symmetric': True, 'strategy': 'group'}
    if (any(group.get('weights', {}).get(k) != v for k, v in expected.items())
            or group.get('input_activations') or group.get('output_activations')):
        raise ValueError('Expected W4A16 symmetric INT4/group32')
    generation = model/'generation_config.json'
    if json.loads(generation.read_text()).get('top_k') != 64:
        raise ValueError('Original checkpoint top_k=64 default is required')
    weights_sha, config_sha = mm.digest(weights), mm.digest(config_path)
    if checkpoint_source is not None and (checkpoint/'adapter').resolve() != adapter.resolve():
        if (mm.digest(checkpoint/'adapter/adapter_model.safetensors') != weights_sha
                or mm.digest(checkpoint/'adapter/adapter_config.json') != config_sha):
            raise ValueError('Packaged adapter differs from the complete training checkpoint')
    return {**inspect_adapter(weights), 'wrapper': 'gemmacodeagent_lora246_server_v1',
        'adapter_source': str(adapter), 'adapter_sha256': weights_sha,
        'adapter_config_sha256': config_sha, 'base_identity': identity,
        'checkpoint_source': str(checkpoint), 'wrapper_sha256': mm.digest(Path(__file__)),
        'base_generation_config_sha256': mm.digest(generation), 'effective_top_k': 64,
        'adapter_runtime_dtype': 'auto (base bfloat16)', 'adapter_file_transformation': 'none',
        'checkpoint_complete_sha256': mm.digest(checkpoint/'COMPLETE.json'),
        'adapter_audit_sha256': mm.digest(checkpoint/'adapter_audit.json')}


def build_command(root, model, adapter, port, memory):
    command = _BASE_BUILD(root, model, 'gemma4_kaggle', SPEC, port, memory)
    command[command.index('--served-model-name')+1] = BASE_ALIAS
    route = {'name': ARM, 'path': str(adapter), 'base_model_name': BASE_ALIAS}
    return command + ['--override-generation-config', json.dumps({'top_k': 64}),
                      '--enable-lora', '--max-lora-rank', '16', '--lora-dtype', 'auto',
                      '--max-loras', '1', '--max-cpu-loras', '1', '--lora-modules', json.dumps(route)]


@contextmanager
def _launch_hooks(adapter, proof, owned):
    """CLI-local hooks reuse the upstream launcher without changing source or packages."""
    with _HOOK_LOCK:
        original_builder, original_save = mm.build_command, mm.save_json
        def builder(root, model, arm, spec, port, memory, adapter_unused=None, rank=None, top_k=None):
            if arm != ARM or spec != SPEC or adapter_unused is not None or rank is not None or top_k is not None:
                raise ValueError('Unexpected native Gemma route parameters')
            return build_command(root, model, adapter, port, memory)
        def save(path, value):
            enriched = {**value, **proof} if path.name in ('launch-inputs.json', 'process.json') else value
            if path.name == 'process.json':
                owned.append(enriched)
            return original_save(path, enriched)
        mm.build_command, mm.save_json = builder, save
        try:
            yield
        finally:
            mm.build_command, mm.save_json = original_builder, original_save


def launch(args):
    mm.validate_options(args.gpu, args.slots, args.memory_utilization, args.port)
    if any(not p.is_absolute() for p in (args.root, args.model, args.adapter, args.state)):
        raise ValueError('All service paths must be absolute')
    checkpoint = getattr(args, 'checkpoint_source', None)
    if checkpoint is not None and not checkpoint.is_absolute():
        raise ValueError('Checkpoint source must be absolute')
    adapter, model, state = args.adapter.resolve(), args.model.resolve(), args.state.resolve()
    proof = {**validate_inputs(model, adapter, checkpoint.resolve() if checkpoint is not None else None),
             'wrapper_started_unix': time.time()}
    state.mkdir(parents=True, exist_ok=True)
    if any((state/name).exists() for name in ('process.json', 'launch-inputs.json', 'routes.json')):
        raise FileExistsError('Refusing to overwrite another service state or prior run')
    specs = state/'native-lora-models.json'
    with specs.open('x') as stream:
        json.dump({ARM: SPEC}, stream)
    bound = SimpleNamespace(**{**vars(args), 'model': model, 'adapter': adapter, 'state': state,
                               'models_json': specs, 'arm': ARM})
    owned = []
    try:
        with _launch_hooks(adapter, proof, owned):
            record = mm.launch(bound)
        return {**record, **proof}
    except BaseException as primary:
        if owned:
            try:
                result = release_owned_model(owned[-1], proof['wrapper_started_unix'])
                mm.save_json(state/'cleanup-secondary.json', result)
            except BaseException as secondary:
                if hasattr(primary, 'add_note'):
                    primary.add_note('Owned cleanup failed: '+type(secondary).__name__)
        raise


def cleanup(state):
    record = json.loads((state/'process.json').read_text())
    if record.get('arm') != ARM or record.get('wrapper') != 'gemmacodeagent_lora246_server_v1':
        raise ValueError('State does not describe this native Gemma LoRA service')
    result = release_owned_model(record, record['wrapper_started_unix'])
    mm.save_json(state/'cleanup.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    start = actions.add_parser('launch')
    for flag in ('root', 'model', 'adapter', 'state'):
        start.add_argument('--'+flag, type=Path, required=True)
    start.add_argument('--checkpoint-source', type=Path)
    start.add_argument('--gpu', type=int, required=True)
    start.add_argument('--port', type=int, required=True)
    start.add_argument('--slots', type=int, default=4)
    start.add_argument('--memory-utilization', type=float, default=.65)
    for name in ('health', 'cleanup'):
        action = actions.add_parser(name)
        action.add_argument('--state', type=Path, required=True)
        if name == 'health':
            action.add_argument('--timeout', type=float, default=1200)
    args = parser.parse_args()
    result = launch(args) if args.action == 'launch' else (mm.health(args.state, timeout=args.timeout)
        if args.action == 'health' else cleanup(args.state))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
