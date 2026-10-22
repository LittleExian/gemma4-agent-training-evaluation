"""Frozen deployment W4 values decoded to BF16, never the unquantized ancestor."""
import hashlib
import json
from pathlib import Path
MODEL_ID = 'google/gemma-4-31B-it-qat-w4a16-ct'
MODEL_REVISION = '52f3f65bc7a02d555763bc923bd1d9094898219d'
WEIGHT_SHA256 = '1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20'
BASE_MODE = 'dequantize_w4_to_bf16_v1'

def base_identity():
    return dict(model=MODEL_ID, revision=MODEL_REVISION, weight_sha256=WEIGHT_SHA256, base_loading=BASE_MODE)

def check_source(model, revision):
    if model != MODEL_ID or revision != MODEL_REVISION:
        raise ValueError('Use the pinned deployment W4 base from base_loading.py; the old unquantized base is not interchangeable.')

def check_resume(folder):
    p = Path(folder) / 'base_identity.json'
    if not p.exists() or json.loads(p.read_text()) != base_identity():
        raise ValueError('Checkpoint base identity is missing or different. Start a fresh W4-aligned run; do not resume old unquantized checkpoints.')

def snapshot_path():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))

def verify_source_hash():
    h = hashlib.sha256()
    with (snapshot_path() / 'model.safetensors').open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''): h.update(chunk)
    if h.hexdigest() != WEIGHT_SHA256: raise ValueError('Deployment checkpoint SHA256 mismatch')
    return h.hexdigest()

def validate_quant_config(config):
    q = config.get('quantization_config', {})
    if q.get('quant_method') != 'compressed-tensors' or q.get('format') != 'pack-quantized':
        raise ValueError('Expected original compressed-tensors pack-quantized checkpoint')
    groups = q.get('config_groups', {})
    if len(groups) != 1: raise ValueError('Unexpected quantization groups')
    g = next(iter(groups.values())); w = g.get('weights', {})
    expected = dict(num_bits=4, type='int', group_size=32, symmetric=True, strategy='group')
    if any(w.get(k) != v for k,v in expected.items()) or g.get('input_activations') or g.get('output_activations'):
        raise ValueError('Expected W4A16 symmetric INT4/group32 without activation quantization')

def load_base(model, revision, device, attention='sdpa'):
    import torch
    from transformers import Gemma4ForConditionalGeneration, CompressedTensorsConfig
    check_source(model, revision)
    validate_quant_config(json.loads((snapshot_path() / 'config.json').read_text()))
    base = Gemma4ForConditionalGeneration.from_pretrained(
        model, revision=revision, dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(dequantize=True),
        attn_implementation=attention, device_map={'': device}, local_files_only=True)
    # Strip quantization wrappers after decoding; preserve the decoded values.
    decoded = 0
    for name, module in list(base.named_modules()):
        if getattr(module, 'quantization_scheme', None) is not None:
            if not isinstance(module, torch.nn.Linear) or not hasattr(module, 'weight') or hasattr(module, 'weight_packed'):
                raise RuntimeError(f'Unsupported or still-compressed layer: {name}')
            dense = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device='meta', dtype=module.weight.dtype)
            dense.weight = module.weight
            if module.bias is not None: dense.bias = module.bias
            base.set_submodule(name, dense); decoded += 1
    if decoded == 0: raise RuntimeError('No W4 layers decoded; refusing an unverified base')
    for name, module in base.model.language_model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight.dtype != torch.bfloat16:
            raise RuntimeError(f'Expected dense BF16 weight: {name} {module.weight.dtype}')
    if hasattr(base.config, 'quantization_config'): delattr(base.config, 'quantization_config')
    for attr in ('hf_quantizer', 'quantization_method'):
        if hasattr(base, attr): delattr(base, attr)
    base.is_quantized = False
    base._decoded_w4_layers = decoded
    base.requires_grad_(False)
    return base

def assert_lora_only(model):
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    if not names or any('.lora_A.' not in n and '.lora_B.' not in n for n in names):
        raise RuntimeError('Only LoRA A/B matrices may be trainable')
    return names
