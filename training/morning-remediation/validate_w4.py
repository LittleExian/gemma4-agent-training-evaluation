"""Independent INT4 unpack checks and adapter-only export validation."""
import json
from pathlib import Path

def unpack_rows(packed, scale, columns):
    import torch
    # CT packs unsigned nibbles (signed value + 8), low nibble first.
    shifts = torch.arange(8, dtype=torch.int64) * 4
    signed = (((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 15) - 8).reshape(packed.shape[0], -1)[:, :columns]
    return (signed.float() * scale.float().repeat_interleave(32, dim=1)[:, :columns]).to(torch.bfloat16)

def audit_decoded(base):
    import torch
    from safetensors import safe_open
    from base_loading import snapshot_path,base_identity
    checked = 0; values = 0; max_diff = 0.
    with safe_open(str(snapshot_path() / 'model.safetensors'), framework='pt', device='cpu') as f:
        keys = set(f.keys())
        for key in sorted(k for k in keys if k.endswith('.weight_packed')):
            name = key[:-len('.weight_packed')]
            module = base.get_submodule(name)
            if hasattr(module, "get_base_layer"): module = module.get_base_layer()
            if type(module) is not torch.nn.Linear or module.weight.requires_grad:
                raise AssertionError(f'Expected a frozen ordinary Linear: {name}')
            rows, columns = module.weight.shape
            for row in sorted(set([0, rows//2, rows-1])):
                packed = f.get_slice(key)[row:row+1]
                scale = f.get_slice(name+'.weight_scale')[row:row+1]
                ref = unpack_rows(packed, scale, columns)
                actual = module.weight[row:row+1].detach().cpu()
                difference = (ref.float()-actual.float()).abs().max().item()
                max_diff = max(max_diff, difference); values += ref.numel()
                if not torch.equal(ref, actual): raise AssertionError(f'Dequantization mismatch: {name} row {row}, max_abs={difference}')
            checked += 1
    if checked != base._decoded_w4_layers: raise AssertionError('Not all packed layers audited')
    return dict(passed=True, **base_identity(), decoded_layers=checked, sampled_values=values, max_abs_difference=max_diff,
                scope='First, middle and last output rows of every packed matrix; not exhaustive elementwise or kernel equivalence.')

def audit_adapter(folder):
    from safetensors import safe_open
    from base_loading import MODEL_ID,MODEL_REVISION
    folder = Path(folder)
    config = json.loads((folder/'adapter_config.json').read_text())
    if config['base_model_name_or_path'] != MODEL_ID or config['revision'] != MODEL_REVISION:
        raise AssertionError('Adapter must name the pinned original deployment W4 base')
    if config['r'] > 128 or config.get('modules_to_save') or config.get('bias','none')!='none':
        raise AssertionError('Expected LoRA-only export')
    file = folder/'adapter_model.safetensors'
    with safe_open(str(file),framework='pt',device='cpu') as f:
        keys = list(f.keys())
        if not keys or any('.lora_A.' not in k and '.lora_B.' not in k for k in keys):
            raise AssertionError('Export contains non-LoRA tensors')
        if any(0 in f.get_slice(k).get_shape() for k in keys): raise AssertionError('Empty LoRA tensors')
    size = file.stat().st_size
    if size >= 3*2**30: raise AssertionError('Adapter exceeds 3 GiB')
    return dict(passed=True, tensor_count=len(keys), weight_bytes=size, rank=config['r'], base_model=MODEL_ID, revision=MODEL_REVISION)

if __name__=='__main__':
    import argparse,torch
    from base_loading import MODEL_ID,MODEL_REVISION,load_base
    ap=argparse.ArgumentParser();ap.add_argument('--adapter');ap.add_argument('--output',required=True);args=ap.parse_args()
    result=audit_adapter(args.adapter) if args.adapter else audit_decoded(load_base(MODEL_ID,MODEL_REVISION,0))
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
