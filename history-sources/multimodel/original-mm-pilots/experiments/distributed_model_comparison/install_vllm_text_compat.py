"""Idempotent compatibility patch for the pinned vLLM 0.19.1 text-only Qwen3.5 loader."""
import importlib.metadata,sysconfig,os,hashlib,json
from pathlib import Path
assert importlib.metadata.version('vllm')=='0.19.1','This compatibility patch is only verified for vLLM 0.19.1'
root=Path(sysconfig.get_paths()['purelib'])/'vllm/model_executor/models';backup=Path(os.environ['CODEAGENT_ROOT'])/'research/vllm_text_compat';backup.mkdir(parents=True,exist_ok=True)
registry=root/'registry.py';model=root/'qwen3_5.py';changes={};r=registry.read_text();t=model.read_text()
key='    "Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),'
if '"Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM")' not in r:
 assert key in r;changes[registry]=r.replace(key,'    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'+key)
old='class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):\n    pass'
if old in t:
 start=t.index('    @classmethod\n    def get_mamba_state_dtype_from_config',t.index('class Qwen3_5ForConditionalGeneration'));end=t.index('\n\n########################################################',start)
 changes[model]=t.replace(old,'class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase, IsHybrid):\n'+t[start:end])
else:assert 'class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase, IsHybrid):' in t,'Unexpected source; refusing to patch'
updated=changes.get(model,t)
key='class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase, IsHybrid):\n'
addition='    supports_mrope = True\n\n    def get_mrope_input_positions(self, input_tokens, mm_features):\n        assert not mm_features, "Text-only checkpoint cannot process multimodal features"\n        return torch.arange(len(input_tokens), dtype=torch.long).unsqueeze(0).expand(3, -1).contiguous(), 0\n\n'
if addition not in updated:
 assert key in updated
 changes[model]=updated.replace(key,key+addition)
for path,new in changes.items():compile(new,str(path),'exec')
audit=[]
for path,new in changes.items():
 original=path.read_bytes();sha=hashlib.sha256(original).hexdigest();(backup/(path.name+'.'+sha+'.bak')).write_bytes(original);path.write_text(new);audit.append({'path':str(path),'before':sha,'after':hashlib.sha256(path.read_bytes()).hexdigest()})
if audit:(backup/'last_patch.json').write_text(json.dumps(audit,indent=2))
print('Qwen text compatibility: patched' if audit else 'Qwen text compatibility: already installed')
