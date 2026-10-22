"""Run in the competition wheelhouse inference environment, after training exits."""
import argparse,json,os,pathlib
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')

def main():
 from safetensors.torch import load_file,save_file
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 from base_loading import snapshot_path,check_resume,MODEL_ID,MODEL_REVISION
 from validate_w4 import audit_adapter
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 checkpoint=pathlib.Path(args.checkpoint);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
 check_resume(checkpoint);adapter=checkpoint/'adapter';export=audit_adapter(adapter)
 # A zero-B adapter tests the route independently of the trained adapter's quality.
 zero=out/'zero_adapter';zero.mkdir(exist_ok=True)
 (zero/'adapter_config.json').write_bytes((adapter/'adapter_config.json').read_bytes())
 state=load_file(str(adapter/'adapter_model.safetensors'))
 for k in state:
  if '.lora_B.' in k:state[k].zero_()
 save_file(state,str(zero/'adapter_model.safetensors'));del state
 llm=LLM(model=str(snapshot_path()),dtype='bfloat16',logprobs_mode='raw_logits',enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=2,
         tensor_parallel_size=1,max_model_len=32768,gpu_memory_utilization=.85,max_num_seqs=1,max_num_batched_tokens=2048,
         enforce_eager=True,enable_prefix_caching=False,limit_mm_per_prompt={'image':0,'audio':0,'video':0})
 tokenizer=llm.get_tokenizer()
 probes=['Explain why a mutable default argument can cause a Python bug.',
         'A Python function returns None for an empty list but its documented return type is int. Describe a minimal fix and test.']
 prompts=[tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=False,add_generation_prompt=True,enable_thinking=True) for s in probes]
 params=SamplingParams(temperature=0,max_tokens=1,logprobs=20,seed=42)
 def run(lora):
  rows=llm.generate(prompts,params,lora_request=lora,use_tqdm=False)
  return [{str(k):float(v.logprob) for k,v in r.outputs[0].logprobs[0].items()} for r in rows]
 plain=run(None);control=run(LoRARequest('zero',1,str(zero.resolve())));trained=run(LoRARequest('trained',2,str(adapter.resolve())))
 def max_delta(a,b):
  differences=[]
  for x,y in zip(a,b):
   common=x.keys() & y.keys()
   if not common:raise AssertionError('No shared tokens for logit comparison')
   differences.extend(abs(x[k]-y[k]) for k in common)
  return max(differences)
 zero_delta=max_delta(plain,control);effect=max_delta(control,trained)
 result=dict(passed=zero_delta<.002 and effect>1e-6,metric_mode="raw_logits",base_model=MODEL_ID,revision=MODEL_REVISION,
             zero_adapter_max_logit_delta=zero_delta,trained_adapter_max_logit_delta=effect,export=export,
             probes=len(prompts),max_model_len=32768,plain=plain,zero=control,trained=trained,
             note='Loading and nonzero adapter-effect gate only; no repair-accuracy or BF16/W4-kernel equivalence claim.')
 (out/'deployment_gate.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['plain','zero','trained']}),flush=True)
 if not result['passed']:raise AssertionError('W4 adapter control/effect gate failed')

if __name__=='__main__':main()
