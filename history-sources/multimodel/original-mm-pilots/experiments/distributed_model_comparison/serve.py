import os,json,subprocess,sys,shutil,hashlib
from pathlib import Path
P=Path(os.environ['CODEAGENT_ROOT']);Q=P/'research/distributed_model_comparison';S=Path(__file__).parent
arm=sys.argv[1]
if arm=='qwen38_efficient':subprocess.run([sys.executable,str(S/'install_vllm_text_compat.py')],check=True)
if arm in ['gemma4_teich_agent','gemma4_capybara']:
 model=Q/'models'/arm
 assert (model/'DOWNLOAD_COMPLETE').exists()
 cmd=[str(P/'envs/eval/bin/python'),'-m','vllm.entrypoints.openai.api_server','--model',str(model),'--served-model-name',arm,'--host','127.0.0.1','--port','8002','--dtype','bfloat16','--max-model-len','32768','--max-num-seqs','32','--max-num-batched-tokens','4096','--gpu-memory-utilization','0.90','--reasoning-parser','gemma4','--tool-call-parser','gemma4','--enable-auto-tool-choice','--generation-config','vllm','--reasoning-config',json.dumps({'reasoning_start_str':'<|channel>','reasoning_end_str':'<channel|>'}),'--limit-mm-per-prompt',json.dumps({'image':0,'audio':0,'video':0}),'--no-scheduler-reserve-full-isl','--quantization','fp8']
 (Q/'routes.json').write_text(json.dumps({arm:{'model':arm,'port':8002}}));(Q/(arm+'_server_command.json')).write_text(json.dumps(cmd,indent=2))
 env={**os.environ,'HF_HOME':str(P/'cache/huggingface'),'VLLM_CACHE_ROOT':str(P/'cache/mm_vllm'),'TRITON_CACHE_DIR':str(P/'cache/triton'),'TORCHINDUCTOR_CACHE_DIR':str(P/'cache/torchinductor'),'TMPDIR':str(P/'tmp/mm')}
 os.execvpe(cmd[0],cmd,env)
if arm=='gemma4_kaggle':
 model=P/'models/gemma-4-31b-it-qat-w4a16-ct'
 cmd=[str(P/'envs/eval/bin/python'),'-m','vllm.entrypoints.openai.api_server','--model',str(model),'--host','127.0.0.1','--port','8002','--served-model-name',arm,'--dtype','bfloat16','--max-model-len','32768','--max-num-seqs','32','--max-num-batched-tokens','4096','--gpu-memory-utilization','0.9','--enable-auto-tool-choice','--tool-call-parser','gemma4','--reasoning-parser','gemma4','--reasoning-config',json.dumps({'reasoning_start_str':'<|channel>','reasoning_end_str':'<channel|>'}),'--limit-mm-per-prompt',json.dumps({'image':0,'audio':0,'video':0}),'--no-scheduler-reserve-full-isl']
 (Q/'routes.json').write_text(json.dumps({arm:{'model':arm,'port':8002}}))
 os.execvpe(cmd[0],cmd,{**os.environ,'HF_HOME':str(P/'cache/huggingface'),'VLLM_CACHE_ROOT':str(P/'cache/mm_vllm'),'TMPDIR':str(P/'tmp/mm')})
spec=json.loads((S/'models.json').read_text())[arm];model=Q/'models'/spec['directory'];assert (model/'DOWNLOAD_COMPLETE').exists()
port=int(os.environ.get('MM_PORT','8002'));seq=int(os.environ.get('MM_MAX_SEQS','32'))
env={**os.environ,'HF_HOME':str(P/'cache/huggingface'),'VLLM_CACHE_ROOT':str(P/'cache/mm_vllm'),'TRITON_CACHE_DIR':str(P/'cache/triton'),'TORCHINDUCTOR_CACHE_DIR':str(P/'cache/torchinductor'),'TMPDIR':str(P/'tmp/mm'),'TOKENIZERS_PARALLELISM':'false'}
Path(env['TMPDIR']).mkdir(exist_ok=True)
cmd=[str(P/'envs/eval/bin/python'),'-m','vllm.entrypoints.openai.api_server','--model',str(model),'--served-model-name',arm,'--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--max-model-len','32768','--max-num-seqs',str(seq),'--max-num-batched-tokens','4096','--gpu-memory-utilization','0.90','--reasoning-parser','qwen3','--tool-call-parser','qwen3_coder','--enable-auto-tool-choice','--generation-config','vllm','--reasoning-config',json.dumps({'reasoning_start_str':'<think>','reasoning_end_str':'</think>'})]
cfg=json.loads((model/'config.json').read_text())
cmd+=['--language-model-only']
if spec['quantization']=='fp8':cmd+=['--quantization','fp8']
routes={arm:{'model':arm,'port':port}}
if arm=='qwen35_base':
 from safetensors import safe_open
 from safetensors.torch import save_file
 src=Q/'models/qwen35_adapter';assert (src/'DOWNLOAD_COMPLETE').exists();dst=Q/'models/qwen35_adapter_vllm';dst.mkdir(exist_ok=True)
 ts={};mapping={}
 with safe_open(src/'adapter_model.safetensors',framework='pt',device='cpu') as f:
  for k in f.keys():
   assert k.startswith('base_model.model.model.layers.'),k
   nk=k.replace('base_model.model.model.layers.','base_model.model.model.language_model.layers.',1);ts[nk]=f.get_tensor(k);mapping[k]=nk
 save_file(ts,str(dst/'adapter_model.safetensors'));shutil.copy2(src/'adapter_config.json',dst/'adapter_config.json')
 with safe_open(dst/'adapter_model.safetensors',framework='pt',device='cpu') as f:
  assert all(f.get_tensor(k).equal(t) for k,t in ts.items())
 (Q/'adapter_mapping.json').write_text(json.dumps({'mapping':mapping,'bitwise_equal':True,'source_sha256':hashlib.sha256((src/'adapter_model.safetensors').read_bytes()).hexdigest()},indent=2))
 ac=json.loads((src/'adapter_config.json').read_text());cmd+=['--enable-lora','--max-lora-rank',str(ac['r']),'--lora-modules','qwen35_swegym='+str(dst)]
 routes['qwen35_swegym']={'model':'qwen35_swegym','port':port}
(Q/'routes.json').write_text(json.dumps(routes,indent=2));(Q/(arm+'_server_command.json')).write_text(json.dumps(cmd,indent=2))
os.execvpe(cmd[0],cmd,env)
