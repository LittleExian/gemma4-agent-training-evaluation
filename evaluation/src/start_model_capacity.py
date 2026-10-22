import os, json, subprocess, argparse
from pathlib import Path
from adk_submission import VllmConfig,VllmServer
parser=argparse.ArgumentParser()
parser.add_argument('--profile',choices=['pro6000','4090-48g'],default='pro6000')
parser.add_argument('--concurrency',type=int,choices=[1,2,3,4],default=1,help='4090 profile only')
parser.add_argument('--dry-run',action='store_true')
parser.add_argument('--port',type=int,default=8000)
parser.add_argument('--state-dir',type=Path)
a=parser.parse_args()
small=a.profile=='4090-48g'
p=Path(__file__).resolve().parents[1]
import socket
with socket.socket() as probe:
 if not a.dry_run and probe.connect_ex(('127.0.0.1',a.port))==0:
  raise SystemExit(f'Port {a.port} is already occupied. Reuse or explicitly stop the existing model service.')
for directory in ['runs','logs','tmp','cache']: (p/directory).mkdir(exist_ok=True)
os.environ.setdefault('HF_HOME',str(p/'cache/huggingface'))
os.environ.setdefault('XDG_CACHE_HOME',str(p/'cache'))
os.environ.setdefault('TMPDIR',str(p/'tmp'))
name="gemma-4-31b-it-qat-w4a16-ct"
cfg=VllmConfig(model=str(p/"models"/name),host="127.0.0.1",port=a.port,
tool_call_parser="gemma4",reasoning_parser="gemma4",max_model_len=32768,
dtype="bfloat16",gpu_memory_utilization=.90,enable_auto_tool_choice=True,
enable_lora=False,scheduler_reserve_full_isl=False,tensor_parallel_size=1,
limit_mm_per_prompt='{"image":0,"audio":0,"video":0}',
extra_args=["--served-model-name",name,"--max-num-seqs",str(a.concurrency if small else 32),"--max-num-batched-tokens",str(1024 if small else 4096)]+(["--enforce-eager","--no-enable-prefix-caching"] if small else []))
server=VllmServer(cfg)
cmd=server.build_cmd()
if a.dry_run:
 print(json.dumps({'profile':a.profile,'command':cmd},indent=2));raise SystemExit(0)
if small:
 import torch
 if not torch.cuda.is_available():raise SystemExit('CUDA GPU unavailable')
 if torch.cuda.get_device_properties(0).total_memory < 44*1024**3:
  raise SystemExit('This profile requires a GPU with approximately 48GB VRAM; not a 24GB card.')
state=a.state_dir or p/'runs'
state.mkdir(parents=True,exist_ok=True)
(state/'server_profile.json').write_text(json.dumps({'profile':a.profile,'eval_slots':a.concurrency if small else 8,'max_model_len':32768,'enforce_eager':small},indent=2))
(state/"server_command.json").write_text(json.dumps(cmd,indent=2))
with ((state/"vllm.log") if a.state_dir else (p/"logs/vllm.log")).open("a") as log:
 proc=subprocess.Popen(cmd,env=server.build_env(),stdout=log,stderr=log,start_new_session=True)
(state/"vllm.pid").write_text(str(proc.pid))
print("vllm_pid",proc.pid,flush=True)
