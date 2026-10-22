"""Bounded single-GPU morning checkpoint-25 evaluation with owned cleanup."""
import fcntl,json,os,signal,sys,time
from pathlib import Path
from gemmacodeagent_mm_pipeline import (atomic_json,bind_json,checked_process,digest,
                                      records_at,validate_records,assert_identity)
import gemmacodeagent_morning25_server as service

ROOT=Path('/workspace/gemma4-morning25-eval-20261022')
BASE=Path('/workspace/gemma4-notebook-run-20261008')
GATE=Path('/workspace/gemma4-env-preflight-20261016/final-gate.json')
CHECKPOINT=Path('/workspace/gemmacodeagent-remediation-20261022/work/full-epoch1/checkpoint-25')
ARM='gemma4_w4_morning25'
ENDPOINT='http://127.0.0.1:18825/v1'
DEADLINE=24*3600

def artifact_hashes(value):
 found={}
 if isinstance(value,dict):
  if set(('path','sha256'))<=value.keys():
   path=Path(value['path'])
   if not path.is_file() or digest(path)!=value['sha256']:raise ValueError('Calibration artifact changed')
   found[str(path)]=value['sha256']
  for item in value.values():found.update(artifact_hashes(item))
 elif isinstance(value,list):
  for item in value:found.update(artifact_hashes(item))
 return found

def summarize(records,ids,gate):
 validate_records(records,ids,allow_excluded=True)
 excluded=sorted(row['instance_id'] for row in records if row['status']=='preflight_excluded')
 expected=sorted(gate['quarantined_ids']+gate['nondiscriminating_ids'])
 if excluded!=expected or len(records)!=129 or any(r.get('resolved') for r in records if r['status']=='preflight_excluded'):raise ValueError('Exceptional task accounting mismatch')
 return {'total':129,'eligible':126,'excluded':excluded,'resolved':sum(r.get('resolved') is True for r in records),
         'timeouts':sum(r['status']=='agent_timeout' for r in records)}

def protocol(root,gate,record):
 source=root/'submissions/mm15_6000/adapters'/ARM
 if json.loads((CHECKPOINT/'COMPLETE.json').read_text())!={'step':25,'world_size':2}:raise ValueError('Wrong checkpoint')
 for name in ['adapter_config.json','adapter_model.safetensors']:
  if digest(source/name)!=digest(CHECKPOINT/'adapter'/name):raise ValueError('Adapter copy differs from checkpoint')
 health=json.loads((root/'services/gpu0-r2/health.json').read_text())
 effect=json.loads((root/'reports/served-route-effect.json').read_text())
 if not health.get('passed') or not effect.get('passed') or effect['adapter_sha256']!=digest(source/'adapter_model.safetensors'):raise ValueError('Serving proof failed')
 if digest(BASE/'data/tasks.jsonl')!=gate['dataset_sha256']:raise ValueError('Task data drift')
 return {'arm':ARM,'checkpoint':25,'checkpoint_path':str(CHECKPOINT),'serialization':'legacy morning pre-prefix-v2 adapter; evaluated using corrected deployment tokenizer/template',
         'gpus':[0],'slots':4,'endpoint':ENDPOINT,'deadline_seconds':DEADLINE,
         'minutes':15,'tool_calls':56,'turns':160,'thinking_tokens':6000,'top_k':64,
         'source_sha256':{str(p.relative_to(root)):digest(p) for p in sorted((root/'src').glob('*.py'))},
         'gate_sha256':digest(GATE),'dataset_sha256':gate['dataset_sha256'],
         'calibration_artifact_sha256':artifact_hashes(gate['task_bindings']),
         'adapter_sha256':digest(source/'adapter_model.safetensors'),
         'health_sha256':digest(root/'services/gpu0-r2/health.json'),'route_effect_sha256':digest(root/'reports/served-route-effect.json'),
         'service_pid':record['pid'],'service_create_time':record['create_time'],
         'budget_contract_upstream':'d53eeddb9da08959de0e3ac9b12c3b4f3e720dad'}

def run():
 root=ROOT;directory=root/'runs/full129-single-gpu0-r3';directory.mkdir(parents=True,exist_ok=True)
 with (directory/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (directory/'controller-status.json').exists():raise ValueError('Use a fresh controller run')
  gate=json.loads(GATE.read_text());tasks=[json.loads(s) for s in (BASE/'data/tasks.jsonl').read_text().splitlines() if s.strip()];ids=[t['instance_id'] for t in tasks]
  if len(ids)!=129 or len(set(ids))!=129:raise ValueError('Expected all 129 unique tasks')
  record=json.loads((root/'services/gpu0-r2/process.json').read_text());assert_identity(record)
  state={'status':'running','started_unix':time.time(),'pid':os.getpid(),'arm':ARM}
  def save(**values):
   state.update(values,updated_unix=time.time());atomic_json(directory/'controller-status.json',state)
  save(phase='binding_protocol')
  try:
   bound=protocol(root,gate,record);bind_json(directory/'protocol.json',bound)
   if json.loads((root/'reports/control-smoke-verification.json').read_text()).get('passed') is not True:raise ValueError('Control smoke not passed')
   command=[str(root/'envs/eval/bin/python'),str(root/'src/gemmacodeagent_mm_run.py'),'--base',str(BASE),'--repo',str(root),'--run-dir',str(directory),'--gate',str(GATE),'--task-runner',str(root/'src/gemmacodeagent_morning25_task.py'),'--endpoints',ENDPOINT,'--workers','4','--fingerprint',str(directory/'protocol.json'),'--ids',*ids]
   env={**os.environ,'G4_MM_ARM':ARM,'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1','OTEL_SDK_DISABLED':'true','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
   save(phase='evaluating',command=command)
   checked_process(command,directory/'runner.log',env,[record],timeout=DEADLINE)
   summary=summarize(records_at(directory),ids,gate);atomic_json(directory/'summary.json',summary)
   save(status='completed',phase='complete',summary=summary)
  except BaseException as error:
   save(status='failed',phase='error',error_type=type(error).__name__,error=str(error));raise
  finally:
   primary=sys.exc_info()[1]
   try:save(service_cleanup=service.cleanup(root/'services/gpu0-r2'))
   except BaseException as error:
    save(status='cleanup_failed',phase='cleanup_error',cleanup_error_type=type(error).__name__)
    if primary is None:raise

if __name__=='__main__':
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 run()
