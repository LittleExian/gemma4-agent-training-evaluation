"""Bounded dual-GPU checkpoint-25 evaluation with owned cleanup."""
import argparse,fcntl,json,os,signal,sys,time
from pathlib import Path
from gemmacodeagent_mm_pipeline import (atomic_json,bind_json,checked_process,digest,
                                      records_at,validate_records,assert_identity)
import gemmacodeagent_step25_retest_server as service

ROOT=Path('/workspace/gemma4-step25-retest-20261023')
BASE=Path('/workspace/gemma4-notebook-run-20261008')
GATE=Path('/workspace/gemma4-env-preflight-20261016/final-gate.json')
CHECKPOINT=Path('/workspace/gemmacodeagent-remediation-20261022/work/full-epoch1/checkpoint-25')
ARM='gemma4_w4_step25_retest'
GPUS=(0,2)
ENDPOINTS=('http://127.0.0.1:18826/v1','http://127.0.0.1:18828/v1')
STATES=('gpu0','gpu2')
WORKERS=(2,2)
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

def service_proof(root,gpu,record,adapter_sha):
 runtime_path=root/f'reports/runtime-verification-gpu{gpu}.json'
 effect_path=root/f'reports/served-route-effect-gpu{gpu}.json'
 health_path=root/f'services/gpu{gpu}/health.json'
 runtime=json.loads(runtime_path.read_text())
 if (runtime.get('passed') is not True or runtime.get('cuda_graph_captured') is not True
     or runtime.get('requested_content_format')!='auto' or runtime.get('resolved_content_format') not in ('string','openai')
     or runtime.get('seed')!=42 or runtime.get('top_k')!=-1):raise ValueError('Runtime CUDA graph, content format or sampling proof failed')
 health=json.loads(health_path.read_text());effect=json.loads(effect_path.read_text())
 if not health.get('passed') or not effect.get('passed') or effect['adapter_sha256']!=adapter_sha:raise ValueError('Serving proof failed')
 if record.get('physical_gpu_index_at_start')!=gpu or record.get('arm')!=ARM:raise ValueError('Service GPU/arm binding mismatch')
 for proof in (runtime,effect):
  if any(proof.get(key)!=value for key,value in {'pid':record['pid'],'create_time':record['create_time'],'gpu':gpu,'adapter_sha256':adapter_sha}.items()):raise ValueError('Runtime proof belongs to a different service')
 return {'gpu':gpu,'endpoint':ENDPOINTS[GPUS.index(gpu)],'workers':WORKERS[GPUS.index(gpu)],
         'service_pid':record['pid'],'service_create_time':record['create_time'],'command':record['command'],
         'launch_wrapper_sha256':record.get('wrapper_sha256'),
         'resolved_content_format':runtime['resolved_content_format'],'runtime_verification_sha256':digest(runtime_path),
         'health_sha256':digest(health_path),'route_effect_sha256':digest(effect_path)}

def protocol(root,gate,records,checkpoint):
 source=root/'submissions/mm15_6000/adapters'/ARM
 if json.loads((checkpoint/'COMPLETE.json').read_text())!={'step':25,'world_size':2}:raise ValueError('Wrong checkpoint')
 for name in ['adapter_config.json','adapter_model.safetensors']:
  if digest(source/name)!=digest(checkpoint/'adapter'/name):raise ValueError('Adapter copy differs from checkpoint')
 if len(records)!=len(GPUS):raise ValueError('Both model services are required')
 adapter_sha=digest(source/'adapter_model.safetensors')
 services=[service_proof(root,gpu,record,adapter_sha) for gpu,record in zip(GPUS,records)]
 if digest(BASE/'data/tasks.jsonl')!=gate['dataset_sha256']:raise ValueError('Task data drift')
 return {'arm':ARM,'checkpoint':25,'checkpoint_path':str(checkpoint),'serialization':'explicitly selected checkpoint-25; pinned deployment tokenizer/template',
         'gpus':list(GPUS),'workers':list(WORKERS),'total_workers':sum(WORKERS),'service_max_num_seqs':4,'endpoints':list(ENDPOINTS),'deadline_seconds':DEADLINE,
         'minutes':15,'tool_calls':56,'turns':160,'thinking_tokens':6000,'top_k':-1,'seed':42,'cuda_graph_requested':True,
         'requested_content_format':'auto','services':services,
         'source_sha256':{str(p.relative_to(root)):digest(p) for p in sorted((root/'src').glob('*.py'))},
         'gate_sha256':digest(GATE),'dataset_sha256':gate['dataset_sha256'],
         'calibration_artifact_sha256':artifact_hashes(gate['task_bindings']),
         'adapter_sha256':adapter_sha,'budget_contract_upstream':'d53eeddb9da08959de0e3ac9b12c3b4f3e720dad'}

def run(root=ROOT,checkpoint=CHECKPOINT):
 root=root.resolve();checkpoint=checkpoint.resolve();directory=root/'runs/full129-gpu0-gpu2';directory.mkdir(parents=True,exist_ok=True)
 with (directory/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (directory/'controller-status.json').exists():raise ValueError('Use a fresh controller run')
  gate=json.loads(GATE.read_text());tasks=[json.loads(s) for s in (BASE/'data/tasks.jsonl').read_text().splitlines() if s.strip()];ids=[t['instance_id'] for t in tasks]
  if len(ids)!=129 or len(set(ids))!=129:raise ValueError('Expected all 129 unique tasks')
  records=[json.loads((root/'services'/name/'process.json').read_text()) for name in STATES]
  for record in records:assert_identity(record)
  state={'status':'running','started_unix':time.time(),'pid':os.getpid(),'arm':ARM}
  def save(**values):
   state.update(values,updated_unix=time.time());atomic_json(directory/'controller-status.json',state)
  save(phase='binding_protocol')
  try:
   bound=protocol(root,gate,records,checkpoint);bind_json(directory/'protocol.json',bound)
   if json.loads((root/'reports/control-smoke-verification.json').read_text()).get('passed') is not True:raise ValueError('Control smoke not passed')
   command=[str(root/'envs/eval/bin/python'),str(root/'src/gemmacodeagent_mm_run.py'),'--base',str(BASE),'--repo',str(root),'--run-dir',str(directory),'--gate',str(GATE),'--task-runner',str(root/'src/gemmacodeagent_step25_retest_task.py'),'--endpoints',*ENDPOINTS,'--workers',*(str(n) for n in WORKERS),'--fingerprint',str(directory/'protocol.json'),'--ids',*ids]
   env={**os.environ,'G4_MM_ARM':ARM,'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1','OTEL_SDK_DISABLED':'true','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
   save(phase='evaluating',command=command)
   checked_process(command,directory/'runner.log',env,records,timeout=DEADLINE)
   summary=summarize(records_at(directory),ids,gate);atomic_json(directory/'summary.json',summary)
   save(status='completed',phase='complete',summary=summary)
  except BaseException as error:
   save(status='failed',phase='error',error_type=type(error).__name__,error=str(error));raise
  finally:
   primary=sys.exc_info()[1];cleanup_results={};cleanup_errors=[]
   for name in STATES:
    try:cleanup_results[name]=service.cleanup(root/'services'/name)
    except BaseException as error:cleanup_errors.append((name,error))
   save(service_cleanup=cleanup_results)
   if cleanup_errors:
    save(status='cleanup_failed',phase='cleanup_error',cleanup_errors={name:type(error).__name__ for name,error in cleanup_errors})
    if primary is None:raise RuntimeError('Owned service cleanup failed') from cleanup_errors[0][1]

if __name__=='__main__':
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,default=ROOT)
 parser.add_argument('--checkpoint-source',type=Path,required=True)
 args=parser.parse_args()
 if not args.root.is_absolute() or not args.checkpoint_source.is_absolute():parser.error('Absolute root and checkpoint required')
 run(root=args.root,checkpoint=args.checkpoint_source)
