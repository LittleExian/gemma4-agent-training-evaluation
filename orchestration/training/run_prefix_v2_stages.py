"""Execute the authorized fresh prefix-v2 smoke gates and one-epoch training."""
import hashlib,json,os,pathlib,subprocess,time,sys,signal
R=pathlib.Path('/workspace/gemmacodeagent-prefix-v2-20261022')
OLD=pathlib.Path('/workspace/gemmacodeagent-weighted-lora-20261020')
PY=str(OLD/'.venv-w4/bin/python')
sys.path.insert(0,str(R/'launcher'))
import remediation_training_supervisor as sup
from gemmacodeagent_owned_cleanup import release_owned_model,cleanup_secondary
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
state={'started_unix':time.time(),'status':'running','phase':'checking_gates','stages':[]}
def save():sup.atomic_json(R/'reports/training-pipeline-progress.json',state)
def hashes():return {str(p.relative_to(R/'src')):sup.fingerprint(p) for p in sorted((R/'src').rglob('*.py'))}
def assert_frozen():
 snapshot=json.loads((R/'reports/source-snapshot.json').read_text())
 assert hashes()==snapshot['source_sha256'],'Source changed after snapshot'
 gate=json.loads((R/'reports/CPU_PREFLIGHT_PASS.json').read_text())
 for split in ('train','validation'):
  expected=gate['dataset'][split]['sha256']
  assert sup.fingerprint(R/'tokenized'/(split+'.jsonl'))==expected,'Data changed after CPU gate'
def manifest(name,data,scope):
 assert_frozen()
 path=R/'reports'/(name+'-manifest.json')
 body={'code_commit':json.loads((R/'reports/source-snapshot.json').read_text())['code_commit'],'source_sha256':hashes(),'training_data_sha256':sup.fingerprint(data),'serialization_version':'gemma4-deployment-prefix-v2','recipe':{'world_size':2,'accum':1 if scope=='smoke' else 4,'global_batch':2 if scope=='smoke' else 8,'objective':'task_mean','learning_rate':2e-5,'seed':42,'fresh_lora':True,'steps':1 if scope=='smoke' else 356,'save_every':1 if scope=='smoke' else 5}}
 sup.atomic_json(path,body)
 return path

def single_smoke():
 import psutil
 assert_frozen()
 out=R/'work/smoke-single-longest';assert not out.exists();out.mkdir()
 command=[PY,str(R/'launcher/remediation_training_supervisor.py'),'--worker','--cap-gib','86','--train-script',str(R/'src/train.py'),'--data',str(R/'tokenized/train.jsonl'),'--output',str(out),'--revision',sup.REVISION,'--objective','task_mean','--accum','4','--lr','2e-5','--seed','42','--attention','sdpa','--smoke','--save-every','1','--wandb-mode','disabled']
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
 sup.select_gpus(inventory,'');sup.validate_host_memory(psutil.virtual_memory().available)
 env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'3','LOCAL_RANK':'0','RANK':'0','WORLD_SIZE':'1','PYTHONNOUSERSITE':'1','HF_HOME':str(R/'hf_cache'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','WANDB_MODE':'disabled'}
 for k in ['WANDB_API_KEY','WANDB_RUN_ID','WANDB_RESUME']:env.pop(k,None)
 state.update(phase='smoke-single-longest',output=str(out));save()
 start=time.time()
 with (out/'train.log').open('w') as f:
  boot=time.clock_gettime(time.CLOCK_BOOTTIME)
  p=subprocess.Popen(command,cwd=R/'src',env=env,stdout=f,stderr=f,start_new_session=True,stdin=subprocess.DEVNULL)
  record={'pid':p.pid,'create_time':psutil.Process(p.pid).create_time(),'uid':os.getuid(),'launch_boottime':boot}
  try:
   sup.atomic_json(out/'process.json',record)
   assert p.wait(timeout=7200)==0,'Single-card smoke failed'
  finally:cleanup_secondary(lambda:release_owned_model(record,start),sys.exc_info()[1])
 smoke=json.loads((out/'SMOKE_PASS.json').read_text())
 assert smoke['passed'] is True and smoke['world_size']==1 and smoke['steps']==1 and smoke['reload_max_abs_logit_difference']==0
 assert json.loads((out/'base_audit.json').read_text())==json.loads((out/'base_audit_after.json').read_text())
 assert json.loads((out/'base_audit.json').read_text())['passed'] is True
 cp=out/'checkpoint-1'
 assert json.loads((cp/'COMPLETE.json').read_text())=={'step':1,'world_size':1}
 assert json.loads((cp/'adapter_audit.json').read_text())['passed'] is True
 assert (cp/'rng.rank0.pt').stat().st_size>0
 state['stages'].append({'name':'smoke-single-longest','passed':True});save()

def stage(name,data,scope):
 import psutil
 out=R/'work'/name
 command=[PY,str(R/'launcher/remediation_training_supervisor.py'),'--root',str(R),'--source',str(R/'src'),'--data',str(data),'--output',str(out),'--scope',scope,'--gpus','2,3','--cap-gib','86','--hf-home',str(R/'hf_cache'),'--python',PY,'--credential-file','/workspace/.credentials/weighted-lora-20261020/wandb-api-key','--wandb-entity','eeexian1-nanjing-university','--run-name','gemma4-w4-prefix-v2-'+name,'--manifest',str(manifest(name,data,scope))]
 state.update(phase=name,output=str(out));save()
 with (R/'reports'/(name+'-supervisor.log')).open('w') as f:
  start=time.time();boot=time.clock_gettime(time.CLOCK_BOOTTIME)
  p=subprocess.Popen(command,stdout=f,stderr=f,stdin=subprocess.DEVNULL,start_new_session=True)
  record={'pid':p.pid,'create_time':psutil.Process(p.pid).create_time(),'uid':os.getuid(),'launch_boottime':boot}
  try:
   sup.atomic_json(R/'reports'/(name+'-supervisor-process.json'),record)
   state['supervisor_pid']=p.pid;save()
   code=p.wait()
  finally:cleanup_secondary(lambda:release_owned_model(record,start),sys.exc_info()[1])
 result=json.loads((out/'exit.json').read_text()) if (out/'exit.json').exists() else {}
 assert code==0 and result.get('state')=='completed',(name,result)
 state['stages'].append({'name':name,'passed':True});save()
try:
 save()
 gate=json.loads((R/'reports/CPU_PREFLIGHT_PASS.json').read_text());assert gate['passed'] is True
 parity=json.loads((R/'reports/serving-prefix-verification.json').read_text());assert parity['passed'] is True and parity['probe_completed'] is True and parity['gpu_cleanup_verified'] is True
 assert __import__('shutil').disk_usage(R).free>150*2**30
 single_smoke()
 seen=set()
 for kind in ('most-supervised','longest'):
  item=gate['stress_samples'][kind]
  assert sup.fingerprint(pathlib.Path(item['path']))==item['sha256'],'Stress data changed after CPU gate'
  if item['sample_id'] not in seen:
   stage('smoke-two-'+kind,pathlib.Path(item['path']),'smoke');seen.add(item['sample_id'])
 stage('full-epoch1',R/'tokenized/train.jsonl','full')
 state.update(status='completed',phase='complete',finished_unix=time.time());save()
except BaseException as error:
 state.update(status='failed',error_type=type(error).__name__,error=str(error),finished_unix=time.time());save();raise
