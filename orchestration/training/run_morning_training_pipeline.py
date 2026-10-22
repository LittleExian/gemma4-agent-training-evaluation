"""Run verified stress controls followed by the user-authorized one-epoch recipe."""
import hashlib,json,pathlib,subprocess,time,os,traceback
R=pathlib.Path('/workspace/gemmacodeagent-remediation-20261022')
OLD=pathlib.Path('/workspace/gemmacodeagent-weighted-lora-20261020')
PY=str(OLD/'.venv-w4/bin/python');CAP=86
state={'started_unix':time.time(),'phase':'checking_preflight','status':'running','gpu_indices':[2,3],'cap_gib':CAP,'steps':356,'stages':[]}
def save():
 p=R/'reports/training-pipeline-progress.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def stage(name,data,scope):
 out=R/'work'/name
 manifest=R/'reports'/(name+'-manifest.json')
 record={'code_commit':'d60fda8eacc602f7d1ad4015c60cdc714523afe1','training_data_sha256':sha(data),'source_sha256':{str(p.relative_to(R/'src')):sha(p) for p in sorted((R/'src').rglob('*.py'))},'recipe':{'objective':'task_mean','world_size':2,'accum':4,'global_batch':8,'steps':1 if scope=='smoke' else 356,'scope':scope,'fresh_lora':True,'model_revision':'52f3f65bc7a02d555763bc923bd1d9094898219d','gpu_indices':[2,3],'torch_allocator_cap_gib':CAP},'ce_adaptation_sha256':sha(R/'reports/ce-checkpoint.patch')}
 manifest.write_text(json.dumps(record,indent=2)+'\n')
 command=[PY,str(R/'launcher/remediation_training_supervisor.py'),'--root',str(R),'--source',str(R/'src'),'--data',str(data),'--output',str(out),'--scope',scope,'--gpus','2,3','--cap-gib',str(CAP),'--hf-home',str(R/'hf_cache'),'--python',PY,'--credential-file','/workspace/.credentials/weighted-lora-20261020/wandb-api-key','--wandb-entity','eeexian1-nanjing-university','--run-name','gemma4-w4-remediation-'+name,'--manifest',str(manifest)]
 state.update(phase=name,output=str(out));save();print('Starting '+name,flush=True)
 with (R/'reports'/(name+'-supervisor.log')).open('w') as f:p=subprocess.Popen(command,stdout=f,stderr=f,stdin=subprocess.DEVNULL)
 state['supervisor_pid']=p.pid;save();code=p.wait();result=json.loads((out/'exit.json').read_text()) if (out/'exit.json').exists() else {}
 if code or result.get('state')!='completed':raise RuntimeError(name+' failed: '+str(result))
 state['stages'].append({'name':name,'scope':scope,'output':str(out),'passed':True,'finished_unix':time.time()});save()
try:
 save()
 for name in ['CPU_PREFLIGHT_PASS.json','base-weight-verification.json','nccl-probe.json','streamed-ce-validation.json']:
  assert json.loads((R/'reports'/name).read_text())['passed'] is True,name
 assert __import__('shutil').disk_usage(R).free>150*2**30
 report=json.loads((R/'reports/CPU_PREFLIGHT_PASS.json').read_text())
 seen=set()
 for kind in ['most-supervised','longest']:
  item=report['stress_samples'][kind]
  if item['sample_id'] not in seen:stage('smoke-'+kind+'-streamed-cap86',pathlib.Path(item['path']),'smoke');seen.add(item['sample_id'])
 stage('full-epoch1',R/'tokenized/train.jsonl','full')
 state.update(status='completed',phase='complete',finished_unix=time.time());save()
except BaseException as e:
 state.update(status='failed',error_type=type(e).__name__,error=str(e),finished_unix=time.time());save();traceback.print_exc();raise
