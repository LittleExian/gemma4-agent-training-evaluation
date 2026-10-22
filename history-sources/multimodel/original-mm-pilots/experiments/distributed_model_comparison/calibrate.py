import os,json,subprocess,concurrent.futures as cf,shutil,time
from pathlib import Path
P=Path(os.environ['CODEAGENT_ROOT']);S=Path(__file__).parent;Q=P/'research/distributed_model_comparison';ids=json.loads((S/os.environ.get('MM_TASK_MANIFEST','pilot8.json')).read_text())['tasks'];profiles=json.loads((P/'research/autodl_matched56/protocol.json').read_text())['profiles']
sub=P/'submissions/mm15_6000'
if not sub.exists():shutil.copytree(S/'submission',sub)
def run(job):
 t,mode=job;rel=f'distributed_model_comparison/calibration/{mode}/{t}';out=P/'runs'/rel;out.mkdir(parents=True,exist_ok=True);result=out/'task_results.jsonl';profile=profiles[t]
 if not result.exists():
  with (out/'worker.log').open('a') as log:
   r=subprocess.run([str(P/'envs/eval/bin/python'),str(S/'run_task.py'),'--mode',mode,'--ids',t,'--submission',sub.name,'--run',rel,'--wheels',profile['wheels'],'--image',profile['image']],env={**os.environ,'ADP_CONCURRENCY':'8','MM_SLOT_LIMIT':'8'},stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,(t,mode,r.returncode)
 d=json.loads(result.read_text());assert bool(d['resolved'])==(mode=='gold'),(t,mode,d.get('error_message'));return {'task':t,'mode':mode,'passed':True}
with cf.ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(run,[(t,m) for t in ids for m in ['gold','empty']]))
(Q/'pilot_calibration.json').write_text(json.dumps({'passed':True,'rows':rows,'finished':time.time()},indent=2));print('CALIBRATION_PASS',len(rows),flush=True)
