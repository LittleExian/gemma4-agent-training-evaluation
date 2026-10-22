"""Rolling queue: calibration gates release independent candidate jobs; resume is hash-guarded."""
import argparse,concurrent.futures as cf,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from gemmacodeagent_mario_adapter import strict_calibration
R=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',default='valid55',choices=['valid55','paired20']);ap.add_argument('--shards',type=int,default=3);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--arms',nargs='+',default=['v5_6000']);ap.add_argument('--run',required=True);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--ids',nargs='+');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
 assert 0<=a.shard<a.shards and 1<=a.workers<=int(os.environ.get('GEMMA_EVAL_SLOTS','8'))<=8
 assert a.run and all(c.isalnum() or c in '-_' for c in a.run)
 assert len(set(a.arms))==len(a.arms)
 for arm in a.arms:assert arm in ['v5_6000','v5_4096','original_baseline']
 plan=json.loads((R/'protocols'/f'{a.cohort}.json').read_text());ids=plan['tasks']
 if a.ids:assert set(a.ids)<=set(ids);ids=[t for t in ids if t in a.ids]
 ids=[t for i,t in enumerate(ids) if i%a.shards==a.shard]
 print(json.dumps({'tasks':ids,'arms':a.arms,'workers':a.workers}),flush=True)
 if a.dry_run:return
 out=R/'runs'/f'{a.run}-shard{a.shard}';out.mkdir(parents=True,exist_ok=True)
 lock=(out/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest={'cohort':a.cohort,'shards':a.shards,'shard':a.shard,'tasks':ids,'arms':a.arms,'workers':a.workers,'runtime_profile':os.environ.get('GEMMA_RUNTIME_PROFILE','pro6000'),'slot_capacity':int(os.environ.get('GEMMA_EVAL_SLOTS','8')),'files':{str(f.relative_to(R)):digest(f) for directory in ['src','submissions','protocols'] for f in sorted((R/directory).rglob('*')) if f.is_file() and f.suffix!='.pyc'},'tasks_sha256':digest(R/'data/tasks.jsonl')}
 mf=out/'manifest.json'
 if mf.exists():assert json.loads(mf.read_text())==manifest,'Run configuration changed; use a NEW --run'
 else:mf.write_text(json.dumps(manifest,indent=2))
 def execute(task,mode,arm):
  key=f'{arm}__{task}__{mode}';work=out/key;work.mkdir(exist_ok=True);result=work/'task_results.jsonl'
  if result.exists():
   rows=[json.loads(l) for l in result.read_text().splitlines() if l.strip()]
   if len(rows)==1 and rows[0].get('test_exit_code') is not None:return rows[0]
   result.rename(work/f'failed_results_{time.time_ns()}.jsonl')
  profile=plan['profiles'][task]
  cmd=[sys.executable,str(R/'src/run_eval_autodl_matched56.py'),'--mode',mode,'--ids',task,'--wheels',profile['wheels'],'--submission',arm,'--image',profile['image'],'--run',str(work.relative_to(R/'runs'))]
  with (work/'worker.log').open('a') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'ADP_CONCURRENCY':str(a.workers)})
  if proc.returncode or not result.exists():raise RuntimeError(f'{key}: process failure, inspect worker.log')
  rows=[json.loads(l) for l in result.read_text().splitlines() if l.strip()];assert len(rows)==1
  row=rows[0]
  if row.get('test_exit_code') is None:raise RuntimeError(f'{key}: infrastructure failure: {row.get("error_message")}')
  return row
 def calibrate(task):
  gold=execute(task,'gold',a.arms[0]);empty=execute(task,'empty',a.arms[0]);return strict_calibration(gold,empty)
 queue=[('calibrate',t,None) for t in ids];active={};finished=[];failed=[]
 def save():
  (out/'state.json.tmp').write_text(json.dumps({'finished':finished,'failed':failed,'active':list(active.values()),'pending':len(queue),'updated':time.time()},indent=2));os.replace(out/'state.json.tmp',out/'state.json')
 with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
  while queue or active:
   while queue and len(active)<a.workers:
    kind,task,arm=queue.pop(0);f=pool.submit(calibrate,task) if kind=='calibrate' else pool.submit(execute,task,'agent',arm);active[f]=(kind,task,arm)
   save()
   done,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
   for f in done:
    kind,task,arm=active.pop(f)
    try:
     result=f.result()
     if kind=='calibrate':
      if not result:raise RuntimeError('gold must pass AND empty must fail; quarantine this task')
      queue[0:0]=[('agent',task,arm) for arm in a.arms]
     else:finished.append({'task_id':task,'arm':arm,'result':result})
    except Exception as e:failed.append({'task_id':task,'arm':arm,'phase':kind,'error':str(e)})
    print(json.dumps({'finished':len(finished),'failures':len(failed),'pending':len(queue)}),flush=True)
 save()
 if failed:raise SystemExit('Incomplete run: inspect state.json and rerun same command after fixing infrastructure')
if __name__=='__main__':main()
