import argparse,concurrent.futures as cf,json,os,time,subprocess,sys,hashlib,shutil,signal,fcntl
from pathlib import Path
S=Path(__file__).resolve().parent;P=Path(os.environ['CODEAGENT_ROOT']);Q=P/'research/distributed_model_comparison';O=P/'runs/distributed_model_comparison';O.mkdir(parents=True,exist_ok=True);Q.mkdir(parents=True,exist_ok=True)
PYTHON=str(P/'envs/eval/bin/python');specs=json.loads((S/'models.json').read_text())
ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['pilot','pilot16','all'],default='all');ap.add_argument('--arms',nargs='+',default=list(specs));ap.add_argument('--concurrency',type=int,default=16);ap.add_argument('--worker-index',type=int,default=0);ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
assert 1<=args.workers<=4 and 0<=args.worker_index<args.workers
assert set(args.arms)<=set(specs)
assert 'qwen35_swegym' not in args.arms or 'qwen35_base' in args.arms, 'LoRA requires shared base arm'
O=O/('worker_'+str(args.worker_index)+'_of_'+str(args.workers));O.mkdir(parents=True,exist_ok=True)
lock=(O/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[json.loads(x) for x in (P/'data/tasks.jsonl').read_text().splitlines()];allids=[x.get('instance_id',x.get('task_id')) for x in rows]
ids=json.loads((S/('pilot16.json' if args.scope=='pilot16' else 'pilot8.json')).read_text())['tasks'] if args.scope!='all' else json.loads((S/'full129.json').read_text())['tasks']
ids=ids[args.worker_index::args.workers]
assert len(set(ids))==len(ids) and set(ids)<=set(allids)
cat={x['task_id']:x for x in json.loads((S/'environment_catalog.json').read_text())['tasks']}
old=json.loads((S/'calibrated_profiles.json').read_text())
sub=P/'submissions/mm15_6000'
if not sub.exists():shutil.copytree(S/'submission',sub)
sourcehash={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in S.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml'] and 'reports' not in p.parts and p.name not in ['pilot_results.json','validated_sources.json']}
manifest={'worker_index':args.worker_index,'workers':args.workers,'scope':args.scope,'tasks':ids,'arms':args.arms,'total':len(ids)*len(args.arms),'max_concurrency':args.concurrency,'minutes':15,'thinking_budget':6000,'tools':56,'turns':160,'output_tokens':8192,'source_hashes':sourcehash,'dtype':'FP8 weights; BF16 activations/KV as checkpoint specifies','speculative_decoding':False,'started':time.time(),'failure_rule':'stop new tasks on infrastructure error; preserve all outcomes; no selective retries','milestones':[int((len(ids)*len(args.arms)*x+3)//4) for x in [1,2,3,4]]}
mp=O/(args.scope+'_protocol.json')
if mp.exists():
 prior=json.loads(mp.read_text())
 for k in ['tasks','arms','source_hashes','minutes','thinking_budget']:assert prior[k]==manifest[k],('Protocol drift',k)
else:mp.write_text(json.dumps(manifest,indent=2))
state={'worker_index':args.worker_index,'workers':args.workers,'phase':'preparing','completed':[],'active':[],'total':manifest['total'],'started':time.time(),'pid':os.getpid(),'failures':[]}
def save():
 state['updated']=time.time();tmp=O/'state.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(O/'state.json');(O/(args.scope+'_state.json')).write_text(json.dumps(state,indent=2))
def process(cmd,log,env=None):
 f=log.open('a');return subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env or os.environ,stdin=subprocess.DEVNULL,start_new_session=True)
def runjob(arm,t,mode='agent'):
 rel=f'distributed_model_comparison/{args.scope}/{arm}/{t}' if mode=='agent' else f'distributed_model_comparison/calibration/{mode}/{t}'
 out=P/'runs'/rel;out.mkdir(parents=True,exist_ok=True);result=out/'task_results.jsonl';profile=old.get(t,cat[t]);snapshot=P/'data/matched56_snapshots'
 if not (snapshot/(t+'.tgz')).exists():snapshot=P/'data/competition_all/snapshots'
 env={**os.environ,'MM_PROXY_BASE':f'http://127.0.0.1:8013/{arm}/{args.scope}__{t}/v1','MM_SNAPSHOTS':str(snapshot),'ADP_CONCURRENCY':str(args.concurrency),'MM_SLOT_LIMIT':str(args.concurrency)}
 if not result.exists():
  cmd=[PYTHON,str(S/'run_task.py'),'--mode',mode,'--submission',sub.name,'--run',rel,'--wheels',profile['wheels'],'--image',profile['image'],'--ids',t]
  p=process(cmd,out/'worker.log',env)
  try:rc=p.wait(timeout=2100)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);raise RuntimeError('Runner infrastructure watchdog '+t)
  assert rc==0,('runner exit',t,rc)
 rr=[json.loads(x) for x in result.read_text().splitlines()];assert len(rr)==1;row=rr[0]
 assert row.get('test_exit_code') is not None and not row.get('traceback'),('infrastructure',arm,t,row.get('error_message'))
 err=(row.get('error_message') or '').lower();assert not any(x in err for x in ['apiconnectionerror','enginedead','cuda out of memory','connection refused']),err
 return {'scope':args.scope,'arm':arm,'task':t,'mode':mode,'result':row,'environment_previously_valid':cat[t]['valid'],'finished':time.time()}
def batch(jobs,n):
 with cf.ThreadPoolExecutor(max_workers=n) as pool:
  pending=iter(jobs);active={}
  while True:
   while len(active)<n and not state['failures']:
    try:j=next(pending)
    except StopIteration:break
    active[pool.submit(runjob,*j)]=j
   state['active']=list(active.values());save()
   if not active:break
   done,_=cf.wait(active,return_when=cf.FIRST_COMPLETED)
   for f in done:
    j=active.pop(f)
    try:
     row=f.result();state['completed'].append(row)
     with (O/'units.jsonl').open('a') as log:log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
    except Exception as e:state['failures'].append({'job':j,'error':repr(e)})
   save();print('PROGRESS',len(state['completed']),manifest['total'],len(state['failures']),flush=True)
  assert not state['failures'],state['failures']
server=None;proxy=None
try:
 save();proxy=process([PYTHON,str(S/'proxy.py')],Q/'proxy.log')
 remaining=[a for a in args.arms if a!='qwen35_swegym']
 for arm in remaining:
  group=[arm]+(['qwen35_swegym'] if arm=='qwen35_base' and 'qwen35_swegym' in args.arms else [])
  state['phase']='download';state['model_group']=group;save()
  for a in group:
   if a=='gemma4_kaggle':
    assert (P/'models/gemma-4-31b-it-qat-w4a16-ct/config.json').exists(), 'Run parent setup.sh to obtain Kaggle checkpoint'
    continue
   path=Q/'models'/specs[a]['directory']
   if not (path/'DOWNLOAD_COMPLETE').exists():
    d=process([PYTHON,str(S/'download.py'),a],Q/(a+'_download.log'));assert d.wait()==0,('download failure',a)
  index=remaining.index(arm)
  if os.environ.get('MM_PREFETCH','0')=='1' and index+1<len(remaining) and remaining[index+1]!='gemma4_kaggle':
   nxt=remaining[index+1];src=Q/(specs[nxt]['directory']+'_source.json')
   required=sum(x.get('size',0) for x in json.loads(src.read_text())['files']) if src.exists() else 35*1024**3
   if shutil.disk_usage(Q).free>required+12*1024**3:
    pre=process([PYTHON,str(S/'download.py'),nxt],Q/(nxt+'_prefetch.log'));state['prefetch']={'arm':nxt,'pid':pre.pid}
  state['phase']='model_health';save();server=process([PYTHON,str(S/'serve.py'),arm],Q/(arm+'_server.log'))
  h=process([PYTHON,str(S/'health.py'),arm,str(min(args.concurrency,len(ids)*len(group)))],Q/(arm+'_health.log'))
  while h.poll() is None:
   if server.poll() is not None:
    os.killpg(h.pid,signal.SIGTERM);raise RuntimeError('Model server exited during health gate: '+arm)
   time.sleep(2)
  assert h.returncode==0,('model health failure',arm)
  capacity=json.loads((Q/(arm+'_health.json')).read_text())['concurrency'];n=min(args.concurrency,capacity)
  state['phase']='calibration';save()
  with cf.ThreadPoolExecutor(max_workers=8) as pool:
   checks=list(pool.map(lambda j:runjob('calibration',j[0],j[1]),[(t,m) for t in ids for m in ['gold','empty']]))
  if args.scope!='all':
   assert all(bool(x['result']['resolved'])==(x['mode']=='gold') for x in checks),'Pilot calibration failure'
  (O/'calibration_audit.json').write_text(json.dumps(checks,indent=2))
  state['phase']='agents';save();batch([(a,t) for t in ids for a in group],n)
  state['phase']='model_complete';save();os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=90);server=None
  # This experiment uses a disposable, reproducible weight cache on the data disk.
  # Retain all source revisions, hashes, configs, prompts, traces and results.
  if os.environ.get('MM_KEEP_WEIGHTS','0')!='1' and arm!='gemma4_kaggle':
   for a in group:
    cache=(Q/'models'/specs[a]['directory']).resolve();assert cache.parent==(Q/'models').resolve()
    removed=[]
    for f in cache.iterdir():
     if f.is_file() and (f.name.endswith(('.safetensors','.ranges','.part','.chunks')) or f.name=='DOWNLOAD_COMPLETE'):
      removed.append({'name':f.name,'bytes':f.stat().st_size});f.unlink()
    with (Q/'weight_cache_evictions.jsonl').open('a') as log:log.write(json.dumps({'arm':a,'files':removed,'time':time.time()})+'\n')

 state['phase']='complete';state['finished']=time.time();save()
 report={a:{'n':sum(x['arm']==a for x in state['completed']),'resolved':sum(x['arm']==a and x['result']['resolved'] for x in state['completed'])} for a in args.arms};(O/'summary.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps(report),flush=True)
except Exception as e:
 state['phase']='blocked';state['fatal']=repr(e);save();raise
finally:
 if server and server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
 if proxy and proxy.poll() is None:os.killpg(proxy.pid,signal.SIGTERM)
