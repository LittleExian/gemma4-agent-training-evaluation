from pathlib import Path
import os,time,json,fcntl,atexit,sys
ROOT=Path(os.environ['CODEAGENT_ROOT'])/'runs/mm_eval_slots';ROOT.mkdir(exist_ok=True)
def alive(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
class Slot:
 def __init__(self):self.pid=str(os.getpid());self.priority=0 if any(x.startswith('autodl_matched56/') for x in sys.argv) else 1
 def __enter__(self):
  while True:
   with (ROOT/'lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);f=ROOT/'state.json';s=json.loads(f.read_text()) if f.exists() else {'holders':{},'waiters':{}}
    for k in ['holders','waiters']:s[k]={p:v for p,v in s[k].items() if alive(p)}
    s['waiters'][self.pid]={'priority':self.priority,'argv':sys.argv[1:]}
    can=self.pid in s['holders'] or (len(s['holders'])<int(os.environ.get('MM_SLOT_LIMIT','8')) and self.priority==min(v['priority'] for v in s['waiters'].values()))
    if can:s['holders'][self.pid]=s['waiters'].pop(self.pid)
    f.write_text(json.dumps(s))
    if can:
     with (ROOT/'occupancy.jsonl').open('a') as out:out.write(json.dumps(dict(unix=time.time(),pid=self.pid,active=len(s['holders']),priority=self.priority))+'\n')
     atexit.register(self.release);return self
   time.sleep(1)
 def release(self):
  with (ROOT/'lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);f=ROOT/'state.json';s=json.loads(f.read_text());s['holders'].pop(self.pid,None);s['waiters'].pop(self.pid,None);f.write_text(json.dumps(s))
 def __exit__(self,*args):self.release()
