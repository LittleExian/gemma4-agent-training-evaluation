"""Configurable host-wide flock leases; crash-safe, no PID reuse or stale JSON leases."""
import fcntl,os,time
from pathlib import Path
class Slot:
 def __enter__(self):
  root=Path(os.environ.get('GEMMA_SLOT_DIR','/root/autodl-tmp/gemma-eval-slots'));root.mkdir(parents=True,exist_ok=True)
  capacity=int(os.environ.get('GEMMA_EVAL_SLOTS','8'))
  if not 1<=capacity<=8:raise ValueError('GEMMA_EVAL_SLOTS must be 1..8')
  with (root/'capacity.lock').open('a+') as config_lock:
   fcntl.flock(config_lock,fcntl.LOCK_EX)
   config=root/'capacity.txt'
   if config.exists() and int(config.read_text())!=capacity:
    raise RuntimeError('Slot capacity differs from other processes. Stop ALL benchmark processes before removing capacity.txt and changing capacity.')
   config.write_text(str(capacity))
  while True:
   for i in range(capacity):
    f=(root/str(i)).open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();continue
    self.file=f;return self
   time.sleep(.25)
 def __exit__(self,*args):self.file.close()
