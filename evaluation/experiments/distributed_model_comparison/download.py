import os,json,time,concurrent.futures as cf,hashlib,threading,math,sys
from pathlib import Path
import requests,fcntl
P=Path(os.environ['CODEAGENT_ROOT']);Q=P/'research/distributed_model_comparison';M=Q/'models';M.mkdir(parents=True,exist_ok=True);_local=threading.local()
def meta(repo):
 for host in ['https://hf-mirror.com','https://huggingface.co']:
  try:
   r=requests.get(host+'/api/models/'+repo+'?blobs=true',timeout=35);r.raise_for_status();return host,r.json()
  except Exception as e:print('metadata_retry',repo,host,type(e).__name__,flush=True)
 raise RuntimeError(repo)
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def _download(repo,sub,name):
 source=Q/(name+'_source.json')
 pinned=Path(__file__).parent/'sources'/(name+'_source.json')
 if not source.exists() and pinned.exists():source.write_bytes(pinned.read_bytes())
 if source.exists():
  prior=json.loads(source.read_text());host='https://hf-mirror.com';d={'sha':prior['revision'],'siblings':prior['files']}
 else:host,d=meta(repo)
 rev=d['sha'];fs=[x for x in d['siblings'] if x['rfilename'].startswith(sub) and '/' not in x['rfilename'][len(sub):] and not x['rfilename'].endswith(('.md','.gguf'))]
 if name=='qwen35_adapter':fs=[x for x in fs if x['rfilename'] in ['adapter_config.json','adapter_model.safetensors','chat_template.jinja']]
 target=M/name;target.mkdir(exist_ok=True);(Q/(name+'_source.json')).write_text(json.dumps({'repo':repo,'revision':rev,'subfolder':sub,'files':fs},indent=2))
 def one(info):
  fn=info['rfilename'];dest=target/Path(fn).name;lfs=info.get('lfs',{});expected=lfs.get('sha256');size=info.get('size',lfs.get('size',0))
  if dest.exists() and (not expected or digest(dest)==expected):return
  hf=host+'/'+repo+'/resolve/'+rev+'/'+fn
  msrepo=next((x.get('modelscope_repo',repo) for x in json.loads((Path(__file__).parent/'models.json').read_text()).values() if x['repo']==repo),repo)
  ms='https://modelscope.cn/models/'+msrepo+'/resolve/master/'+fn+'?revision_check='+rev
  url=hf
  if expected:
   try:
    r=requests.get(ms,headers={'Range':'bytes=0-0'},stream=True,timeout=30);r.raise_for_status();print('MS_PROBE',name,r.status_code,r.headers.get('Content-Range'),size,flush=True)
    if r.status_code==206 and r.headers.get('Content-Range','').endswith('/'+str(size)):url=ms
   except Exception as e:print('MS_PROBE_ERROR',type(e).__name__,str(e)[:150],flush=True)
  if size>32*1024**2 and expected:
   part=dest.with_suffix(dest.suffix+'.ranges');journal=dest.with_suffix(dest.suffix+'.chunks');chunk=8*1024**2;lock=threading.Lock()
   done=set(int(x) for x in journal.read_text().splitlines()) if journal.exists() and part.exists() else set()
   fd=os.open(part,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,size)
   def block(i):
    if i in done:return
    if not hasattr(_local,'session'):_local.session=requests.Session()
    start=i*chunk;end=min(size,start+chunk)-1
    for attempt in range(6):
     try:
      r=_local.session.get(url+('&' if '?' in url else '?')+'part='+str(i),headers={'Range':f'bytes={start}-{end}'},timeout=(20,90));r.raise_for_status()
      assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}',(r.status_code,r.headers.get('Content-Range'))
      b=r.content;assert len(b)==end-start+1
      off=0
      while off<len(b):off+=os.pwrite(fd,b[off:],start+off)
      with lock:
       with journal.open('a') as f:f.write(str(i)+'\n');f.flush()
       done.add(i)
      return
     except Exception as e:
      if attempt==5:raise
      time.sleep(attempt+1)
   print('range_start',name,fn,size,url.split('/')[2],flush=True)
   try:
    with cf.ThreadPoolExecutor(max_workers=64) as ex:list(ex.map(block,range(math.ceil(size/chunk))))
    os.fsync(fd)
   finally:os.close(fd)
   assert digest(part)==expected,('Hash mismatch',fn);part.replace(dest)
  else:
   r=requests.get(hf,timeout=90);r.raise_for_status();dest.write_bytes(r.content)
   if expected:assert digest(dest)==expected
  print('downloaded',name,fn,dest.stat().st_size,flush=True)
 # One large file at a time, 64 concurrent ranges; tiny metadata parallel.
 small=[x for x in fs if x.get('size',0)<=32*1024**2];large=[x for x in fs if x not in small]
 with cf.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(one,small))
 for info in large:one(info)
 (target/'DOWNLOAD_COMPLETE').write_text(rev);return target
def download(repo,sub,name):
 with (M/(name+'.download.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _download(repo,sub,name)
if __name__=='__main__':
 specs=json.loads((Path(__file__).parent/'models.json').read_text())
 for arm in sys.argv[1:]:
  x=specs[arm];download(x['repo'],x['subfolder'],x['directory'])
