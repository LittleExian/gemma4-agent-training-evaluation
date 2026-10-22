import os,json,hashlib,concurrent.futures as cf,time
from pathlib import Path
from urllib.parse import quote
import requests
S=Path(__file__).parent;P=Path(os.environ['CODEAGENT_ROOT']);files=json.loads((S/'data_manifest_all.json').read_text())['files']
def get(x):
 name=x['name'];dest=P/'data/competition_all'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 def valid(p):
  if not p.exists() or p.stat().st_size!=x['bytes']:return False
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
 if valid(dest):return
 token=os.environ['KAGGLE_API_TOKEN'];url='https://www.kaggle.com/api/v1/competitions/data/download/gemma-4-developer-agent/'+quote(name,safe='')
 for attempt in range(5):
  try:
   with requests.get(url,headers={'Authorization':'Bearer '+token},stream=True,timeout=(30,180)) as r:
    r.raise_for_status()
    with dest.with_suffix(dest.suffix+'.part').open('wb') as f:
     for b in r.iter_content(4*1024**2):f.write(b)
   part=dest.with_suffix(dest.suffix+'.part');assert valid(part),'hash mismatch';part.replace(dest);return
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
with cf.ThreadPoolExecutor(max_workers=16) as ex:list(ex.map(get,files))
for name in ['tasks.jsonl','graphs','embeddings']:
 src=P/'data/competition_all'/name;dst=P/'data'/name
 if not dst.exists():dst.symlink_to(src,target_is_directory=src.is_dir())
print('ALL_DATA_VERIFIED',len(files))
