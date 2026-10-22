"""Integration check: raw input summaries never enter default tokenized training."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
from curate import read_rows,request_kind
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--tokenizer',required=True);a=p.parse_args()
agent=summary=None
for r in read_rows(Path(a.data)/'train.jsonl'):
 if request_kind(r)=='compaction' and summary is None:summary=r
 if request_kind(r)=='agent' and agent is None:agent=r
 if summary and agent:break
validation=next(r for r in read_rows(Path(a.data)/'validation.jsonl') if request_kind(r)=='agent')
assert summary and agent
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);data=root/'data';data.mkdir()
 (data/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in [agent,summary])+'\n')
 (data/'validation.jsonl').write_text(json.dumps(validation)+'\n')
 for include in [False,True]:
  out=root/str(include)
  cmd=[sys.executable,str(Path(__file__).with_name('prepare.py')),'--data',str(data),'--output',str(out),'--model',a.tokenizer,'--revision','52f3f65bc7a02d555763bc923bd1d9094898219d']
  if include:cmd.append('--include-compaction')
  subprocess.run(cmd,check=True,capture_output=True,text=True)
  rows=list(read_rows(out/'train.jsonl'))
  assert len(rows)==(2 if include else 1)
  assert any(r['sample_id']==summary['sample_id'] for r in rows)==include
print(json.dumps({'passed':True,'default_excludes_summary':True,'explicit_legacy_opt_in':True}))
