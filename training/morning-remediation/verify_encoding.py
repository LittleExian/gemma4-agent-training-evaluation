import sys,json,pathlib,time
import argparse
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--baseline-tokens',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--output',required=True);args=p.parse_args()
from transformers import AutoTokenizer
from prepare import encode
from curate import request_kind
model=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
counts={};started=time.time()
for split in ['train','validation']:
 old={r['sample_id']:r for r in map(json.loads,open(pathlib.Path(args.baseline_tokens)/(split+'.jsonl')))}
 n=0;kinds={}
 for row in map(json.loads,open(pathlib.Path(args.data)/(split+'.jsonl'))):
  new=encode(model,row);previous=old[row['sample_id']]
  assert new['input_ids']==previous['input_ids'],row['sample_id']
  assert new['weights']==previous['weights'],row['sample_id']
  key=request_kind(row)
  if key not in kinds:
   zero=encode(model,row,reasoning_weight=0)
   assert zero['input_ids']==new['input_ids']
   assert sum(w>0 for w in zero['weights'])==new['action_tokens']
   one=encode(model,row,reasoning_weight=1)
   assert one['reasoning_tokens']==new['reasoning_tokens']
   kinds[key]=row['sample_id']
  n+=1
  if n%500==0:print(split,n,flush=True)
 counts[split]=n
result={'passed':True,'legacy_encoding_identical':counts,'reasoning_weight_ablation_preserves_tokens':True,'seconds':time.time()-started}
pathlib.Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
