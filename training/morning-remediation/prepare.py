"""Build native-template, last-assistant-only weighted examples. No truncation."""
import argparse,copy,hashlib,json,pathlib,math
from transformers import AutoTokenizer
from curate import request_kind

def normalize(messages):
 messages=copy.deepcopy(messages)
 for m in messages:
  m['content']=m.get('content') or ''
  for tc in m.get('tool_calls') or []:
   a=tc['function'].get('arguments')
   if isinstance(a,str):tc['function']['arguments']=json.loads(a)
 return messages

def encode(t,row,max_length=24576,reasoning_weight=.25):
 if not math.isfinite(reasoning_weight) or reasoning_weight<0:raise ValueError("Invalid reasoning weight")
 messages=normalize(row['messages']);target=normalize([row['target']])[0];tools=row['tools']
 marker='TARGET_BOUNDARY_61fe6086'
 dummy={'role':'assistant','reasoning_content':marker,'content':'target'}
 opts=dict(tools=tools,tokenize=False,enable_thinking=True,preserve_thinking=False,add_generation_prompt=False)
 probe=t.apply_chat_template(messages+[dummy],**opts)
 anchor='<|channel>thought\n'+marker+'\n<channel|>'
 assert probe.count(anchor)==1
 prefix=probe.split(anchor)[0]
 full=t.apply_chat_template(messages+[target],**opts)
 assert full.startswith(prefix),'Native template changed history before target'
 start=len(prefix);reason=target.get('reasoning_content') or target.get('reasoning')
 rstart=rend=-1
 if reason:
  block='<|channel>thought\n'+reason+'\n<channel|>'
  assert full[start:].startswith(block)
  rstart=start+len('<|channel>thought\n');rend=rstart+len(reason)
 enc=t(full,add_special_tokens=False,return_offsets_mapping=True)
 if len(enc['input_ids'])>max_length:return None
 weights=[];segments=[]
 for s,e in enc['offset_mapping']:
  if e<=start:weights.append(0.);segments.append('prompt')
  elif rstart>=0 and s>=rstart and e<=rend:weights.append(reasoning_weight);segments.append('reasoning')
  else:weights.append(1.);segments.append('action')
 assert any(w==1 for w in weights)
 if reason:assert 'reasoning' in segments
 return dict(sample_id=row['sample_id'],task_id=row['task_id'],input_ids=enc['input_ids'],weights=weights,prompt_tokens=segments.count('prompt'),reasoning_tokens=segments.count('reasoning'),action_tokens=segments.count('action'),reasoning_weight=reasoning_weight,training_metadata=row.get('training_metadata',{}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--model',default='google/gemma-4-31B-it-qat-w4a16-ct');ap.add_argument('--revision',required=True);ap.add_argument('--max-length',type=int,default=24576);ap.add_argument('--reasoning-weight',type=float,default=.25);ap.add_argument('--include-compaction',action='store_true',help='Legacy reproduction only: include standalone summary targets');args=ap.parse_args()
 t=AutoTokenizer.from_pretrained(args.model,revision=args.revision);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True);stats={}
 for split in ['train','validation']:
  n=0;lengths=[];excluded=[]
  with open(out/(split+'.jsonl'),'w') as f:
   for line in open(pathlib.Path(args.data)/(split+'.jsonl')):
    row=json.loads(line)
    if request_kind(row)=='compaction' and not args.include_compaction:
     excluded.append({'sample_id':row['sample_id'],'error':'compaction target excluded by policy'});continue
    try:sample=encode(t,row,args.max_length,args.reasoning_weight)
    except (ValueError,AssertionError,TypeError,json.JSONDecodeError) as e:excluded.append({'sample_id':row['sample_id'],'error':str(e)});continue
    if sample is None:excluded.append({'sample_id':row['sample_id'],'error':'overlength; not truncated'});continue
    f.write(json.dumps(sample,separators=(',',':'))+'\n');n+=1;lengths.append(len(sample['input_ids']))
  lengths.sort();stats[split]={'samples':n,'tokens':sum(lengths),'min':min(lengths),'p50':lengths[len(lengths)//2],'p95':lengths[int(.95*len(lengths))],'max':max(lengths),'excluded':excluded}
 stats['include_compaction']=args.include_compaction;stats['reasoning_weight']=args.reasoning_weight;stats['model']=args.model;stats['revision']=args.revision;stats['template_sha256']=hashlib.sha256(t.chat_template.encode()).hexdigest();(out/'preparation.json').write_text(json.dumps(stats,indent=2));t.save_pretrained(out/'tokenizer');print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
