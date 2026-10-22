"""Build native-template, last-assistant-only weighted examples. No truncation."""
import argparse,copy,hashlib,json,pathlib,math
from transformers import AutoTokenizer
from curate import request_kind

from serialization import (normalize, render_example, deployment_prompt, ids_hash,
                           SERIALIZATION_VERSION, validate_serialization)

def encode(t,row,max_length=24576,reasoning_weight=.25):
 if not math.isfinite(reasoning_weight) or reasoning_weight<0:raise ValueError("Invalid reasoning weight")
 rendered=render_example(t,row)
 # Keep server-consumed tokens intact: do not merge BPE across this boundary.
 prompt_ids=t(rendered['prompt'],add_special_tokens=False)['input_ids']
 tail=t(rendered['completion'],add_special_tokens=False,return_offsets_mapping=True)
 ids=prompt_ids+tail['input_ids']
 if len(ids)>max_length:return None
 weights=[0.]*len(prompt_ids);reasoning_tokens=0;action_tokens=0
 for s,e in tail['offset_mapping']:
  is_reason=rendered['reason_start']>=0 and s>=rendered['reason_start'] and e<=rendered['reason_end'] and e>s
  weights.append(reasoning_weight if is_reason else 1.)
  reasoning_tokens+=int(is_reason);action_tokens+=int(not is_reason)
 if not action_tokens:raise ValueError('No supervised action/boundary tokens')
 if rendered['reason_start']>=0 and not reasoning_tokens:raise ValueError('Reasoning has no tokens')
 sample=dict(sample_id=row['sample_id'],task_id=row['task_id'],input_ids=ids,
  weights=weights,prompt_tokens=len(prompt_ids),reasoning_tokens=reasoning_tokens,
  action_tokens=action_tokens,reasoning_weight=reasoning_weight,
  serialization_version=SERIALIZATION_VERSION,prompt_ids_sha256=ids_hash(prompt_ids),
  serialization_audit={k:rendered[k] for k in ['inserted_empty_thought','native_prefix_matched']},
  training_metadata=row.get('training_metadata',{}))
 validate_serialization(sample)
 return sample

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--model',default='google/gemma-4-31B-it-qat-w4a16-ct');ap.add_argument('--revision',required=True);ap.add_argument('--max-length',type=int,default=24576);ap.add_argument('--reasoning-weight',type=float,default=.25);ap.add_argument('--include-compaction',action='store_true',help='Legacy reproduction only: include standalone summary targets');args=ap.parse_args()
 t=AutoTokenizer.from_pretrained(args.model,revision=args.revision);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);stats={}
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
  lengths.sort();stats[split]={'samples':n,'tokens':sum(lengths),'min':min(lengths) if lengths else None,'p50':lengths[len(lengths)//2] if lengths else None,'p95':lengths[int(.95*len(lengths))] if lengths else None,'max':max(lengths) if lengths else None,'excluded':excluded}
 stats['serialization_version']=SERIALIZATION_VERSION;stats['include_compaction']=args.include_compaction;stats['reasoning_weight']=args.reasoning_weight;stats['model']=args.model;stats['revision']=args.revision;stats['template_sha256']=hashlib.sha256(t.chat_template.encode()).hexdigest();(out/'preparation.json').write_text(json.dumps(stats,indent=2));t.save_pretrained(out/'tokenizer');print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
