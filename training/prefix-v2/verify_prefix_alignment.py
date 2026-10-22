"""Audit an entire prepared dataset against independently rendered prompts."""
import argparse,collections,copy,hashlib,json,time
from pathlib import Path
from transformers import AutoTokenizer
from serialization import normalize,validate_serialization,SERIALIZATION_VERSION,render_example
from curate import request_kind

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--tokens',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    t=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);report={};started=time.time()
    for split in ['train','validation']:
        source={r['sample_id']:r for r in map(json.loads,open(Path(a.data)/(split+'.jsonl')))}
        seen=set();counts=collections.Counter();errors=[]
        for row in map(json.loads,open(Path(a.tokens)/(split+'.jsonl'))):
            sid=row['sample_id']
            if sid in seen:raise ValueError('Duplicate prepared sample '+sid)
            seen.add(sid);raw=source[sid];validate_serialization(row)
            prompt=t.apply_chat_template(normalize(raw['messages']),tools=raw['tools'],
                tokenize=False,enable_thinking=True,preserve_thinking=False,add_generation_prompt=True)
            expected=t(prompt,add_special_tokens=False)['input_ids'];n=len(expected)
            if row['prompt_tokens']!=n or row['input_ids'][:n]!=expected or any(row['weights'][:n]):
                errors.append(sid)
            rendered=render_example(t,raw)
            assert t.decode(row['input_ids'][n:])==rendered['completion'],sid
            counts['samples']+=1
            counts['inserted_empty_thought']+=rendered['inserted_empty_thought']
            counts['native_prefix_mismatch_repaired']+=not rendered['native_prefix_matched']
            counts['submit_targets_retained']+=any(c['function']['name']=='submit_patch' for c in raw['target'].get('tool_calls',[]))
            assert request_kind(raw)!='compaction',sid
        missing=set(source)-seen
        # Every omission must be accounted for by prepare.py, never silently lost.
        prep=json.loads((Path(a.tokens)/'preparation.json').read_text())[split]
        assert missing=={x['sample_id'] for x in prep['excluded']},missing
        assert not errors,errors
        report[split]=dict(counts,prefix_errors=errors,excluded=len(missing))
    report.update(passed=True,serialization_version=SERIALIZATION_VERSION,seconds=time.time()-started)
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
