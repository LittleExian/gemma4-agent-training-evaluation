"""Report supervision concentration; token mass is not a gradient measurement."""
import argparse,collections,json,math
from pathlib import Path
from curate import read_rows,request_kind,digest

def summarize(data,tokens):
    raw={r['sample_id']:r for r in read_rows(data)}
    totals=collections.Counter();tasks=collections.defaultdict(collections.Counter)
    kinds=collections.defaultdict(collections.Counter);seen=set()
    for row in read_rows(tokens):
        sid=row['sample_id']
        if sid in seen:raise ValueError('Duplicate tokenized sample')
        seen.add(sid);original=raw[sid]
        if original['task_id']!=row['task_id']:raise ValueError('Task identity mismatch')
        values={'rows':1,'weighted_tokens':sum(row['weights'][1:]),
                'reasoning_tokens':row['reasoning_tokens'],'action_tokens':row['action_tokens']}
        totals.update(values);tasks[row['task_id']].update(values)
        kinds[request_kind(original)].update(values)
    n=max(1,math.ceil(len(tasks)/10))
    masses=sorted([t['weighted_tokens'] for t in tasks.values()],reverse=True)
    return {'data_sha256':digest(data),'tokenized_sha256':digest(tokens),'totals':dict(totals),
            'tasks':len(tasks),'top_decile_task_count':n,
            'top_decile_weighted_token_fraction':sum(masses[:n])/sum(masses) if masses else None,
            'request_kinds':{k:dict(v) for k,v in kinds.items()},
            'raw_rows_without_tokenized_example':len(set(raw)-seen)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--tokens',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=summarize(a.data,a.tokens);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
