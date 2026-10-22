"""A complete paired-development gate, not a statistical proof of improvement.
Accepts official resolved booleans; pipeline status SUCCESS is not solve success.
Use an independently locked task list. Never silently drop failures or duplicates.
"""
import argparse,json,math,statistics
from pathlib import Path

def assess(rows,tasks,baseline='base',candidate='lora',max_time_ratio=1.25):
    if not tasks or len(tasks)!=len(set(tasks)):raise ValueError('Invalid locked task list')
    if baseline==candidate:raise ValueError('Arms must differ')
    index={};expected=set(tasks)
    for row in rows:
        if row['arm'] not in [baseline,candidate]:raise ValueError('Unexpected arm')
        key=(row['arm'],row['task_id'])
        if row['task_id'] not in expected or key in index:raise ValueError('Unexpected/duplicate task result')
        if not isinstance(row['resolved'],bool):raise ValueError('resolved must be boolean')
        if row.get('process_exit_code')!=0:raise ValueError('Infrastructure failure: rerun explicitly, preserve failed ledger')
        seconds=row['elapsed_seconds']
        if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid elapsed time')
        index[key]=row
    missing=[(a,t) for a in [baseline,candidate] for t in tasks if (a,t) not in index]
    if missing:raise ValueError(f'Incomplete paired evaluation: {missing}')
    wins=[];losses=[];metrics={}
    for a in [baseline,candidate]:
        arm=[index[a,t] for t in tasks];durations=[r['elapsed_seconds'] for r in arm]
        metrics[a]={'solved':sum(r['resolved'] for r in arm),'count':len(arm),
                    'mean_seconds':statistics.mean(durations),'median_seconds':statistics.median(durations),
                    'session_timeouts':sum('session timeout' in str(r.get('error_message','')).lower() for r in arm)}
    for t in tasks:
        b,c=index[baseline,t]['resolved'],index[candidate,t]['resolved']
        if c and not b:wins.append(t)
        if b and not c:losses.append(t)
    ratio=metrics[candidate]['mean_seconds']/metrics[baseline]['mean_seconds']
    passed=len(wins)>=len(losses) and ratio<=max_time_ratio
    return {'passed':passed,'policy':'no solve-count regression and bounded mean elapsed time on the complete locked development set',
            'not_a_generalization_claim':True,'arms':metrics,'candidate_only':wins,'baseline_only':losses,
            'mean_time_ratio':ratio,'max_time_ratio':max_time_ratio}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True)
    p.add_argument('--tasks',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline',default='base');p.add_argument('--candidate',default='lora')
    p.add_argument('--max-time-ratio',type=float,default=1.25);a=p.parse_args()
    try:
        report=assess([json.loads(l) for l in open(a.results)],json.loads(Path(a.tasks).read_text()),a.baseline,a.candidate,a.max_time_ratio)
    except (ValueError,KeyError,TypeError) as e:
        report={'passed':False,'invalid_evaluation':str(e)}
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 2)
if __name__=='__main__':main()
