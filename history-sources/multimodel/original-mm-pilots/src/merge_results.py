import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args();seen={};manifests=[]
for dirname in a.runs:
 root=Path(dirname);m=json.loads((root/'manifest.json').read_text());manifests.append(m)
 state=json.loads((root/'state.json').read_text());assert not state['failed'] and not state['active'] and state['pending']==0,dirname+' incomplete'
 assert len(state['finished'])==len(m['tasks'])*len(m['arms'])
 for item in state['finished']:
  key=(item['arm'],item['task_id']);assert key not in seen,'Duplicate task/arm';seen[key]=item['result']
ref=manifests[0]
for m in manifests:assert all(m[k]==ref[k] for k in ['cohort','shards','arms','files','tasks_sha256','runtime_profile','slot_capacity'])
assert sorted(m['shard'] for m in manifests)==list(range(ref['shards'])),'Missing or duplicate shard'
for arm in ref['arms']:
 rows=[r for (a,t),r in seen.items() if a==arm];times=[r['wall_seconds'] for r in rows]
 print(json.dumps({'arm':arm,'n':len(rows),'resolved':sum(bool(r['resolved']) for r in rows),'mean_seconds':statistics.mean(times),'median_seconds':statistics.median(times),'sorted_seconds':sorted(times)}))
if len(ref['arms'])==2:
 a,b=ref['arms'];tasks=[t for ar,t in seen if ar==a]
 print(json.dumps({'a':a,'b':b,'b_only_wins':[t for t in tasks if seen[b,t]['resolved'] and not seen[a,t]['resolved']],'a_only_wins':[t for t in tasks if seen[a,t]['resolved'] and not seen[b,t]['resolved']]}))
