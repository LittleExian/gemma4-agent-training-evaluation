import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+');ap.add_argument('--output',required=True);a=ap.parse_args();s=Path(__file__).parent
ids=json.loads((s/'full129.json').read_text())['tasks'];arms=set(json.loads((s/'models.json').read_text()));rows=[];seen=set();workers=set();counts=set()
for file in a.files:
 d=json.loads(Path(file).read_text());assert d['phase']=='complete' and not d['failures'],('incomplete',file)
 w=d['worker_index'];n=d['workers'];assert w not in workers;workers.add(w);counts.add(n)
 for r in d['completed']:
  key=(r['arm'],r['task']);assert r['scope']=='all' and key not in seen and r['task'] in ids[w::n] and r['arm'] in arms,key;seen.add(key);rows.append(r)
assert len(counts)==1;n=next(iter(counts));assert workers==set(range(n));assert seen=={(arm,t) for arm in arms for t in ids}
out={'total':len(rows),'workers':n,'results':{arm:{'n':sum(r['arm']==arm for r in rows),'resolved':sum(r['arm']==arm and r['result']['resolved'] for r in rows),'previously_valid_n':sum(r['arm']==arm and r['environment_previously_valid'] for r in rows),'previously_valid_resolved':sum(r['arm']==arm and r['environment_previously_valid'] and r['result']['resolved'] for r in rows)} for arm in sorted(arms)},'rows':rows}
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out['results'],indent=2))
