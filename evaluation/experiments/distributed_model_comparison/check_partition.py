import json
from pathlib import Path
ids=json.loads((Path(__file__).parent/'full129.json').read_text())['tasks']
assert len(ids)==len(set(ids))==129
for n in [3,4]:
 shards=[ids[i::n] for i in range(n)];flat=sum(shards,[]);assert len(flat)==len(set(flat))==129 and set(flat)==set(ids)
 print(n,'servers:',list(map(len,shards)),'tasks; exhaustive and disjoint')
