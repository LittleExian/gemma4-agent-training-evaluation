"""CPU regression tests using the pinned native tokenizer, no model weights."""
import copy,json,os
from pathlib import Path
from transformers import AutoTokenizer
from prepare import encode
from serialization import deployment_prompt,render_example,validate_serialization,THOUGHT_OPEN
lock=json.loads(Path(__file__).with_name('revisions.json').read_text())['model']
t=AutoTokenizer.from_pretrained(os.environ.get('TOKENIZER',lock['repo_id']),revision=lock['revision'])
tools=[{'type':'function','function':{'name':'run_command','parameters':{'type':'object','properties':{'command':{'type':'string'}}}}}]
call={'id':'a','type':'function','function':{'name':'run_command','arguments':'{"command":"pwd"}'}}
base=[{'role':'system','content':'Repair the repository.'},{'role':'user','content':'Fix it.'}]
histories=[base,base+[{'role':'assistant','content':'','tool_calls':[call]},{'role':'tool','tool_call_id':'a','content':'/workspace'}],base+[{'role':'assistant','content':'I will inspect it.','tool_calls':[call]},{'role':'tool','tool_call_id':'a','content':'/workspace'}]]
targets=[{'role':'assistant','content':'Done.'},{'role':'assistant','content':'','tool_calls':[dict(call,id='b')]},{'role':'assistant','content':'','reasoning_content':'Inspect first.','tool_calls':[dict(call,id='b')]},{'role':'assistant','content':'Done.','reasoning':'Preferred reasoning','reasoning_content':'Ignored alternative'}]
checks=0
for messages in histories:
 for target in targets:
  row=dict(sample_id='unit/0',task_id='unit',messages=messages,target=target,tools=tools)
  before=copy.deepcopy(row)
  for weight in [0.,.25,1.]:
   x=encode(t,row,reasoning_weight=weight);r=render_example(t,row)
   prompt=t(deployment_prompt(t,messages,tools),add_special_tokens=False)['input_ids']
   assert x['input_ids'][:x['prompt_tokens']]==prompt
   assert not any(x['weights'][:len(prompt)])
   assert t.decode(x['input_ids'][len(prompt):])==r['completion']
   assert x['action_tokens']+x['reasoning_tokens']+len(prompt)==len(x['input_ids'])
   if r['inserted_empty_thought']:assert r['completion'].startswith('<channel|>')
   if r['prompt'].endswith(THOUGHT_OPEN):assert not r['completion'].startswith(THOUGHT_OPEN)
   validate_serialization(x);checks+=1
  assert row==before
  assert encode(t,row,max_length=1) is None
x=encode(t,row)
for mutate in [lambda q:q.pop('serialization_version'),lambda q:q['weights'].__setitem__(0,1.),lambda q:q['input_ids'].__setitem__(0,42)]:
 bad=copy.deepcopy(x);mutate(bad)
 try:validate_serialization(bad)
 except ValueError:pass
 else:raise AssertionError('Invalid token cache accepted')
print(json.dumps({'passed':True,'cases':checks,'prompt_masked':True,'empty_thought_closed':True,'lookahead_history_preserved':True,'old_cache_rejected':True}))
