import json,pathlib,os
from transformers import AutoTokenizer
from prepare import encode
lock=json.loads(pathlib.Path(__file__).with_name('revisions.json').read_text())['model'];t=AutoTokenizer.from_pretrained(os.environ.get('TOKENIZER',lock['repo_id']),revision=lock['revision'])
tools=[{'type':'function','function':{'name':'run_command','description':'Run a command','parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}]
messages=[{'role':'system','content':'Repair the repository.'},{'role':'user','content':'Fix the parser.'},{'role':'assistant','content':'','tool_calls':[{'id':'a','type':'function','function':{'name':'run_command','arguments':'{"command":"pwd"}'}}]},{'role':'tool','tool_call_id':'a','content':'/workspace'}]
row={'sample_id':'unit/0','task_id':'unit','messages':messages,'tools':tools,'target':{'role':'assistant','reasoning_content':'Inspect the parser first.','content':'','tool_calls':[{'id':'b','type':'function','function':{'name':'run_command','arguments':'{"command":"cat parser.py"}'}}]}}
x=encode(t,row);assert x['reasoning_tokens']>0 and x['action_tokens']>0 and x['prompt_tokens']>0
supervised=t.decode([i for i,w in zip(x['input_ids'],x['weights']) if w>0]);assert 'Fix the parser.' not in supervised and 'Inspect the parser first.' in supervised and 'cat parser.py' in supervised
assert encode(t,row,max_length=10) is None
row['target']={'role':'assistant','content':'Done.'};y=encode(t,row);assert y['reasoning_tokens']==0 and y['action_tokens']>0
row['target']={'role':'assistant','content':'','tool_calls':[{'id':'x','type':'function','function':{'name':'run_command','arguments':'not valid JSON'}}]}
try:encode(t,row)
except json.JSONDecodeError:pass
else:raise AssertionError('Malformed tool arguments must not silently train')
print(json.dumps({'passed':True,'native_template':True,'prompt_masked':True,'reasoning_weight':.25,'action_weight':1,'overlength_rejected':True,'malformed_arguments_rejected':True}))
