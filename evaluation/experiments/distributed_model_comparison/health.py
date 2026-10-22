import asyncio,json,time,os,sys
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer
P=Path(os.environ['CODEAGENT_ROOT']);Q=P/'research/distributed_model_comparison';arm=sys.argv[1];spec=json.loads((Path(__file__).parent/'models.json').read_text())[arm]
async def main():
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
  for i in range(240):
   try:
    async with s.get('http://127.0.0.1:8002/v1/models') as r:
     if r.status==200:break
   except:pass
   await asyncio.sleep(5)
  else:raise RuntimeError('Server not ready')
  models=[arm,'qwen35_swegym'] if arm=='qwen35_base' else [arm];rows=[]
  async def req(m,b):
   start=time.time()
   async with s.post('http://127.0.0.1:8002/v1/chat/completions',json={**b,'model':m}) as r:
    d=await r.json();assert r.status==200,d
    row={'model':m,'request':b,'response':d,'seconds':time.time()-start};rows.append(row);return d
  b={'messages':[{'role':'user','content':'Use run_command to execute pwd.'}],'tools':[{'type':'function','function':{'name':'run_command','description':'Run a shell command','parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}],'temperature':0.2,'top_p':0.95,'max_tokens':1024,'thinking_token_budget':6000,'chat_template_kwargs':{'enable_thinking':True}}
  for m in models:
   d=await req(m,b);assert d['choices'][0]['message'].get('tool_calls'),('tool parser failed',m,d)
  tok=AutoTokenizer.from_pretrained(str(P/'models/gemma-4-31b-it-qat-w4a16-ct' if arm=='gemma4_kaggle' else Q/'models'/spec['directory']),local_files_only=True)
  probe={'messages':[{'role':'user','content':'Prove that there are infinitely many primes, with detailed reasoning.'}],'max_tokens':256,'temperature':0,'thinking_token_budget':16,'chat_template_kwargs':{'enable_thinking':True}}
  d=await req(arm,probe);msg=d['choices'][0]['message'];reason=msg.get('reasoning') or msg.get('reasoning_content') or '';n=len(tok.encode(reason,add_special_tokens=False));assert 0<n<=24 and msg.get('content'),('thinking budget not enforced',n,msg)
  if len(models)==2:
   b0={'messages':[{'role':'user','content':'Write a Python binary search.'}],'max_tokens':8,'temperature':0,'logprobs':True,'top_logprobs':5,'chat_template_kwargs':{'enable_thinking':False}}
   a=await req(models[0],b0);b1=await req(models[1],b0);assert a['choices'][0]['logprobs']!=b1['choices'][0]['logprobs'],'LoRA inactive'
  # Pilot has 8 distinct tasks per model, or 16 for the shared 9B pair.
  target=int(sys.argv[2]) if len(sys.argv)>2 else (16 if len(models)==2 else 8)
  prompt={'messages':[{'role':'user','content':('def f(x): return x+1\n'*900)+'\nExplain the function.'}],'max_tokens':128,'temperature':0.2,'top_p':0.95,'thinking_token_budget':6000,'chat_template_kwargs':{'enable_thinking':True}}
  trials=[]
  for c in [x for x in [8,16,24,32] if x<=target]:
   start=time.time();out=await asyncio.gather(*(req(models[i%len(models)],prompt) for i in range(c)));seconds=time.time()-start
   trials.append({'concurrency':c,'seconds':seconds,'tokens_per_second':sum(d['usage']['completion_tokens'] for d in out)/seconds})
  best=max(trials,key=lambda x:x['tokens_per_second']);c=best['concurrency']
  (Q/(arm+'_health.json')).write_text(json.dumps({'passed':True,'concurrency':c,'trials':trials,'rows':rows,'finished':time.time()},indent=2));print('HEALTH_PASS',arm,c,flush=True)

asyncio.run(main())
