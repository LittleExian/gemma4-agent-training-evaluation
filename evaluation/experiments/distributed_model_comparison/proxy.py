import json,time,uuid,os
from pathlib import Path
from aiohttp import web,ClientSession,ClientTimeout
P=Path(os.environ['CODEAGENT_ROOT']);Q=P/'research/distributed_model_comparison'
async def handle(req):
 arm=req.match_info['arm'];job=req.match_info['job']
 if not all(c.isalnum() or c in '_-' for c in arm+job):raise web.HTTPBadRequest()
 routes=json.loads((Q/'routes.json').read_text());route=routes[arm]
 body=await req.json();incoming=dict(body)
 for key in ['reasoning','reasoning_effort','chat_template_kwargs']:body.pop(key,None)
 body.update(model=route['model'],thinking_token_budget=6000,chat_template_kwargs={'enable_thinking':True})
 folder=P/'runs/distributed_model_comparison/raw'/arm/job;folder.mkdir(parents=True,exist_ok=True)
 ident=str(time.time_ns())+'_'+uuid.uuid4().hex[:8];path=folder/(ident+'.json')
 row={'started':time.time(),'incoming':incoming,'outgoing':body,'upstream_port':route['port']};path.write_text(json.dumps(row))
 try:
  async with req.app['client'].post(f"http://127.0.0.1:{route['port']}/v1/chat/completions",json=body) as r:
   raw=await r.read();status=r.status
  (folder/(ident+'.response')).write_bytes(raw);d=json.loads(raw)
  row.update(status=status,response=d,finished=time.time());path.write_text(json.dumps(row))
  for c in d.get('choices',[]):
   m=c.get('message',{})
   if m.get('reasoning'):m['reasoning_content']=m['reasoning']
  return web.json_response(d,status=status)
 except Exception as e:
  row.update(error=repr(e),finished=time.time());path.write_text(json.dumps(row));return web.json_response({'error':{'message':repr(e)}},status=502)
async def ctx(app):
 async with ClientSession(timeout=ClientTimeout(total=300)) as s:app['client']=s;yield
app=web.Application(client_max_size=16*1024*1024);app.cleanup_ctx.append(ctx);app.router.add_post('/{arm}/{job}/v1/chat/completions',handle)
web.run_app(app,host='127.0.0.1',port=8013)
