"""Weighted reasoning LoRA; BF16 base; DDP on 1-4 96GB cards."""
import argparse,contextlib,json,os,pathlib,random,time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from peft import LoraConfig,get_peft_model,PeftModel
from checkpointing import save_checkpoint,load_optimizer,restore_rng
from validate_w4 import audit_decoded,audit_adapter
from base_loading import MODEL_ID,check_source,check_resume,load_base,base_identity,assert_lora_only
from provenance import build_manifest, check_resume_manifest, save_manifest
from serialization import validate_serialization
from objectives import task_scales, validate_rows, scalar_loss
from tracking import add_arguments,Tracker,progress_metrics,finish_active,gather_rank_stats

from weighted_lora_streamed_ce import weighted_sum

def forward_sum(model,row,device):
 ids=torch.tensor(row['input_ids'],device=device,dtype=torch.long).unsqueeze(0)
 w=torch.tensor(row['weights'],device=device,dtype=torch.float32)
 positions=torch.nonzero(w[1:]>0).flatten()
 labels=ids[0,positions+1];weights=w[positions+1]
 with torch.autocast("cuda",dtype=torch.bfloat16):
  out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=positions)
 return weighted_sum(out.logits,labels,weights),weights.sum()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--model',default=MODEL_ID);ap.add_argument('--revision',required=True);ap.add_argument('--steps',type=int,default=0);ap.add_argument('--accum',type=int,default=4);ap.add_argument('--lr',type=float,default=2e-5);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--objective',choices=['token_mean','task_mean'],default='token_mean');ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume');ap.add_argument('--save-every',type=int,default=50);ap.add_argument('--attention',default='sdpa');add_arguments(ap);args=ap.parse_args()
 rows=[json.loads(l) for l in open(args.data)];assert rows;validate_rows(rows)
 for row in rows:validate_serialization(row)
 check_source(args.model,args.revision)
 if args.resume:check_resume(args.resume)
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));assert 1<=world<=4
 assert args.accum>0 and args.lr>0
 torch.cuda.set_device(local);device=torch.device('cuda',local)
 assert torch.cuda.get_device_properties(local).total_memory>=90*2**30,'This BF16 recipe requires a 96GB-class GPU'
 if world>1:dist.init_process_group('nccl')
 torch.manual_seed(args.seed);random.seed(args.seed)
 out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)

 if args.steps==0:args.steps=(len(rows)+world*args.accum-1)//(world*args.accum)
 if args.smoke:
  rows=[max(rows,key=lambda x:len(x['input_ids']))];args.steps=1;args.accum=1
 scales=task_scales(rows)
 manifest=build_manifest(args)
 if args.resume:check_resume_manifest(args.resume,manifest)
 if rank==0:save_manifest(out,manifest)
 base=load_base(args.model,args.revision,local,args.attention)
 if rank==0:
  audit=audit_decoded(base);(out/'base_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
 base.config.use_cache=False;base.config.text_config.use_cache=False
 # Nested FFN checkpointing bounds activation memory for long repository histories.
 from torch.utils.checkpoint import checkpoint
 for module in base.model.language_model.modules():
  if module.__class__.__name__=='Gemma4TextMLP':
   original=module.forward
   def chunked(x,fn=original,mod=module):
    if not mod.training or x.shape[1]<=2048:return fn(x)
    return torch.cat([checkpoint(fn,z,use_reentrant=False) for z in x.split(2048,dim=1)],dim=1)
   module.forward=chunked
 targets=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
 if args.resume:model=PeftModel.from_pretrained(base,str(pathlib.Path(args.resume)/'adapter'),is_trainable=True)
 else:model=get_peft_model(base,LoraConfig(revision=args.revision,r=16,lora_alpha=16,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
 assert_lora_only(model)
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 trainable=[p for p in model.parameters() if p.requires_grad];assert trainable
 if rank==0:
  (out/'base_identity.json').write_text(json.dumps(base_identity(),indent=2))
  model.print_trainable_parameters();(out/'config.json').write_text(json.dumps(vars(args)|{'world_size':world,'trainable_parameters':sum(p.numel() for p in trainable),'loss':args.objective,'gpu':torch.cuda.get_device_name()},indent=2))
 optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=.01)
 start=0
 if args.resume:start=load_optimizer(optimizer,args.resume,world)
 tracker=None;tracking_status=[None,False]
 if rank==0:
  try:
   tracker=Tracker(args,vars(args)|{'world_size':world,'trainable_parameters':sum(p.numel() for p in trainable),'training_samples':len(rows),'training_sequence_tokens':sum(len(r['input_ids']) for r in rows)},start)
   tracking_status[1]=tracker.run is not None
  except Exception as exc:tracking_status[0]=str(exc)
 if world>1:dist.broadcast_object_list(tracking_status,src=0)
 if tracking_status[0]:raise RuntimeError('W&B initialization failed: '+tracking_status[0])
 tracking_enabled=tracking_status[1]
 wrapped=DDP(model,device_ids=[local],find_unused_parameters=False) if world>1 else model
 sampler=DistributedSampler(rows,num_replicas=world,rank=rank,shuffle=not args.smoke,seed=args.seed,drop_last=False)
 epoch=0;sampler.set_epoch(epoch);it=iter(DataLoader(rows,batch_size=None,sampler=sampler,num_workers=0))
 def take():
  nonlocal it,epoch
  try:return next(it)
  except StopIteration:
   epoch+=1;sampler.set_epoch(epoch);it=iter(DataLoader(rows,batch_size=None,sampler=sampler,num_workers=0));return next(it)
 for _ in range(start*args.accum):take()
 if args.resume:restore_rng(args.resume,rank,world,device)
 log=open(out/f'metrics.rank{rank}.jsonl','a',buffering=1)
 def save(step):
  saved_at=time.monotonic();save_checkpoint(model,optimizer,out,step,rank,world,device,base_identity())
  if rank==0:
   export=audit_adapter(out/f'checkpoint-{step}'/'adapter');(out/f'checkpoint-{step}'/'adapter_audit.json').write_text(json.dumps(export,indent=2))
  if tracker:tracker.checkpoint(out/f'checkpoint-{step}',step,time.monotonic()-saved_at)
 t0=time.monotonic()
 for step in range(start,args.steps):
  iteration_started=time.monotonic()
  batch=[take() for _ in range(args.accum)]
  global_weight=torch.tensor(sum(sum(x['weights'][1:]) for x in batch) if args.objective=='token_mean' else len(batch),device=device,dtype=torch.float32)
  if world>1:dist.all_reduce(global_weight)
  optimizer.zero_grad(set_to_none=True);model.train();torch.cuda.reset_peak_memory_stats(device);numerator=torch.zeros((),device=device);before=trainable[1].detach().clone();started=time.monotonic()
  for j,row in enumerate(batch):
   sync=wrapped.no_sync() if world>1 and j+1<len(batch) else contextlib.nullcontext()
   with sync:
    n,w=forward_sum(wrapped,row,device);assert torch.isfinite(n)
    objective_n=scalar_loss(n,w,scales[row['task_id']],args.objective)
    (objective_n*world/global_weight).backward();numerator+=objective_n.detach()
  norm=torch.nn.utils.clip_grad_norm_(trainable,1.0);assert torch.isfinite(norm) and norm>0
  optimizer.step();changed=(trainable[1].detach()-before).abs().max().item();assert changed>0,'LoRA B matrix did not update'
  if world>1:dist.all_reduce(numerator)
  metric={'step':step+1,'loss':(numerator/global_weight).item(),'gradient_norm':norm.item(),'adapter_delta_max':changed,'seconds':time.monotonic()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,'lengths':[len(x['input_ids']) for x in batch],'sample_ids':[x['sample_id'] for x in batch]};log.write(json.dumps(metric)+'\n')
  if tracking_enabled:
   local_stats=torch.tensor([sum(len(x['input_ids']) for x in batch),sum(sum(w>0 for w in x['weights']) for x in batch),sum(x.get('reasoning_tokens',x['weights'].count(.25)) if x.get('reasoning_weight',.25)>0 else 0 for x in batch),sum(x.get('action_tokens',x['weights'].count(1.)) for x in batch),metric['peak_allocated_gib'],torch.cuda.max_memory_reserved(device)/2**30,metric['seconds'],max(metric['lengths'])],dtype=torch.float64,device=device)
   stats=gather_rank_stats(local_stats,world)
   if rank==0:
    tracked=progress_metrics(step+1,args.steps,world,args.accum,len(rows),time.monotonic()-iteration_started,time.monotonic()-t0,step+1-start,[r.tolist() for r in stats])
    tracked.update({'train/loss':metric['loss'],'train/gradient_norm':metric['gradient_norm'],'train/adapter_delta_max':changed,'train/learning_rate':optimizer.param_groups[0]['lr'],'train/objective_denominator':global_weight.item()})
    tracker.log(tracked)
  if rank==0:print(json.dumps(metric),flush=True)
  if (step+1)%args.save_every==0 or step+1==args.steps:save(step+1)
 if args.smoke:
  if rank==0:
   frozen=audit_decoded(base);(out/'base_audit_after.json').write_text(json.dumps(frozen,indent=2))
  model.eval();short=min(rows,key=lambda x:len(x['input_ids']));ids=torch.tensor(short['input_ids'][:256],device=device).unsqueeze(0)
  with torch.no_grad():reference=model(input_ids=ids,logits_to_keep=1,use_cache=False).logits.detach()
  model.load_adapter(out/f'checkpoint-{args.steps}'/'adapter',adapter_name='roundtrip',is_trainable=False);model.set_adapter('roundtrip');model.eval()
  with torch.no_grad():restored=model(input_ids=ids,logits_to_keep=1,use_cache=False).logits.detach()
  diff=(reference-restored).abs().max().item();assert diff==0,('reload mismatch',diff)
  result={'passed':True,'steps':args.steps,'world_size':world,'reload_max_abs_logit_difference':diff,'elapsed_seconds':time.monotonic()-t0,'longest_real_sequence':max(len(x['input_ids']) for x in rows),'note':'Optimization correctness only; no downstream quality claim.'};(out/'SMOKE_PASS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 log.close()
 if world>1:dist.destroy_process_group()
if __name__=='__main__':
 try:main()
 except BaseException:
  finish_active(exit_code=1)
  raise
 else:finish_active(exit_code=0)
