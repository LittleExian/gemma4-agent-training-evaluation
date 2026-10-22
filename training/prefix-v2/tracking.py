"""Opt-in W&B scalar tracking. No embedded account, key, dataset text or artifacts."""
import json,os,pathlib,shutil,uuid,warnings
ACTIVE=None
CONFIG_KEYS=('model','revision','steps','accum','lr','seed','smoke','attention',
             'world_size','trainable_parameters','training_samples','training_sequence_tokens')

def add_arguments(parser):
 mode=os.environ.get('WANDB_MODE') or ('online' if os.environ.get('WANDB_PROJECT') else 'disabled')
 parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default=mode)
 parser.add_argument('--wandb-project',default=os.environ.get('WANDB_PROJECT'))
 parser.add_argument('--wandb-entity',default=os.environ.get('WANDB_ENTITY'))
 parser.add_argument('--wandb-name',default=os.environ.get('WANDB_NAME'))
 parser.add_argument('--wandb-id',default=os.environ.get('WANDB_RUN_ID'))
 parser.add_argument('--wandb-new-run',action='store_true',help='Fork monitoring when resuming weights, including another account\'s checkpoint')

class Tracker:
 def __init__(self,args,config,start_step,sdk=None):
  global ACTIVE
  self.run=None;self.state=None;self.path=pathlib.Path(args.output)/'wandb_run.json'
  if args.wandb_mode=='disabled':return
  if not args.wandb_project:raise ValueError('Set WANDB_PROJECT or --wandb-project when enabling W&B')
  if sdk is None:
   import wandb as sdk
  prior=None
  if args.resume and not args.wandb_new_run:
   candidates=[pathlib.Path(args.resume)/'wandb_run.json',self.path]
   for p in candidates:
    if p.exists():prior=json.loads(p.read_text());break
  elif self.path.exists() and not args.wandb_new_run:
   raise ValueError('Output already has a W&B run. Use --resume, a new output directory, or --wandb-new-run')
  resume=None;run_id=args.wandb_id;entity=args.wandb_entity
  if prior and args.wandb_mode=='online':
   if prior['mode']!='online':raise ValueError('Offline runs cannot resume online here; use --wandb-new-run and sync offline logs separately')
   if args.wandb_project!=prior['project'] or (entity and entity!=prior['entity']):raise ValueError('W&B account/project differs from saved run; use --wandb-new-run to fork')
   if run_id and run_id!=prior['id']:raise ValueError('Run ID differs from saved run; use --wandb-new-run')
   run_id=prior['id'];entity=entity or prior['entity'];resume='must'
  elif args.resume and args.wandb_mode=='online' and run_id and not args.wandb_new_run:resume='must'
  elif args.wandb_mode=='online':resume='never'
  if args.wandb_mode=='offline' and args.resume:
   warnings.warn('Offline resume creates a separate monitoring segment; offline SDK resume is unsupported')
   run_id=None
  run_id=run_id or uuid.uuid4().hex[:8]
  local_dir=pathlib.Path(args.output);local_dir.mkdir(parents=True,exist_ok=True)
  safe={k:v for k,v in config.items() if k in CONFIG_KEYS}
  safe.update(reasoning_weight=.25,action_weight=1.,prompt_weight=0.,lora_rank=16,lora_alpha=16,lora_dropout=.05)
  self.run=sdk.init(project=args.wandb_project,entity=entity,name=args.wandb_name,id=run_id,
                    resume=resume,mode=args.wandb_mode,dir=str(local_dir),config=safe,allow_val_change=True,
                    settings=sdk.Settings(disable_git=True,save_code=False,console='off'))
  self.run.define_metric('train/optimizer_step')
  self.run.define_metric('train/*',step_metric='train/optimizer_step')
  self.run.define_metric('gpu/*',step_metric='train/optimizer_step')
  self.run.define_metric('checkpoint/*',step_metric='train/optimizer_step')
  self.run.summary['resumed_from_optimizer_step']=start_step
  self.state={'id':self.run.id,'entity':self.run.entity,'project':self.run.project,'mode':args.wandb_mode,'url':self.run.url if args.wandb_mode=='online' else None}
  self.path.write_text(json.dumps(self.state,indent=2));ACTIVE=self
  print('W&B monitoring: '+str(self.state['url'] or local_dir),flush=True)
 def log(self,metrics):
  if self.run:self.run.log(metrics)
 def checkpoint(self,folder,step,seconds):
  if self.run:
   shutil.copyfile(self.path,pathlib.Path(folder)/'wandb_run.json')
   self.log({'train/optimizer_step':step,'checkpoint/last_saved_step':step,'checkpoint/save_seconds':seconds})
 def finish(self,exit_code=0):
  if self.run:
   self.run.summary['training_exit_code']=exit_code
   self.run.finish(exit_code=exit_code);self.run=None

def finish_active(exit_code=0):
 if ACTIVE:ACTIVE.finish(exit_code)

def progress_metrics(step,total_steps,world,accum,num_samples,step_seconds,elapsed,completed_this_session,rank_stats):
 # rank_stats: [sequence_tokens, completion_tokens, reasoning_tokens, action_tokens,
 #              peak_allocated_gib, peak_reserved_gib, compute_seconds, max_sequence]
 totals=[sum(r[i] for r in rank_stats) for i in range(4)]
 slowest=max(r[6] for r in rank_stats);rate_time=max(step_seconds,1e-9)
 m={'train/optimizer_step':step,'train/total_steps':total_steps,'train/progress_percent':100*step/total_steps,
    'train/epoch_fraction':step*world*accum/num_samples,'train/step_seconds':step_seconds,
    'train/slowest_rank_compute_seconds':slowest,'train/elapsed_seconds':elapsed,
    'train/eta_seconds':max(0,total_steps-step)*elapsed/max(completed_this_session,1),
    'train/sequence_tokens_per_second':totals[0]/rate_time,'train/decisions_per_second':world*accum/rate_time,
    'train/sequence_tokens':totals[0],'train/completion_tokens':totals[1],
    'train/reasoning_tokens':totals[2],'train/action_tokens':totals[3],
    'train/max_sequence_length':max(r[7] for r in rank_stats)}
 for rank,r in enumerate(rank_stats):
  m[f'gpu/rank_{rank}/peak_allocated_gib']=r[4];m[f'gpu/rank_{rank}/peak_reserved_gib']=r[5]
 return m

def gather_rank_stats(local_stats,world):
 import torch
 import torch.distributed as dist
 if world==1:return [local_stats]
 result=[torch.zeros_like(local_stats) for _ in range(world)]
 dist.all_gather(result,local_stats)
 return result
