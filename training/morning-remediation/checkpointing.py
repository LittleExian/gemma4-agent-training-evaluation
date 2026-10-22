"""Shared-filesystem checkpoints; publish completion marker only after every rank."""
import json,pathlib,torch
import torch.distributed as dist

def barrier(world):
 if world>1:dist.barrier()

def save_checkpoint(model,optimizer,out,step,rank,world,device,base_identity=None):
 p=pathlib.Path(out)/f'checkpoint-{step}'
 if rank==0:
  p.mkdir(parents=True,exist_ok=True)
  if (p/'COMPLETE.json').exists():(p/'COMPLETE.json').unlink()
  model.save_pretrained(p/'adapter')
  if base_identity is not None:(p/'base_identity.json').write_text(json.dumps(base_identity,indent=2))
  torch.save({'step':step,'optimizer':optimizer.state_dict()},p/'training_state.pt')
 barrier(world)
 rng={'world_size':world,'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None}
 torch.save(rng,p/f'rng.rank{rank}.pt');barrier(world)
 if rank==0:(p/'COMPLETE.json').write_text(json.dumps({'step':step,'world_size':world}))
 barrier(world)

def load_optimizer(optimizer,path,world):
 p=pathlib.Path(path);meta=json.loads((p/'COMPLETE.json').read_text())
 if meta['world_size']!=world:raise ValueError('Resume world size must match checkpoint')
 state=torch.load(p/'training_state.pt',map_location='cpu',weights_only=False);optimizer.load_state_dict(state['optimizer']);return state['step']

def restore_rng(path,rank,world,device):
 rng=torch.load(pathlib.Path(path)/f'rng.rank{rank}.pt',map_location='cpu',weights_only=False)
 if rng['world_size']!=world:raise ValueError('Resume world size must match checkpoint')
 torch.set_rng_state(rng['cpu'])
 if device.type=='cuda':torch.cuda.set_rng_state(rng['cuda'],device)
