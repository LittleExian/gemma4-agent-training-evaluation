import os,json,pathlib,tempfile,shutil,torch
import torch.distributed as dist
from checkpointing import save_checkpoint,load_optimizer,restore_rng
from base_loading import base_identity,check_resume
world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
if world>1:dist.init_process_group('gloo')
paths=[tempfile.mkdtemp(prefix='lora-checkpoint-') if rank==0 else None]
if world>1:dist.broadcast_object_list(paths,src=0)
out=pathlib.Path(paths[0]);device=torch.device('cpu')
class Tiny(torch.nn.Linear):
 def save_pretrained(self,path):
  pathlib.Path(path).mkdir(exist_ok=True);torch.save(self.state_dict(),pathlib.Path(path)/'model.pt')
torch.manual_seed(40+rank);model=Tiny(3,2);opt=torch.optim.AdamW(model.parameters());model(torch.ones(1,3)).sum().backward();opt.step();save_checkpoint(model,opt,out,7,rank,world,device,base_identity());check_resume(out/"checkpoint-7")
expected=torch.rand(5);torch.manual_seed(9);opt2=torch.optim.AdamW(model.parameters());step=load_optimizer(opt2,out/'checkpoint-7',world);restore_rng(out/'checkpoint-7',rank,world,device);assert torch.equal(expected,torch.rand(5));assert step==7
for state in opt2.state.values():assert state['step']==1 and torch.isfinite(state['exp_avg']).all()
try:load_optimizer(opt2,out/'checkpoint-7',world+1)
except ValueError:pass
else:raise AssertionError('World size mismatch accepted')
if world>1:dist.barrier()
if rank==0:
 print(json.dumps({'passed':True,'world_size':world,'per_rank_rng_roundtrip':True,'optimizer_roundtrip':True,'world_size_mismatch_rejected':True,'completion_marker':True,'base_identity_roundtrip':True}));shutil.rmtree(out)
if world>1:dist.destroy_process_group()
