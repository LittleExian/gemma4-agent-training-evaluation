"""Run directly for local tests, or torchrun --standalone --nproc_per_node=N."""
import json,os,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train import weighted_sum

def main():
 world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));torch.manual_seed(1)
 if world>1:dist.init_process_group('gloo')
 model=torch.nn.Linear(4,7,bias=False);reference=torch.nn.Linear(4,7,bias=False);reference.load_state_dict(model.state_dict());wrapped=DDP(model) if world>1 else model
 xs=[torch.randn(3,4)+i for i in range(world)];ys=[torch.tensor([1,2,3]) for _ in range(world)];ws=[torch.tensor([0.,.25,1.+i]) for i in range(world)]
 total=sum(w.sum() for w in ws)
 loss=weighted_sum(wrapped(xs[rank]).unsqueeze(0),ys[rank],ws[rank])*world/total;loss.backward()
 expected=sum(weighted_sum(reference(x).unsqueeze(0),y,w) for x,y,w in zip(xs,ys,ws))/total;expected.backward()
 error=(model.weight.grad-reference.weight.grad).abs().max().item();assert error<1e-6,error
 z=torch.randn(1,3,7,requires_grad=True);w=torch.tensor([0.,.25,1.]);l=weighted_sum(z,ys[0],w)/w.sum();l.backward();assert z.grad[0,0].abs().max()==0
 # Verify weighted objective against an explicit reference.
 z=torch.randn(1,3,7);ce=torch.nn.functional.cross_entropy(z[0],ys[0],reduction='none');assert torch.allclose(weighted_sum(z,ys[0],w),(ce*w).sum())
 if rank==0:print(json.dumps({'passed':True,'world_size':world,'gradient_max_error':error,'prompt_gradient_zero':True}))
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
