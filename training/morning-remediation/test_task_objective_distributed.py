"""Compare task-balanced DDP gradients to a direct task-average reference.
Run: torchrun --standalone --nproc_per_node=2 test_task_objective_distributed.py
"""
import os,json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from objectives import task_scales, scalar_loss

def main():
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
    if world>1:dist.init_process_group('gloo')
    torch.manual_seed(31)
    model=torch.nn.Linear(3,4,bias=False);reference=torch.nn.Linear(3,4,bias=False)
    reference.load_state_dict(model.state_dict())
    wrapped=DDP(model) if world>1 else model
    rows=[{'task_id':'a'}]+[{'task_id':'b'}]*(2*world-1)
    scales=task_scales(rows)
    xs=[torch.randn(i+1,3) for i in range(len(rows))]
    ys=[torch.arange(i+1)%4 for i in range(len(rows))]
    weights=[torch.tensor([.25 if j%2 else 1. for j in range(i+1)]) for i in range(len(rows))]
    def decision(m,i):
        ce=torch.nn.functional.cross_entropy(m(xs[i]),ys[i],reduction='none')
        return (ce*weights[i]).sum(),weights[i].sum()
    for i in range(rank,len(rows),world):
        n,w=decision(wrapped,i)
        (scalar_loss(n,w,scales[rows[i]['task_id']],'task_mean')*world/len(rows)).backward()
    by_task={}
    for i,row in enumerate(rows):
        n,w=decision(reference,i);by_task.setdefault(row['task_id'],[]).append(n/w)
    expected=sum(sum(v)/len(v) for v in by_task.values())/len(by_task)
    expected.backward()
    error=(model.weight.grad-reference.weight.grad).abs().max().item()
    assert error<1e-6,error
    if rank==0:print(json.dumps({'passed':True,'world_size':world,'gradient_max_abs_error':error}))
    if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
