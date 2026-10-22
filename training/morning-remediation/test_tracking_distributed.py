"""CPU collective test: torchrun --standalone --nproc_per_node=2 (or 3/4)."""
import json,os,torch
import torch.distributed as dist
from tracking import gather_rank_stats,progress_metrics
world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
if world>1:dist.init_process_group('gloo')
stats=torch.tensor([100*(rank+1),20,5,15,60+rank,62+rank,2.,100*(rank+1)],dtype=torch.float64)
all_stats=gather_rank_stats(stats,world)
assert len(all_stats)==world
if rank==0:
 m=progress_metrics(1,10,world,1,100,2.,2.,1,[x.tolist() for x in all_stats]);assert m['train/sequence_tokens']==100*world*(world+1)/2
 assert m['train/sequence_tokens_per_second']==50*world*(world+1)/2
 assert m[f'gpu/rank_{world-1}/peak_allocated_gib']==59+world
 print(json.dumps({'passed':True,'world_size':world,'global_sequence_tokens':m['train/sequence_tokens'],'all_rank_memory_metrics':True}))
if world>1:dist.destroy_process_group()
