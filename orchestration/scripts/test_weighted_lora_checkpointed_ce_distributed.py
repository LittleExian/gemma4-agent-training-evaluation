"""Real Gloo DDP accumulation against an independent centralized objective."""
import contextlib
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from weighted_lora_checkpointed_ce import weighted_sum


def batches(world):
    generator = torch.Generator().manual_seed(739)
    result = []
    for rank in range(world):
        rank_batches = []
        for micro, length in enumerate((129 + rank * 17, 257 + rank * 11)):
            inputs = torch.randn(length, 4, generator=generator)
            labels = torch.randint(23, (length,), generator=generator)
            weights = torch.tensor([0.0, 0.25, 1.0])[
                (torch.arange(length) + rank + micro) % 3
            ]
            rank_batches.append((inputs, labels, weights))
        result.append(rank_batches)
    return result


def main():
    torch.set_num_threads(1)
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    if world > 1:
        dist.init_process_group('gloo')
    try:
        torch.manual_seed(41)
        model = torch.nn.Linear(4, 23, bias=False)
        reference = torch.nn.Linear(4, 23, bias=False)
        reference.load_state_dict(model.state_dict())
        wrapped = DDP(model, find_unused_parameters=False) if world > 1 else model
        data = batches(world)
        global_weight = sum(weights.sum() for _, _, weights in data[rank])
        if world > 1:
            dist.all_reduce(global_weight)
        numerator = torch.tensor(0.0)
        for micro, (inputs, labels, weights) in enumerate(data[rank]):
            context = (wrapped.no_sync() if world > 1 and micro == 0
                       else contextlib.nullcontext())
            with context:
                loss = weighted_sum(wrapped(inputs).unsqueeze(0), labels, weights)
                (loss * world / global_weight).backward()
                numerator += loss.detach()
        if world > 1:
            dist.all_reduce(numerator)
        # Reference deliberately uses no production chunk/checkpoint helper.
        flat = [row for rank_batches in data for row in rank_batches]
        inputs = torch.cat([row[0] for row in flat])
        labels = torch.cat([row[1] for row in flat])
        weights = torch.cat([row[2] for row in flat])
        expected = (F.cross_entropy(reference(inputs), labels,
                                    reduction='none') * weights).sum() / weights.sum()
        expected.backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad,
                                   rtol=2e-6, atol=1e-6)
        torch.testing.assert_close(numerator / global_weight, expected.detach(),
                                   rtol=2e-6, atol=1e-6)
        error = (model.weight.grad - reference.weight.grad).abs().max().item()
        if rank == 0:
            print(json.dumps({'passed': True, 'world_size': world,
                              'microbatches_per_rank': 2,
                              'unequal_lengths': True, 'no_sync_first': world > 1,
                              'gradient_max_abs_error': error,
                              'global_weight_sum': global_weight.item()}), flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
