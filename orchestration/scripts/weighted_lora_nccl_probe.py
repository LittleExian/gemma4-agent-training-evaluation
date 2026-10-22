"""Bounded three-GPU communication gate for the weighted LoRA launch."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist


def main():
    local, rank, world = (int(os.environ[n]) for n in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'))
    assert world == 3
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    try:
        start = time.monotonic()
        # A 64 MiB tensor exercises the large-message transport as well.
        values = torch.full((16 * 1024 * 1024,), rank + 1., device=f'cuda:{local}')
        dist.all_reduce(values)
        assert torch.all(values == 6).item()
        dist.broadcast(values, src=0)
        assert torch.all(values == 6).item()
        dist.barrier(device_ids=[local])
        torch.cuda.synchronize()
        if rank == 0:
            result = {'passed': True, 'world_size': world, 'backend': 'nccl',
                      'bytes_per_rank': values.numel() * values.element_size(),
                      'all_reduce_sum': 6, 'broadcast': True,
                      'seconds': time.monotonic() - start,
                      'visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
            Path(sys.argv[1]).write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
