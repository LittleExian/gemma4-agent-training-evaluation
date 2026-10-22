"""Original 128-token weighted CE with recomputation of FP32 activations."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_sum(logits, labels, weights):
    losses = F.cross_entropy(logits.float(), labels, reduction='none')
    return (losses * weights).sum()


def weighted_sum(logits, labels, weights):
    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, labels.numel(), 128):
        total = total + checkpoint(
            _chunk_sum,
            logits[0, start:start + 128],
            labels[start:start + 128],
            weights[start:start + 128],
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return total
