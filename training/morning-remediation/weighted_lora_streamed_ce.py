"""Exact 128-token CE recomputation with a single full logits gradient buffer.

Only first-order logits gradients are supported; labels and weights are constants.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _StreamedWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, weights):
        ctx.save_for_backward(logits, labels, weights)
        total = logits.new_zeros((), dtype=torch.float32)
        for start in range(0, labels.numel(), 128):
            losses = F.cross_entropy(logits[0, start:start + 128].float(),
                                     labels[start:start + 128], reduction='none')
            total = total + (losses * weights[start:start + 128]).sum()
        return total

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        logits, labels, weights = ctx.saved_tensors
        gradient = torch.zeros_like(logits)
        for start in range(0, labels.numel(), 128):
            with torch.enable_grad():
                chunk = logits[0, start:start + 128].detach().requires_grad_()
                losses = F.cross_entropy(chunk.float(), labels[start:start + 128],
                                         reduction='none')
                total = (losses * weights[start:start + 128]).sum()
                (chunk_gradient,) = torch.autograd.grad(total, chunk, upstream)
            gradient[0, start:start + 128].copy_(chunk_gradient)
        return gradient, None, None


def weighted_sum(logits, labels, weights):
    return _StreamedWeightedSum.apply(logits, labels, weights)
