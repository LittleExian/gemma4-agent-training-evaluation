"""Exact loss/gradient and bounded intermediate allocation regression checks."""
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from weighted_lora_checkpointed_ce import weighted_sum as reference
from weighted_lora_streamed_ce import weighted_sum


class StreamedCrossEntropyTests(unittest.TestCase):
    def test_matches_original_chunk_order_and_gradients(self):
        torch.set_num_threads(1)
        for dtype in (torch.float32, torch.bfloat16):
            for length in (1, 127, 128, 129, 257):
                for scale in (1., 80.):
                    with self.subTest(dtype=dtype, length=length, scale=scale):
                        torch.manual_seed(length)
                        logits = (torch.randn(1, length, 31) * scale).to(dtype)
                        actual_input = logits.clone().requires_grad_()
                        reference_input = logits.clone().requires_grad_()
                        labels = torch.arange(length) % 31
                        weights = torch.tensor([0., .25, 1.])[torch.arange(length) % 3]
                        actual = weighted_sum(actual_input, labels, weights)
                        expected = reference(reference_input, labels, weights)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        actual.backward(torch.tensor(-2.375))
                        expected.backward(torch.tensor(-2.375))
                        torch.testing.assert_close(actual_input.grad, reference_input.grad,
                                                   rtol=0, atol=0)

    def test_only_input_storage_saved_and_one_full_gradient_allocation(self):
        logits = torch.randn(1, 257, 211, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.arange(257) % 211
        weights = torch.ones(257)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
            loss = weighted_sum(logits, labels, weights)
        self.assertEqual([x.data_ptr() for x in saved],
                         [x.data_ptr() for x in (logits, labels, weights)])
        allocations = []
        original = torch.zeros_like

        def record(value, *args, **kwargs):
            allocations.append(tuple(value.shape))
            return original(value, *args, **kwargs)

        with patch('weighted_lora_streamed_ce.torch.zeros_like', side_effect=record):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        profile_memory=True, record_shapes=True) as prof:
                loss.backward()
        self.assertEqual(allocations, [tuple(logits.shape)])
        events = prof.events()
        full_bytes = logits.numel() * logits.element_size()
        self.assertEqual(sum(event.self_cpu_memory_usage >= full_bytes for event in events), 1)
        self.assertFalse(any(event.name == 'aten::slice_backward' for event in events))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_checkpoint_and_autocast_compatibility(self):
        for reentrant in (False, True):
            logits = torch.randn(1, 129, 31, dtype=torch.bfloat16, requires_grad=True)
            baseline = logits.detach().clone().requires_grad_()
            labels = torch.arange(129) % 31
            weights = torch.linspace(0, 1, 129)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = checkpoint(weighted_sum, logits, labels, weights,
                                    use_reentrant=reentrant)
                expected = reference(baseline, labels, weights)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(logits.grad, baseline.grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
