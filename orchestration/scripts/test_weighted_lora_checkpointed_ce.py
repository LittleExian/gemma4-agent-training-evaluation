"""Objective, gradient and retained-activation checks for chunked weighted CE."""
import unittest

import torch
import torch.nn.functional as F

from weighted_lora_checkpointed_ce import weighted_sum


class WeightedCrossEntropyTests(unittest.TestCase):
    def test_matches_independent_full_cross_entropy(self):
        for dtype in (torch.float32, torch.bfloat16):
            for length in (1, 127, 128, 129, 255, 257):
                for scale in (1.0, 60.0):
                    with self.subTest(dtype=dtype, length=length, scale=scale):
                        generator = torch.Generator().manual_seed(length)
                        values = torch.randn(1, length, 23, generator=generator) * scale
                        actual_logits = values.to(dtype).requires_grad_()
                        expected_logits = actual_logits.detach().clone().requires_grad_()
                        labels = torch.arange(length) % 23
                        weights = torch.tensor([0.0, 0.25, 1.0])[torch.arange(length) % 3]
                        actual = weighted_sum(actual_logits, labels, weights)
                        expected = (
                            F.cross_entropy(expected_logits[0].float(), labels,
                                            reduction='none') * weights
                        ).sum()
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
                        actual.backward()
                        expected.backward()
                        torch.testing.assert_close(actual_logits.grad, expected_logits.grad,
                                                   rtol=2e-6, atol=2e-6)
                        self.assertEqual(actual_logits.grad[:, weights == 0].abs().max().item(), 0)

    def test_no_fp32_vocabulary_activations_retained(self):
        length, vocabulary = 257, 211
        logits = torch.randn(1, length, vocabulary, dtype=torch.bfloat16,
                             requires_grad=True)
        labels = torch.arange(length) % vocabulary
        weights = torch.ones(length, dtype=torch.float32)
        retained = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            retained[(tensor.dtype, storage.data_ptr())] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = weighted_sum(logits, labels, weights)
        fp32_bytes = sum(size for (dtype, _), size in retained.items()
                         if dtype == torch.float32)
        # Only labels/weights and checkpoint sentinels may be retained in FP32;
        # the original implementation retains N*V log-softmax values.
        self.assertLessEqual(fp32_bytes, length * 4)
        self.assertGreaterEqual(sum(size for (dtype, _), size in retained.items()
                                    if dtype == torch.bfloat16), logits.numel() * 2)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
