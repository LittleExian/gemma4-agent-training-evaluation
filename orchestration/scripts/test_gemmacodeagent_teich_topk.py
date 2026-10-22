import unittest
from gemmacodeagent_teich_topk import sampling_override


class OverrideTests(unittest.TestCase):
    def test_exact_override(self):
        self.assertEqual(sampling_override(['--generation-config', 'vllm',
            '--override-generation-config', '{"top_k":64}']), {'top_k': 64})

    def test_refuses_missing_ambiguous_or_extra_changes(self):
        for command in ([], ['--generation-config', 'auto'],
            ['--generation-config', 'vllm', '--override-generation-config', '{"top_k":0}'],
            ['--generation-config', 'vllm', '--override-generation-config', '{"top_k":64,"top_p":0.8}'],
            ['--generation-config', 'vllm', '--override-generation-config', '{"top_k":64}',
             '--override-generation-config', '{"top_k":64}']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                sampling_override(command)


if __name__ == '__main__':
    unittest.main()
