import unittest
from gemmacodeagent_mm_run import fail_fast_rolling


class DispatchTests(unittest.TestCase):
    def test_infra_failure_stops_new_jobs_and_keeps_inflight(self):
        records = []
        def work(task, endpoint):
            return {'instance_id': task, 'status': 'infra_error' if task == 0 else 'completed'}
        with self.assertRaises(RuntimeError):
            fail_fast_rolling(list(range(20)), ['a', 'b'], work, records.append)
        self.assertEqual({row['instance_id'] for row in records}, {0, 1})

    def test_normal_patch_failures_continue(self):
        records = []
        fail_fast_rolling(list(range(6)), ['a', 'b'],
                          lambda task, endpoint: {'instance_id': task, 'status': 'completed'}, records.append)
        self.assertEqual(len(records), 6)


if __name__ == '__main__': unittest.main()
