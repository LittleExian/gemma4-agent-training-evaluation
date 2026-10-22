import importlib.util
import os
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gemmacodeagent_owned_cleanup as cleanup


class Gone(Exception):
    pass


class OwnedCleanupTests(unittest.TestCase):
    def setUp(self):
        self.root = self.process(100, 100.0)
        self.child = self.process(101, 100.01)
        self.foreign = self.process(102, 100.01, 999)
        self.root.children.return_value = [self.child, self.foreign]
        self.psutil = SimpleNamespace(Process=Mock(return_value=self.root),
                                      NoSuchProcess=Gone, boot_time=lambda: 0, wait_procs=Mock(return_value=([], [])))
        self.record = {'pid': 100, 'create_time': 100.0, 'uid': 123, 'launch_boottime': 100.009}
        self.modules = patch.dict(sys.modules, {'psutil': self.psutil})
        self.modules.start()
        self.uid = patch.object(cleanup.os, 'getuid', return_value=123)
        self.uid.start()
        self.tick = patch.object(cleanup.os, 'sysconf', return_value=100)
        self.tick.start()
        self.platform = patch.object(cleanup.sys, 'platform', 'linux')
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.addCleanup(self.modules.stop)
        self.addCleanup(self.uid.stop)
        self.addCleanup(self.tick.stop)

    @staticmethod
    def process(pid, created, uid=123):
        result = Mock(pid=pid)
        result.create_time.return_value = created
        result.uids.return_value = SimpleNamespace(real=uid)
        return result

    def test_one_tick_rounding_allowed_and_foreign_child_untouched(self):
        report = cleanup.release_owned_model(self.record, 100.009)
        self.assertEqual(set(report['terminated']), {100, 101})
        self.foreign.terminate.assert_not_called()
        self.foreign.kill.assert_not_called()

    def test_genuinely_predating_process_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'predating'):
            cleanup.release_owned_model({**self.record, 'launch_boottime': 100.011}, 100.011)
        self.root.terminate.assert_not_called()

    def test_changed_pid_identity_or_uid_denied(self):
        for field, value in [('create_time', 101), ('uid', 999)]:
            with self.subTest(field=field), self.assertRaises((RuntimeError, ValueError)):
                cleanup.release_owned_model({**self.record, field: value}, 99)
        self.root.terminate.assert_not_called()

    def test_current_uid_must_match_record(self):
        self.root.uids.return_value = SimpleNamespace(real=999)
        with self.assertRaises(RuntimeError):
            cleanup.release_owned_model(self.record, 99)
        self.child.terminate.assert_not_called()

    def test_exited_root_harmless(self):
        self.psutil.Process.side_effect = Gone()
        self.assertEqual(cleanup.release_owned_model(self.record, 99), {'terminated': [], 'killed': []})

    def test_child_uid_rechecked_before_terminate(self):
        self.child.uids.side_effect = [SimpleNamespace(real=123), SimpleNamespace(real=999)]
        cleanup.release_owned_model(self.record, 99)
        self.child.terminate.assert_not_called()

    def test_child_pid_reused_before_kill_not_killed(self):
        self.psutil.wait_procs.return_value = ([], [self.child])
        self.child.create_time.side_effect = [100.01, 100.01, 101]
        cleanup.release_owned_model(self.record, 99)
        self.child.terminate.assert_called_once()
        self.child.kill.assert_not_called()

    def test_surviving_owned_identity_killed(self):
        self.psutil.wait_procs.return_value = ([], [self.child])
        report = cleanup.release_owned_model(self.record, 99)
        self.assertEqual(report['killed'], [101])

    def test_missing_record_uid_rejected(self):
        with self.assertRaises((KeyError, ValueError)):
            cleanup.release_owned_model({'pid': 100, 'create_time': 100}, 99)

    def test_root_rechecked_after_child_snapshot(self):
        self.root.create_time.side_effect = [100, 101]
        with self.assertRaises(RuntimeError):
            cleanup.release_owned_model(self.record, 99)
        self.child.terminate.assert_not_called()

    def test_linux_epoch_origin_offset_does_not_widen_ownership_window(self):
        self.psutil.boot_time = lambda: 90.0
        record = {**self.record, 'launch_boottime': 10.009}
        self.assertIn(100, cleanup.release_owned_model(record, 100.059)['terminated'])
        with self.assertRaisesRegex(RuntimeError, 'predating'):
            cleanup.release_owned_model({**record, 'launch_boottime': 10.011}, 100.061)

    def test_linux_missing_launch_boundary_refused(self):
        record = {key: value for key, value in self.record.items() if key != 'launch_boottime'}
        with self.assertRaisesRegex(ValueError, 'CLOCK_BOOTTIME'):
            cleanup.release_owned_model(record, 99)
        self.root.terminate.assert_not_called()

    def test_invalid_kernel_precision_refused(self):
        cleanup.os.sysconf.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'timestamp precision'):
            cleanup.release_owned_model(self.record, 99)
        self.root.terminate.assert_not_called()

    def test_disappearing_child_during_snapshot_ignored(self):
        self.child.create_time.side_effect = Gone()
        report = cleanup.release_owned_model(self.record, 99)
        self.assertEqual(report['terminated'], [100])

    def test_disappearing_root_before_terminate_harmless(self):
        self.root.create_time.side_effect = [100, 100, Gone()]
        report = cleanup.release_owned_model(self.record, 99)
        self.assertEqual(report['terminated'], [101])

    def test_disappearing_child_before_kill_harmless(self):
        self.psutil.wait_procs.return_value = ([], [self.child])
        self.child.create_time.side_effect = [100.01, 100.01, Gone()]
        report = cleanup.release_owned_model(self.record, 99)
        self.assertEqual(report['killed'], [])

    def test_cleanup_error_preserves_primary(self):
        primary = RuntimeError('health probe failed')
        def fail():
            raise RuntimeError('cleanup failed')
        report = cleanup.cleanup_secondary(fail, primary)
        self.assertIn('cleanup failed', report['cleanup_error'])
        self.assertIn('cleanup failed', primary.__notes__[0])
        self.assertEqual(str(primary), 'health probe failed')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            cleanup.cleanup_secondary(fail)

    def test_cleanup_success_returned(self):
        self.assertEqual(cleanup.cleanup_secondary(lambda: {'terminated': [1]}), {'terminated': [1]})


@unittest.skipUnless(importlib.util.find_spec('psutil'), 'psutil required for real process integration')
class RealProcessTests(unittest.TestCase):
    def test_immediately_created_child_cleanup_and_sibling_survives(self):
        import psutil
        sibling = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        started = time.time()
        launch_boot = (time.clock_gettime(time.CLOCK_BOOTTIME)
                       if sys.platform.startswith('linux') else None)
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        try:
            record = {'pid': child.pid, 'create_time': psutil.Process(child.pid).create_time(),
                      'uid': os.getuid(), 'launch_boottime': launch_boot}
            report = cleanup.release_owned_model(record, started, wait_timeout=2)
            child.wait(timeout=5)
            self.assertIn(child.pid, report['terminated'])
            self.assertIsNone(sibling.poll())
        finally:
            for process in [child, sibling]:
                if process.poll() is None:
                    process.kill()
                process.wait()


if __name__ == '__main__':
    unittest.main()
