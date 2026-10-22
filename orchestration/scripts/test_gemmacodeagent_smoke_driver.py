"""Ownership cleanup tests with fake psutil; never signal real processes."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

import gemmacodeagent_smoke_driver as driver


class Gone(Exception):
    pass


class FakeProcess:
    def __init__(self, pid, created=110, uid=1005, children=()):
        self.pid, self.created, self.uid = pid, created, uid
        self.descendants = list(children)
        self.terminate = Mock()
        self.kill = Mock()
        self.create_hook = None

    def create_time(self):
        if self.create_hook:
            self.create_hook(self)
        return self.created

    def uids(self):
        return types.SimpleNamespace(real=self.uid)

    def children(self, recursive=False):
        assert recursive
        return self.descendants


class CleanupOwnershipTests(unittest.TestCase):
    def invoke(self, root, record=None, wait=None):
        fake = types.SimpleNamespace(Process=Mock(return_value=root), NoSuchProcess=Gone,
                                     wait_procs=Mock(side_effect=wait or (lambda processes, timeout: (processes, []))))
        with patch.dict(sys.modules, {'psutil': fake}), patch.object(driver.os, 'getuid', return_value=1005):
            result = driver.release_owned_model(record or {'pid': root.pid, 'create_time': 110}, started=100)
        return result, fake

    def test_refuses_model_record_predating_this_driver_before_process_lookup(self):
        root = FakeProcess(1)
        fake = types.SimpleNamespace(Process=Mock(), NoSuchProcess=Gone)
        with patch.dict(sys.modules, {'psutil': fake}):
            with self.assertRaisesRegex(RuntimeError, 'predating'):
                driver.release_owned_model({'pid': 1, 'create_time': 99}, started=100)
        fake.Process.assert_not_called()
        root.terminate.assert_not_called()

    def test_refuses_reused_root_pid_and_foreign_uid(self):
        for root in (FakeProcess(1, created=111), FakeProcess(1, uid=2000)):
            with self.subTest(created=root.created, uid=root.uid):
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    self.invoke(root)
                root.terminate.assert_not_called()
                root.kill.assert_not_called()

    def test_root_identity_is_not_rebound_if_pid_changes_during_target_snapshot(self):
        root = FakeProcess(1)
        reads = []

        def change_during_snapshot(process):
            reads.append(True)
            if len(reads) > 1:
                process.created = 999

        root.create_hook = change_during_snapshot
        try:
            self.invoke(root)
        except RuntimeError:
            pass
        root.terminate.assert_not_called()
        root.kill.assert_not_called()

    def test_terminates_only_owned_root_and_owned_descendants(self):
        own, foreign = FakeProcess(2), FakeProcess(3, uid=2000)
        root = FakeProcess(1, children=[own, foreign])
        result, fake = self.invoke(root)
        self.assertEqual(result, {'terminated': [1, 2], 'killed': []})
        root.terminate.assert_called_once()
        own.terminate.assert_called_once()
        foreign.terminate.assert_not_called()
        foreign.kill.assert_not_called()
        self.assertEqual(fake.wait_procs.call_args.args[0], [root, own])

    def test_gone_root_returns_empty_report(self):
        fake = types.SimpleNamespace(Process=Mock(side_effect=Gone), NoSuchProcess=Gone)
        with patch.dict(sys.modules, {'psutil': fake}):
            self.assertEqual(driver.release_owned_model({'pid': 1, 'create_time': 110}, 100), {'terminated': [], 'killed': []})

    def test_kill_after_wait_requires_unchanged_creation_time_and_current_uid(self):
        stable, reused, foreign = FakeProcess(2), FakeProcess(3), FakeProcess(4)
        root = FakeProcess(1, children=[stable, reused, foreign])

        def wait(processes, timeout):
            self.assertEqual(timeout, 15)
            reused.created = 999
            foreign.uid = 2000
            return [], [stable, reused, foreign]

        result, _ = self.invoke(root, wait=wait)
        self.assertEqual(result['killed'], [2])
        stable.kill.assert_called_once()
        reused.kill.assert_not_called()
        foreign.kill.assert_not_called()

    def test_no_signal_if_uid_changes_between_discovery_and_terminate(self):
        child = FakeProcess(2)
        child.create_hook = lambda process: setattr(process, 'uid', 2000)
        root = FakeProcess(1, children=[child])
        result, _ = self.invoke(root)
        child.terminate.assert_not_called()
        self.assertNotIn(child.pid, result['terminated'])


if __name__ == '__main__':
    unittest.main()
