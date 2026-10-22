"""Native sandbox escape regressions, using disposable fake host data only."""
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import gemmacodeagent_native_guard as guard_module


class Manager:
    def __init__(self, shard):
        self.shard, self.sandboxes = shard, {}
        self._default_sandbox_id = None
        self.initial_files = None
        self.counter = 0
    def start(self):
        self.counter += 1
        identifier = str(self.counter)
        root = self.shard / 'tmp' / ('swegemma_sandbox_' + identifier)
        paths = {name: root / name for name in ('workspace', 'tmp', 'wheels', 'venv')}
        paths['root'] = root
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        self.sandboxes[identifier] = paths
        self._default_sandbox_id = self._default_sandbox_id or identifier
        return identifier
    def initialize(self):
        self.start()
    def exec(self, identifier, command, **kwargs):
        return 'executed'
    def stop(self, identifier):
        import shutil
        shutil.rmtree(self.sandboxes.pop(identifier)['root'])
    def copy_to(self, source_or_id, destination_or_source, destination=None):
        if destination is None:
            self._copy_to_sandbox(self._default_sandbox_id, source_or_id, destination_or_source)
        else:
            self._copy_to_sandbox(source_or_id, destination_or_source, destination)
    def copy_from(self, source_or_id, destination_or_source, destination=None):
        if destination is None:
            self._copy_from_sandbox(self._default_sandbox_id, source_or_id, destination_or_source)
        else:
            self._copy_from_sandbox(source_or_id, destination_or_source, destination)


class NativeGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.shard = self.root / 'job'
        self.manager = Manager(self.shard)
        guard_module.install_native_guards(self.manager, self.shard)
        self.identifier = self.manager.start()
        self.paths = self.manager.sandboxes[self.identifier]
        self.host = self.root / 'fake-host'
        self.host.mkdir()
        self.sentinel = self.host / 'sentinel'
        self.sentinel.write_text('host-only')
    def tearDown(self):
        for identifier in list(self.manager.sandboxes):
            try:
                self.manager.stop(identifier)
            except (RuntimeError, OSError):
                pass
        self.temporary.cleanup()
    def test_normal_read_write_and_both_copy_signatures(self):
        self.manager.write_file('nested/test.txt', 'inside')
        self.assertEqual(self.manager.read_file('/workspace/nested/test.txt'), 'inside')
        self.manager.copy_to(self.identifier, self.sentinel, '/tmp/')
        self.assertEqual(self.manager.read_file('/tmp/sentinel'), 'host-only')
        self.manager.copy_to(self.sentinel, '/workspace/new/parents/copied')
        self.assertEqual(self.manager.read_file('new/parents/copied'), 'host-only')
        self.manager.copy_to(self.sentinel, '/workspace/copied')
        output = self.root / 'output'
        self.manager.copy_from('/workspace/copied', output)
        self.assertEqual(output.read_text(), 'host-only')
        self.manager.copy_from(self.identifier, '/workspace/nested', self.root / 'directory-output')
        self.assertEqual((self.root / 'directory-output/test.txt').read_text(), 'inside')
        self.manager.copy_to(self.host, '/workspace/folder')
        self.assertEqual(self.manager.read_file('folder/sentinel'), 'host-only')
    def test_workspace_link_or_inode_replacement_blocks_every_entry(self):
        for replacement in ('symlink', 'directory'):
            with self.subTest(replacement=replacement):
                workspace = self.paths['workspace']
                moved = self.paths['root'] / ('old-' + replacement)
                workspace.rename(moved)
                if replacement == 'symlink':
                    workspace.symlink_to(self.host, target_is_directory=True)
                else:
                    workspace.mkdir()
                for operation in (lambda: self.manager.read_file('sentinel'),
                                  lambda: self.manager.write_file('sentinel', 'bad'),
                                  lambda: self.manager.copy_from('/workspace/sentinel', self.root / 'leak'),
                                  lambda: self.manager.copy_to(self.sentinel, '/workspace/out'),
                                  lambda: self.manager.exec(self.identifier, 'true')):
                    with self.assertRaises((RuntimeError, OSError)):
                        operation()
                if workspace.is_symlink():
                    workspace.unlink()
                else:
                    workspace.rmdir()
                moved.rename(workspace)
        self.assertEqual(self.sentinel.read_text(), 'host-only')
        self.assertFalse((self.root / 'leak').exists())
    def test_leaf_and_nested_symlinks_never_read_or_write_host(self):
        (self.paths['workspace'] / 'leaf').symlink_to(self.sentinel)
        (self.paths['workspace'] / 'directory').symlink_to(self.host, target_is_directory=True)
        for path in ('leaf', 'directory/sentinel'):
            for operation in (lambda: self.manager.read_file(path),
                              lambda: self.manager.write_file(path, 'bad'),
                              lambda: self.manager.copy_from(path, self.root / 'leak'),
                              lambda: self.manager.copy_to(self.sentinel, '/workspace/' + path)):
                with self.assertRaises((RuntimeError, OSError)):
                    operation()
        self.assertEqual(self.sentinel.read_text(), 'host-only')
        self.assertFalse((self.root / 'leak').exists())
    def test_component_swap_between_check_and_open_is_rejected(self):
        inner = self.paths['workspace'] / 'inner'
        inner.mkdir()
        (inner / 'sentinel').write_text('inside')
        original_open = os.open
        triggered = []
        def switch(name, flags, *args, **kwargs):
            if name == 'inner' and kwargs.get('dir_fd') is not None and not triggered:
                triggered.append(True)
                inner.rename(self.paths['workspace'] / 'moved')
                inner.symlink_to(self.host, target_is_directory=True)
            return original_open(name, flags, *args, **kwargs)
        with patch.object(guard_module.os, 'open', side_effect=switch):
            with self.assertRaises(OSError):
                self.manager.read_file('inner/sentinel')
        self.assertTrue(triggered)
        self.assertEqual(self.sentinel.read_text(), 'host-only')
    def test_recursive_copy_rejects_link_and_regular_hardlink(self):
        directory = self.paths['workspace'] / 'tree'
        directory.mkdir()
        (directory / 'escape').symlink_to(self.sentinel)
        with self.assertRaises(OSError):
            self.manager.copy_from('tree', self.root / 'result')
        (directory / 'escape').unlink()
        os.link(self.sentinel, directory / 'hardlink')
        with self.assertRaises(RuntimeError):
            self.manager.read_file('tree/hardlink')
        with self.assertRaises(RuntimeError):
            self.manager.write_file('tree/hardlink', 'bad')
        with self.assertRaises(RuntimeError):
            self.manager.copy_from('tree', self.root / 'result2')
        self.assertEqual(self.sentinel.read_text(), 'host-only')
    def test_traversal_absolute_host_path_and_reset_rejected(self):
        for path in ('../fake-host/sentinel', str(self.sentinel)):
            with self.assertRaises((ValueError, OSError)):
                self.manager.read_file(path)
        with self.assertRaises(RuntimeError):
            self.manager.reset()
    def test_cleanup_does_not_follow_replaced_workspace(self):
        workspace = self.paths['workspace']
        workspace.rename(self.paths['root'] / 'old')
        workspace.symlink_to(self.host, target_is_directory=True)
        self.manager.stop(self.identifier)
        self.assertTrue(self.sentinel.exists())


if __name__ == '__main__':
    unittest.main()
