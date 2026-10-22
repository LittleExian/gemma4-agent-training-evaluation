"""Descriptor-relative native file access for the Mario sandbox adapter.

Native helpers run in the harness process, so Landlock on shell commands alone
cannot protect them. Reject links and keep all filesystem traversal anchored to
an already-open sandbox directory; never validate a resolved path then reopen it.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat


class NativeTree:
    def __init__(self, paths, expected):
        self.paths = {name: Path(value).absolute() for name, value in paths.items()}
        self.root = self.paths['root']
        if self.root.parent != Path(expected).absolute() or not self.root.name.startswith('swegemma_sandbox_'):
            raise RuntimeError('Unexpected native sandbox root')
        self.pins = {}
        for name in ('root', 'workspace', 'tmp', 'wheels', 'venv'):
            path = self.paths[name]
            if name != 'root' and path.parent != self.root:
                raise RuntimeError('Native sandbox layout changed')
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError('Native sandbox directory is a link or not a directory')
            self.pins[name] = info.st_dev, info.st_ino
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if self.identity(os.fstat(self.fd)) != self.pins['root']:
            self.close()
            raise RuntimeError('Native root changed while opening it')

    @staticmethod
    def identity(info):
        return info.st_dev, info.st_ino

    def check(self):
        for name, expected in self.pins.items():
            info = self.paths[name].lstat()
            if not stat.S_ISDIR(info.st_mode) or self.identity(info) != expected:
                raise RuntimeError(f'Native sandbox {name} identity changed')
        if self.identity(os.fstat(self.fd)) != self.pins['root']:
            raise RuntimeError('Native sandbox root descriptor changed')

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def relative(self, raw):
        path = Path(raw)
        if '..' in path.parts:
            raise ValueError('Native path traversal rejected')
        if not path.is_absolute():
            return Path('workspace') / path
        for guest, name in (('/workspace', 'workspace'), ('/tmp', 'tmp'),
                            ('/wheels', 'wheels'), ('/usr/local/bin', 'venv')):
            guest_path = Path(guest)
            if path == guest_path or guest_path in path.parents:
                suffix = path.relative_to(guest_path)
                return Path(name) / ('bin' if name == 'venv' else '') / suffix
        if path == self.root or self.root in path.parents:
            return path.relative_to(self.root)
        raise ValueError('Absolute native path is outside sandbox mappings')

    def parent(self, relative, create=False):
        parts = Path(relative).parts
        if not parts or any(part in ('..', '/') for part in parts):
            raise ValueError('Native operation requires a sandbox file path')
        descriptor = os.dup(self.fd)
        try:
            for index, name in enumerate(parts[:-1]):
                if create:
                    try:
                        os.mkdir(name, mode=0o755, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                next_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_fd
                if index == 0 and name in self.pins:
                    if self.identity(os.fstat(descriptor)) != self.pins[name]:
                        raise RuntimeError('Native pinned directory changed during traversal')
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    def open_file(self, relative, writing=False):
        parent, name = self.parent(relative, create=writing)
        flags = os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_WRONLY | os.O_CREAT if writing else os.O_RDONLY)
        try:
            descriptor = os.open(name, flags, mode=0o644, dir_fd=parent)
        finally:
            os.close(parent)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            os.close(descriptor)
            raise RuntimeError('Native file must be regular and have no hard links')
        if writing:
            os.ftruncate(descriptor, 0)
        return descriptor

    def write(self, relative, content):
        with os.fdopen(self.open_file(relative, writing=True), 'wb') as handle:
            handle.write(content.encode() if isinstance(content, str) else content)

    def read(self, relative):
        with os.fdopen(self.open_file(relative), 'rb') as handle:
            return handle.read().decode('utf-8', errors='replace')

    def is_directory(self, relative):
        try:
            parent, name = self.parent(relative)
        except FileNotFoundError:
            return False
        try:
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            return stat.S_ISDIR(info.st_mode)
        except FileNotFoundError:
            return False
        finally:
            os.close(parent)

    def copy_in(self, source, relative):
        source = Path(source)
        if source.is_symlink():
            raise RuntimeError('Native copy source link rejected')
        if source.is_dir():
            parent, name = self.parent(Path(relative) / '_entry', create=True)
            os.close(parent)
            for entry in source.iterdir():
                self.copy_in(entry, Path(relative) / entry.name)
        else:
            with source.open('rb') as incoming, os.fdopen(self.open_file(relative, writing=True), 'wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                os.fchmod(outgoing.fileno(), source.stat().st_mode & 0o777)

    def copy_out(self, relative, destination):
        destination = Path(destination)
        parent, name = self.parent(relative)
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            info = os.fstat(descriptor)
            parts = Path(relative).parts
            if len(parts) == 1 and parts[0] in self.pins and self.identity(info) != self.pins[parts[0]]:
                raise RuntimeError('Native pinned directory changed during copy')
            if stat.S_ISDIR(info.st_mode):
                destination.mkdir(parents=True, exist_ok=True)
                # Open children via this anchored descriptor, not the original path.
                self.copy_directory_out(descriptor, destination)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with os.fdopen(os.dup(descriptor), 'rb') as incoming, destination.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            else:
                raise RuntimeError('Native copy source is not a regular unlinked file/directory')
        finally:
            os.close(descriptor)

    def copy_directory_out(self, directory_fd, destination):
        for name in os.listdir(directory_fd):
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            try:
                info = os.fstat(child)
                target = destination / name
                if stat.S_ISDIR(info.st_mode):
                    target.mkdir(exist_ok=True)
                    self.copy_directory_out(child, target)
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    with os.fdopen(os.dup(child), 'rb') as incoming, target.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                else:
                    raise RuntimeError('Native recursive copy refuses symlinks/special files/hard links')
            finally:
                os.close(child)


def install_native_guards(manager, shard):
    trees = {identifier: NativeTree(paths, shard / 'tmp') for identifier, paths in manager.sandboxes.items()}
    start, execute, stop = manager.start, manager.exec, manager.stop

    def tree(identifier=None):
        selected = identifier if identifier is not None else manager._default_sandbox_id
        if selected is None:
            manager.initialize()
            selected = manager._default_sandbox_id
        guard = trees[selected]
        guard.check()
        return guard

    def guarded_start(*args, **kwargs):
        identifier = start(*args, **kwargs)
        trees[identifier] = NativeTree(manager.sandboxes[identifier], shard / 'tmp')
        return identifier

    def guarded_exec(identifier, command, **kwargs):
        tree(identifier)
        return execute(identifier, command, **kwargs)

    def read_file(path):
        guard = tree()
        return guard.read(guard.relative(path))

    def write_file(path, content):
        guard = tree()
        guard.write(guard.relative(path), content)

    def copy_to(identifier, source, destination):
        guard = tree(identifier)
        relative = guard.relative(destination)
        source = Path(source).resolve(strict=True)
        if source.is_file() and (str(destination).endswith('/') or guard.is_directory(relative)):
            relative /= source.name
        guard.copy_in(source, relative)

    def copy_from(identifier, source, destination):
        guard = tree(identifier)
        guard.copy_out(guard.relative(source), destination)

    def guarded_stop(identifier):
        # rmtree is fd-safe on Linux; do not follow replaced child directories.
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError('Safe sandbox cleanup is unavailable on this platform')
        guard = trees.get(identifier)
        if guard and guard.identity(guard.paths['root'].lstat()) != guard.pins['root']:
            raise RuntimeError('Refusing cleanup of replaced sandbox root')
        try:
            return stop(identifier)
        finally:
            if guard:
                guard.close()
                trees.pop(identifier, None)

    def reject_reset():
        raise RuntimeError('Native reset is disabled; create a fresh sandbox')

    manager.start, manager.exec, manager.stop = guarded_start, guarded_exec, guarded_stop
    manager.read_file, manager.write_file = read_file, write_file
    manager._copy_to_sandbox, manager._copy_from_sandbox = copy_to, copy_from
    manager.reset = reject_reset
    return manager
