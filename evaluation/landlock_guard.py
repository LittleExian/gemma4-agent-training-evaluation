#!/usr/bin/env python3
"""Execute a command with an inherited, fail-closed Landlock filesystem policy.

Linux x86_64 only. Read paths permit reading and execution. Write paths also
permit ordinary filesystem changes, but never creation of block/character
devices. Supply every required interpreter/library path explicitly. Example:

    python landlock_guard.py --read /usr --read /etc --write /tmp/task -- python ...

This limits filesystem access, not network access, CPU, memory, signals, or
already-open standard input/output/error. Additional inherited descriptors are
closed. The caller must not pass secret-bearing standard descriptors.
Broad /proc access exposes process metadata and may expose secrets
in other processes' environments; prefer narrowly selected /proc entries.
"""
import argparse
import ctypes
import os
import platform
import stat
import sys


EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ_FILE = 1 << 2
READ_DIR = 1 << 3
MAKE_CHAR = 1 << 6
MAKE_BLOCK = 1 << 11
REFER = 1 << 13
TRUNCATE = 1 << 14
FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE
READ_RIGHTS = EXECUTE | READ_FILE | READ_DIR
CREATE_RULESET = 444
ADD_RULE = 445
RESTRICT_SELF = 446
RULE_PATH_BENEATH = 1
CREATE_RULESET_VERSION = 1
PR_SET_NO_NEW_PRIVS = 38


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64),
                ("parent_fd", ctypes.c_int32)]


def handled_rights(abi):
    """Handle all filesystem rights available through ABI 4."""
    if abi < 1:
        raise RuntimeError("Landlock ABI 1 or newer is required")
    return (1 << (13 if abi == 1 else 14 if abi == 2 else 15)) - 1


def allowed_rights(abi, writable, is_directory):
    mask = handled_rights(abi)
    allowed = mask & ~(MAKE_CHAR | MAKE_BLOCK) if writable else READ_RIGHTS
    return allowed & mask if is_directory else allowed & mask & FILE_RIGHTS


def checked_syscall(libc, number, *args):
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def add_path(libc, ruleset_fd, path, abi, writable):
    # O_PATH resolves symlinks once, binds the rule to the actual inode, and
    # avoids opening file contents. O_CLOEXEC prevents leaking policy fds.
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        mode = os.fstat(path_fd).st_mode
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode) and not stat.S_ISCHR(mode):
            raise ValueError("Rule path must be a directory, regular file, or character device: " + path)
        rule = PathBeneathAttr(allowed_rights(abi, writable, stat.S_ISDIR(mode)), path_fd)
        checked_syscall(libc, ADD_RULE, ctypes.c_int(ruleset_fd),
                        ctypes.c_int(RULE_PATH_BENEATH), ctypes.byref(rule),
                        ctypes.c_uint(0))
    finally:
        os.close(path_fd)


def restrict(read_paths, write_paths):
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("This launcher requires Linux x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.restype = ctypes.c_int
    abi = checked_syscall(libc, CREATE_RULESET, ctypes.c_void_p(0),
                          ctypes.c_size_t(0), ctypes.c_uint(CREATE_RULESET_VERSION))
    attr = RulesetAttr(handled_rights(abi))
    ruleset_fd = checked_syscall(libc, CREATE_RULESET, ctypes.byref(attr),
                                ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint(0))
    try:
        for path in read_paths:
            add_path(libc, ruleset_fd, path, abi, False)
        for path in write_paths:
            add_path(libc, ruleset_fd, path, abi, True)
        if libc.prctl(ctypes.c_int(PR_SET_NO_NEW_PRIVS), ctypes.c_ulong(1),
                      ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, "PR_SET_NO_NEW_PRIVS failed: " + os.strerror(error))
        checked_syscall(libc, RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint(0))
    finally:
        os.close(ruleset_fd)
    return abi


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--read", action="append", default=[], metavar="PATH")
    parser.add_argument("--write", action="append", default=[], metavar="PATH")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command following -- is required")
    if not args.read and not args.write:
        parser.error("at least one explicit read or write path is required")
    return args.read, args.write, command


def close_inherited_descriptors():
    # This launcher is single-threaded. Enumerate before applying the policy;
    # Landlock cannot revoke access through already-open file descriptors.
    if sys.platform != "linux":
        return
    for name in os.listdir("/proc/self/fd"):
        descriptor = int(name)
        if descriptor > 2:
            try:
                os.close(descriptor)
            except OSError as error:
                # The descriptor used by listdir itself has already closed.
                if error.errno != 9:
                    raise


def main(argv=None):
    read_paths, write_paths, command = parse_args(argv)
    try:
        close_inherited_descriptors()
        abi = restrict(read_paths, write_paths)
        print("Landlock filesystem confinement active (ABI %d)" % abi,
              file=sys.stderr, flush=True)
        os.execvp(command[0], command)
    except (OSError, RuntimeError, ValueError) as error:
        print("Landlock launcher refused to execute: %s" % error,
              file=sys.stderr, flush=True)
        return 126


if __name__ == "__main__":
    sys.exit(main())
