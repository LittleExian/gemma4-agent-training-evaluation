"""Rootless, networkless containers for the unchanged official agent runner.

Only immutable, pre-pulled images are admitted. No host mount, GPU, credential,
gold patch, source record, or test payload enters the agent container.
"""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time
import uuid

from swegemma.sandbox.base import BaseSandboxManager, ExecResult


LIMIT = 10 * 1024 * 1024
LABEL = 'io.gemma.validation20.owner'

EXEC_SCRIPT = r'''
import json, os, signal, subprocess, sys, tempfile, time
p = json.load(sys.stdin)
start = time.monotonic()
limit = min(int(p.get('limit', 10485760)), 10485760)
timeout = min(float(p['timeout']), 240.0)
timed_out = False
overflow = False
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
    child = subprocess.Popen(['/bin/bash', '-lc', p['command']], cwd=p['cwd'],
                             stdout=out, stderr=err, start_new_session=True)
    while child.poll() is None:
        timed_out = time.monotonic() - start >= timeout
        overflow = os.fstat(out.fileno()).st_size > limit or os.fstat(err.fileno()).st_size > limit
        if timed_out or overflow:
            break
        time.sleep(0.02)
    # Kill only this command's process group, including descendants left behind
    # when the command exits normally. Never address a host PID or other command.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)
    out.seek(0)
    err.seek(0)
    stdout = out.read(limit).decode('utf-8', 'replace')
    stderr = err.read(limit).decode('utf-8', 'replace')
    if overflow:
        stderr = stderr[:max(0, limit-64)] + '\n[output limit reached]'
    result = dict(exit_code=124 if timed_out else (125 if overflow else child.returncode),
                  stdout=stdout, stderr=stderr, timed_out=timed_out,
                  duration_seconds=time.monotonic()-start)
    sys.stdout.write(json.dumps(result))
'''


PREPARE_SCRIPT = r'''
import errno, hashlib, json, os, shutil, stat, subprocess, sys
p = json.load(sys.stdin)
repo = p['repo_cwd']
proof = dict(head_before=None, removed_git=[], removed_artifacts=[], other_git=[],
             artifacts_remaining=[], forbidden_objects_absent=False, inaccessible_paths=[])
def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory='+repo, '-C', repo]+list(args),
                                   stderr=subprocess.STDOUT).decode().strip()
def submodule_audit(path):
    if not os.path.lexists(path):
        return {'kind': 'absent'}
    if os.path.islink(path):
        return {'kind': 'symlink', 'target': os.readlink(path)}
    if not os.path.isdir(path):
        raise RuntimeError('gitlink working tree must be a directory')
    digest = hashlib.sha256()
    files = symlinks = directories = 0
    for parent, dirs, names in os.walk(path, topdown=True, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name != '.git')
        for name in sorted(dirs+[name for name in names if name != '.git']):
            target = os.path.join(parent, name)
            relative = os.path.relpath(target, path)
            if os.path.islink(target):
                digest.update(b'L\0'+relative.encode()+b'\0'+os.readlink(target).encode()+b'\0')
                symlinks += 1
            elif os.path.isdir(target):
                digest.update(b'D\0'+relative.encode()+b'\0')
                directories += 1
            elif os.path.isfile(target):
                digest.update(b'F\0'+relative.encode()+b'\0')
                with open(target, 'rb') as handle:
                    while True:
                        chunk = handle.read(1048576)
                        if not chunk:
                            break
                        digest.update(chunk)
                digest.update(b'\0')
                files += 1
            else:
                raise RuntimeError('unsupported node in gitlink working tree: '+target)
    return {'kind': 'directory', 'sha256': digest.hexdigest(), 'files': files,
            'symlinks': symlinks, 'directories': directories}
def source_digest(paths, gitlinks=None):
    gitlinks = gitlinks or {}
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = os.path.join(repo, relative)
        digest.update(relative.encode()+b'\0')
        if relative in gitlinks:
            digest.update(b'GITLINK\0'+gitlinks[relative].encode()+b'\0'+
                          json.dumps(submodule_audit(path), sort_keys=True).encode()+b'\0')
        elif os.path.islink(path):
            digest.update(b'LINK\0'+os.readlink(path).encode()+b'\0')
        else:
            with open(path, 'rb') as handle:
                while True:
                    chunk = handle.read(1048576)
                    if not chunk:
                        break
                    digest.update(chunk)
            digest.update(b'\0')
    return digest.hexdigest()
def erase(path):
    parent = os.path.dirname(path)
    info = os.stat(parent)
    mode = stat.S_IMODE(info.st_mode)
    changed = info.st_uid == os.geteuid() and not mode & stat.S_IWUSR
    if changed:
        os.chmod(parent, mode | stat.S_IWUSR)
    try:
        if os.path.islink(path) or not os.path.isdir(path):
            os.unlink(path)
        else:
            def repair(function, target, error):
                if os.stat(target).st_uid != os.geteuid():
                    raise error[1]
                os.chmod(target, 0o700)
                function(target)
            if os.stat(path).st_uid == os.geteuid():
                os.chmod(path, 0o700)
            shutil.rmtree(path, onerror=repair)
    finally:
        if changed:
            os.chmod(parent, mode)
def inventory():
    histories, artifacts = [], []
    exact = {'r2e_tests', 'gold.patch', 'test.patch', 'parsed_commit.json',
             'exec_result.json', 'source-records', 'source_records'}
    def failed(error):
        # CAP_DAC_OVERRIDE, CAP_FOWNER, CAP_SETUID and privilege gains are disabled.
        # A foreign-owned, unwritable directory inaccessible here is equally
        # inaccessible to every subsequent agent process; record it explicitly.
        info = os.stat(error.filename)
        if (error.errno == errno.EACCES and info.st_uid != os.geteuid() and
                not os.access(error.filename, os.W_OK)):
            if error.filename not in proof['inaccessible_paths']:
                proof['inaccessible_paths'].append(error.filename)
            return
        raise error
    for parent, dirs, files in os.walk('/', topdown=True, followlinks=False, onerror=failed):
        if parent == '/':
            dirs[:] = [d for d in dirs if d not in ('proc', 'sys', 'dev')]
        if (parent != '/' and 'HEAD' in files and 'objects' in dirs and
                ('config' in files or 'refs' in dirs)):
            histories.append(parent)
            dirs[:] = []
            continue
        for name in list(dirs)+files:
            path = os.path.join(parent, name)
            lower = name.lower()
            if name == '.git':
                histories.append(path)
                if name in dirs:
                    dirs.remove(name)
            elif name in exact or (lower.endswith('.patch') and any(
                    lower.startswith(token+separator) for token in ('gold', 'reference', 'solution', 'r2e')
                    for separator in ('.', '-', '_'))):
                artifacts.append(path)
                if name in dirs:
                    dirs.remove(name)
    return histories, artifacts
try:
    if os.path.islink(repo) or not os.path.isdir(repo):
        raise RuntimeError('repository must be a real directory')
    proof['original_head'] = git('rev-parse', 'HEAD')
    proof['original_ref_objects'] = sorted(set(git('for-each-ref', '--format=%(objectname)').split()))
    git('checkout', '--force', p['resolved_base_commit'])
    proof['head_before'] = git('rev-parse', 'HEAD')
    if proof['head_before'] != p['resolved_base_commit']:
        raise RuntimeError('base commit mismatch')
    git('reset', '--hard', p['resolved_base_commit'])
    tracked = [path for path in git('ls-files', '-z').split('\0') if path]
    gitlinks = {}
    for entry in git('ls-files', '--stage', '-z').split('\0'):
        if not entry:
            continue
        metadata, separator, path = entry.partition('\t')
        mode, object_id, stage = metadata.split()
        if not separator or stage != '0':
            raise RuntimeError('unexpected staged repository metadata')
        if mode == '160000':
            gitlinks[path] = object_id
    proof['tracked_file_count'] = len(tracked)
    proof['gitlink_commits'] = gitlinks
    proof['submodule_source_trees_before'] = {
        path: submodule_audit(os.path.join(repo, path)) for path in sorted(gitlinks)}
    proof['source_tree_before'] = source_digest(tracked, gitlinks)
    histories, artifacts = inventory()
    # Remove every original Git object database, including alternate source copies.
    # Keep all actual repository/venv locations in place for editable installs.
    for path in sorted(histories, key=len, reverse=True):
        erase(path)
        proof['removed_git'].append(path)
    for path in sorted(artifacts, key=len, reverse=True):
        if os.path.lexists(path):
            erase(path)
            proof['removed_artifacts'].append(path)
    git('init', '-q')
    git('config', 'user.email', 'validation@localhost')
    git('config', 'user.name', 'Validation baseline')
    with open(repo+'/.git/info/exclude', 'w') as handle:
        handle.write('.venv/\n__pycache__/\n.pytest_cache/\n*.pyc\n.coverage*\n'
                     '*.so\n*.o\n*.egg-info/\nbuild/\ndist/\n.cache/\n')
    regular_tracked = [path for path in tracked if path not in gitlinks]
    for start in range(0, len(regular_tracked), 100):
        git('add', '--force', '--', *regular_tracked[start:start+100])
    for path, object_id in sorted(gitlinks.items()):
        # Retain the index gitlink without importing its old commit object or
        # flattening compiled/untracked dependency files into the new baseline.
        git('update-index', '--add', '--cacheinfo', '160000', object_id, path)
    git('commit', '-q', '--allow-empty', '-m', 'Sanitized validation baseline')
    proof['baseline_head'] = git('rev-parse', 'HEAD')
    proof['history_count'] = int(git('rev-list', '--all', '--count'))
    if proof['history_count'] != 1:
        raise RuntimeError('baseline must contain one commit')
    proof['submodule_source_trees_after'] = {
        path: submodule_audit(os.path.join(repo, path)) for path in sorted(gitlinks)}
    proof['source_tree_after'] = source_digest(tracked, gitlinks)
    if proof['source_tree_before'] != proof['source_tree_after']:
        raise RuntimeError('base source changed during sanitization')
    absent = []
    checked = sorted(set(p.get('forbidden_commits', []) + proof['original_ref_objects'] +
                         [proof['original_head'], p['resolved_base_commit']] + list(gitlinks.values())))
    for sha in checked:
        rc = subprocess.call(['git', '-C', repo, 'cat-file', '-e', sha],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        absent.append(rc != 0)
    proof['forbidden_objects_absent'] = all(absent)
    proof['forbidden_commits_checked'] = p.get('forbidden_commits', [])
    proof['original_objects_checked'] = checked
    histories, artifacts = inventory()
    proof['other_git'] = [path for path in histories if path != repo+'/.git']
    proof['artifacts_remaining'] = artifacts
    if proof['other_git'] or artifacts or not all(absent):
        raise RuntimeError('reference leakage gate failed')
    if (repo != '/workspace' and os.path.isdir('/workspace') and not os.path.islink('/workspace') and
            set(os.listdir('/workspace')) <= {'.cache'}):
        if os.path.lexists('/workspace/.cache'):
            if os.path.islink('/workspace/.cache') or os.path.lexists(repo+'/.cache'):
                raise RuntimeError('unexpected workspace cache symlink or destination collision')
            # OverlayFS lower-layer directory renames may return EXDEV even on
            # the same device. move preserves the cache via copy/unlink fallback.
            shutil.move('/workspace/.cache', repo+'/.cache')
            proof['workspace_cache_relocated'] = repo+'/.cache'
        erase('/workspace')
        proof['empty_workspace_replaced'] = True
    if os.path.lexists('/workspace'):
        if os.path.realpath('/workspace') == repo and (os.path.islink('/workspace') or repo == '/workspace'):
            pass
        else:
            raise RuntimeError('unexpected existing /workspace; refusing to hide another tree')
    else:
        mode = stat.S_IMODE(os.stat('/').st_mode)
        os.chmod('/', mode | stat.S_IWUSR)
        try:
            os.symlink(repo, '/workspace')
        finally:
            os.chmod('/', mode)
    proof['workspace_target'] = os.path.realpath('/workspace')
    proof['python_exists'] = os.path.isfile(p['python_executable'])
    if not proof['python_exists']:
        raise RuntimeError('baked task Python missing after sanitization')
    proof['passed'] = True
except Exception as error:
    proof['passed'] = False
    proof['error'] = type(error).__name__+': '+str(error)
print(json.dumps(proof))
if not proof['passed']:
    sys.exit(1)
'''


COPY_TO_SCRIPT = r'''
import base64, json, os, sys
p = json.load(sys.stdin)
data = base64.b64decode(p['data'], validate=True)
if len(data) > 10485760:
    raise ValueError('file exceeds transfer limit')
fd = os.open(p['path'], os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, 'wb') as handle:
    handle.write(data)
'''

COPY_FROM_SCRIPT = r'''
import base64, json, os, stat, sys
p = json.load(sys.stdin)
fd = os.open(p['path'], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
with os.fdopen(fd, 'rb') as handle:
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > 10485760:
        raise ValueError('only bounded regular files may be copied')
    data = handle.read(10485761)
if len(data) > 10485760:
    raise ValueError('file exceeds transfer limit')
sys.stdout.write(base64.b64encode(data).decode('ascii'))
'''


class PodmanManager(BaseSandboxManager):
    """Each manager owns exactly its receipts and never trusts image-supplied IDs."""

    label = LABEL

    def __init__(self, runtime: dict, output: Path):
        self.runtime = dict(runtime)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.tmp = self.output / 'tmp'
        self.tmp.mkdir(exist_ok=True, mode=0o700)
        if self.tmp.is_symlink():
            raise PermissionError('manager temporary root must not be a symlink')
        self.owner = uuid.uuid4().hex
        self._owned: dict[str, str] = {}
        self._active: str | None = None
        self.pids_limit = self.runtime.get('container_pids_limit', 256)
        if type(self.pids_limit) is not int or not 256 <= self.pids_limit <= 1024:
            raise ValueError('container_pids_limit must be an integer from 256 to 1024')
        if self.pids_limit != 256 and self.runtime.get('task_id') != 'tobymao__sqlglot-899':
            raise ValueError('Only the calibrated SQLglot-899 runtime may increase its PID limit')
        image = self.runtime.get('image', '')
        if not re.fullmatch(r'[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}', image):
            raise ValueError('immutable image digest required')
        if not re.fullmatch(r'[0-9a-f]{40}', self.runtime.get('resolved_base_commit', '')):
            raise ValueError('exact base commit required')
        cwd = self.runtime.get('repo_cwd', '')
        reserved = ('/proc', '/sys', '/dev', '/etc', '/usr', '/bin', '/sbin', '/lib', '/run')
        if (not re.fullmatch(r'/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', cwd) or
                any(part in ('.', '..') for part in Path(cwd).parts) or
                any(cwd == prefix or cwd.startswith(prefix+'/') for prefix in reserved)):
            raise ValueError('safe absolute original repository directory required')
        executable = self.runtime.get('python_executable', '')
        if not executable.startswith('/') or '\x00' in executable or '..' in Path(executable).parts:
            raise ValueError('absolute baked Python path required')
        for commit in self.runtime.get('forbidden_commits', []):
            if not re.fullmatch(r'[0-9a-f]{40}', commit):
                raise ValueError('forbidden commits must be exact hashes')

    def _podman(self, args, *, input_data=None, timeout=60):
        """Bound both process lifetime and captured bytes; never inherit secret env."""
        env = {key: os.environ[key] for key in ('PATH', 'HOME', 'USER', 'LOGNAME',
               'XDG_RUNTIME_DIR') if key in os.environ}
        with tempfile.TemporaryFile(dir=self.tmp) as out, tempfile.TemporaryFile(dir=self.tmp) as err:
            child = subprocess.Popen(['podman', *args], stdin=subprocess.PIPE,
                                     stdout=out, stderr=err, env=env, start_new_session=True)
            # A bounded feeder avoids blocking on a malicious exec which never reads stdin.
            import threading
            def feed():
                try:
                    child.stdin.write(input_data or b'')
                    child.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            feeder = threading.Thread(target=feed, daemon=True)
            feeder.start()
            start = time.monotonic()
            failure = None
            while child.poll() is None:
                if time.monotonic() - start >= timeout:
                    failure = subprocess.TimeoutExpired(['podman', *args], timeout)
                    break
                if max(os.fstat(out.fileno()).st_size, os.fstat(err.fileno()).st_size) > 4 * LIMIT:
                    failure = RuntimeError('podman output exceeded outer transfer limit')
                    break
                time.sleep(.02)
            if failure is not None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=10)
            feeder.join(timeout=1)
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read(4*LIMIT+1), err.read(4*LIMIT+1)
            if failure is not None:
                raise failure
            if len(stdout) > 4*LIMIT or len(stderr) > 4*LIMIT:
                raise RuntimeError('podman output exceeded outer transfer limit')
            if child.returncode:
                raise RuntimeError('podman failed: '+stderr.decode('utf-8', 'replace')[-4000:]+
                                   '; stdout: '+stdout.decode('utf-8', 'replace')[-4000:])
            return stdout

    def _verify_owned(self, sandbox_id):
        if not re.fullmatch(r'[0-9a-f]{64}', sandbox_id) or self._owned.get(sandbox_id) != self.owner:
            raise PermissionError('container is not owned by this manager')
        inspected = json.loads(self._podman(['inspect', sandbox_id], timeout=15))
        if (len(inspected) != 1 or inspected[0].get('Id') != sandbox_id or
                inspected[0].get('Config', {}).get('Labels', {}).get(self.label) != self.owner):
            raise PermissionError('container ID or ownership label changed')

    def _receipt(self):
        target = self.output / 'owned-containers.json'
        temp = self.output / ('receipt-'+uuid.uuid4().hex+'.tmp')
        temp.write_text(json.dumps({'owner': self.owner, 'label': self.label,
                                   'containers': self._owned}, indent=2))
        os.replace(temp, target)

    def start(self):
        args = ['run', '--detach', '--pull=never', '--network=none', '--cpus=2',
                '--memory=4g', '--pids-limit='+str(self.pids_limit), '--cap-drop=all',
                '--security-opt=no-new-privileges', '--label='+self.label+'='+self.owner,
                '--entrypoint=/bin/sh']
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
            args.append('--env='+key+'=1')
        args += [self.runtime['image'], '-c', 'while :; do sleep 3600; done']
        cid = self._podman(args, timeout=60).decode().strip()
        if not re.fullmatch(r'[0-9a-f]{64}', cid):
            raise RuntimeError('podman returned an invalid full container ID')
        self._owned[cid] = self.owner
        self._receipt()
        try:
            self._verify_owned(cid)
            prepare = {key: self.runtime[key] for key in ('repo_cwd', 'resolved_base_commit',
                       'python_executable', 'forbidden_commits') if key in self.runtime}
            raw = self._podman(['exec', '-i', cid, self.runtime['python_executable'],
                                '-c', PREPARE_SCRIPT],
                               input_data=json.dumps(prepare).encode(), timeout=180)
            proof = json.loads(raw)
            (self.output/f'sandbox-proof-{cid}.json').write_text(json.dumps(proof, indent=2))
            if proof.get('passed') is not True or proof.get('history_count') != 1:
                raise RuntimeError('container sanitization did not pass')
        except BaseException as error:
            (self.output/f'sandbox-start-error-{cid}.json').write_text(json.dumps({
                'container_id': cid, 'error': str(error), 'passed': False}, indent=2))
            try:
                self.stop(cid)
            except Exception as cleanup_error:
                error.add_note('Owned container cleanup also failed: '+str(cleanup_error))
            raise
        self._active = cid
        return cid

    def exec(self, sandbox_id, command, *, timeout=None):
        self._verify_owned(sandbox_id)
        requested = 240 if timeout is None else float(timeout)
        if not math.isfinite(requested):
            raise ValueError('finite command timeout required')
        seconds = max(.01, min(requested, 240))
        if not isinstance(command, str) or '\x00' in command:
            raise ValueError('command must be a NUL-free string')
        payload = {'command': command, 'timeout': seconds, 'limit': LIMIT,
                   'cwd': self.runtime['repo_cwd']}
        try:
            raw = self._podman(['exec', '-i', sandbox_id, self.runtime['python_executable'],
                                '-c', EXEC_SCRIPT], input_data=json.dumps(payload).encode(),
                               timeout=seconds+20)
        except subprocess.TimeoutExpired:
            self.stop(sandbox_id)
            return ExecResult(exit_code=124, stderr='outer container exec timeout; owned sandbox removed',
                              timed_out=True, duration_seconds=seconds+20)
        except RuntimeError:
            self.stop(sandbox_id)
            raise
        result = json.loads(raw)
        return ExecResult(**result)

    def _copy_args(self, first, second, third):
        cid, source, destination = (self._active, first, second) if third is None else (first, second, third)
        if cid is None:
            raise RuntimeError('no active sandbox')
        self._verify_owned(str(cid))
        return str(cid), Path(source), str(destination)

    def _host_fd(self, path, *, writing):
        """Walk host temporary files via dirfds with no symlinks, including parents."""
        path = Path(path)
        if not path.is_absolute() or '..' in path.parts:
            raise PermissionError('host transfers require absolute paths without traversal')
        try:
            relative = path.relative_to(self.tmp)
        except ValueError as error:
            raise PermissionError('host transfer is outside manager temporary root') from error
        if not relative.parts:
            raise PermissionError('a regular temporary file is required')
        parent = os.open(self.tmp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in relative.parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = next_fd
            flags = os.O_WRONLY | os.O_CREAT if writing else os.O_RDONLY | os.O_NONBLOCK
            fd = os.open(relative.name, flags | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > LIMIT:
                os.close(fd)
                raise PermissionError('only bounded, single-link regular host files are allowed')
            if writing:
                os.ftruncate(fd, 0)
            return fd
        finally:
            os.close(parent)

    def copy_to(self, src_or_container, dst_or_src, dst=None):
        cid, source, destination = self._copy_args(src_or_container, dst_or_src, dst)
        with os.fdopen(self._host_fd(source, writing=False), 'rb') as handle:
            data = handle.read(LIMIT+1)
        if len(data) > LIMIT:
            raise ValueError('file exceeds transfer limit')
        payload = {'path': destination, 'data': base64.b64encode(data).decode()}
        self._transfer(cid, COPY_TO_SCRIPT, json.dumps(payload).encode())

    def _transfer(self, cid, script, payload):
        try:
            return self._podman(['exec', '-i', cid, self.runtime['python_executable'], '-c', script],
                                input_data=payload, timeout=30)
        except subprocess.TimeoutExpired as error:
            try:
                self.stop(cid)
            except Exception as cleanup_error:
                error.add_note('Owned container cleanup also failed: '+str(cleanup_error))
            raise

    def copy_from(self, src_or_container, dst_or_src, dst=None):
        cid, source, destination = self._copy_args(src_or_container, dst_or_src, dst)
        # Validate/open before requesting bytes; an image can never choose a host path.
        with os.fdopen(self._host_fd(destination, writing=True), 'wb') as handle:
            encoded = self._transfer(cid, COPY_FROM_SCRIPT, json.dumps({'path': str(source)}).encode())
            data = base64.b64decode(encoded, validate=True)
            if len(data) > LIMIT:
                raise ValueError('file exceeds transfer limit')
            handle.write(data)

    def stop(self, sandbox_id):
        self._verify_owned(sandbox_id)
        self._podman(['rm', '--force', '--time=2', sandbox_id], timeout=30)
        del self._owned[sandbox_id]
        self._receipt()
        if self._active == sandbox_id:
            self._active = None

    def cleanup_all(self):
        errors = []
        for cid in tuple(self._owned):
            try:
                self.stop(cid)
            except Exception as error:
                errors.append(f'{cid}: {error}')
        if errors:
            raise RuntimeError('; '.join(errors))

    @property
    def work_dir(self):
        return '/workspace'

    def initialize(self, files=None):
        if self._active is None:
            self.start()
        if files:
            self.get_sandbox(self._active).initialize(files)

    def run_command(self, command, timeout=240, work_dir=None):
        import shlex
        if self._active is None:
            raise RuntimeError('no active sandbox')
        wd = work_dir or self.work_dir
        return self.exec(self._active, 'cd '+shlex.quote(wd)+' && '+command, timeout=timeout)

    def write_file(self, path, content):
        return self.get_sandbox(self._active).write_file(path, content)

    def read_file(self, path):
        return self.get_sandbox(self._active).read_file(path)

    def reset(self):
        raise RuntimeError('create a fresh immutable-image sandbox instead of resetting')

    def close(self):
        self.cleanup_all()
