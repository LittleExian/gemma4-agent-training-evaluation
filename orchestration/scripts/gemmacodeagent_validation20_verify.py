"""Fresh sanitized, mount-free verification shared by R2E and SWE tasks.

Only required hidden tests are copied in. Gold/source records never enter the
container. Hidden-file hashes detect tampering; this preserves the benchmark's
normal writable repository and does not claim OS isolation against malicious
same-UID test execution which rewrites and restores files between observations.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import tempfile
import uuid
import xml.etree.ElementTree as ET

from gemmacodeagent_validation20_calibrate import patch_paths, parse_junit, validate_candidate
from gemmacodeagent_validation20_podman import PodmanManager, LIMIT


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(value):
    parts = PurePosixPath(value).parts
    if (not value or value.startswith('/') or '..' in parts or '.' in value.split('/')
            or '.git' in parts or '//' in value or any(ord(char) < 32 for char in value)):
        raise ValueError('Unsafe hidden-test relative path: '+repr(value))
    return value


def regular_bytes(path, root=None):
    path = Path(path)
    if root is not None:
        root = Path(root)
        relative = path.relative_to(root)
        current = root
        if current.is_symlink():
            raise PermissionError('Symlink payload root')
        for part in relative.parts:
            current = current/part
            if current.is_symlink():
                raise PermissionError('Symlink in hidden payload path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT:
            raise ValueError('Only bounded regular payload/patch files are accepted')
        data = handle.read(LIMIT+1)
    if len(data) > LIMIT:
        raise ValueError('Payload/patch exceeds transfer limit')
    return data


def hidden_plan(runtime):
    payload = Path(runtime['payload_dir'])
    if not payload.is_absolute() or not payload.is_dir():
        raise ValueError('payload_dir must be an existing absolute trusted directory')
    cwd = runtime['repo_cwd']
    plan = []
    if runtime['task_id'].startswith('r2e_'):
        names = sorted(set(['__init__.py', *runtime['test_files']]))
        for name in names:
            name = safe_relative(name)
            if not name.endswith('.py'):
                raise ValueError('R2E hidden test source must be a Python file')
            plan.append({'source': payload/'r2e_tests'/name,
                         'destination': cwd+'/r2e_tests/'+name, 'kind': 'file'})
        if '/probe/run_unittest_junit.py' in runtime['test_command']:
            plan.append({'source': payload/'run_unittest_junit.py',
                         'destination': '/probe/run_unittest_junit.py', 'kind': 'file'})
    else:
        plan.append({'source': payload/'test.patch', 'destination': '/tmp/validation-hidden.patch',
                     'kind': 'patch'})
    pinned = runtime.get('payload_sha256', {})
    for item in plan:
        data = regular_bytes(item['source'], payload)
        key = str(item['source'].relative_to(payload))
        item['sha256'] = sha(data)
        if pinned and pinned.get(key) != item['sha256']:
            raise ValueError('Pinned hidden payload changed: '+key)
    return plan


def validate_patch(text, hidden_paths, cwd):
    relative = [path[len(cwd)+1:] for path in hidden_paths if path.startswith(cwd+'/')]
    paths = validate_candidate(text, relative)
    protected = {'gold.patch', 'old-files.json', 'expected-output.json', 'hidden-files.json',
                 'run_unittest_junit.py', 'verify_base.py', 'verify_hidden.py'}
    for path in paths:
        parts = PurePosixPath(path).parts
        if 'r2e_tests' in parts or PurePosixPath(path).name in protected:
            raise ValueError('Candidate changes protected verification path: '+path)
    # Git's parser additionally rejects malformed/binary/rename representations
    # which the conservative unified-path check alone cannot validate.
    if text.strip():
        import subprocess
        parsed = subprocess.run(['git', 'apply', '--numstat', '-z'], input=text.encode(),
                                capture_output=True, timeout=10)
        if parsed.returncode:
            raise ValueError('Candidate is not a valid Git patch')
        for record in parsed.stdout.decode().split('\0'):
            if record:
                fields = record.split('\t', 2)
                if len(fields) != 3 or fields[2] not in paths:
                    raise ValueError('Unsupported candidate rename or hidden path')
    return paths


def assess_result(runtime, exit_code, outcomes, *, failure_kind=None, infra_error=False, **extra):
    expected = runtime.get('results', {}).get('gold', {}).get('test_outcomes', {})
    covered = bool(expected) and set(outcomes) == set(expected)
    resolved = (not infra_error and exit_code == 0 and covered and outcomes == expected
                and any(value == 'passed' for value in outcomes.values()))
    counts = {label: sum(state == label for state in outcomes.values())
              for label in ('passed', 'failed', 'error')}
    return {'task_id': runtime['task_id'], 'exit_code': int(exit_code),
            'test_exit_code': int(exit_code), 'passed': counts['passed'],
            'failed': counts['failed'], 'errors': counts['error'],
            'test_count': len(outcomes), 'test_outcomes': outcomes,
            'expected_tests_covered': covered, 'collection_matches_gold': covered,
            'infra_error': bool(infra_error), 'infra_failed': bool(infra_error),
            'resolved': resolved, 'failure_kind': None if resolved else
            (failure_kind or 'candidate_test_failure'), **extra}


def r2e_junit(text):
    priorities = {'passed': 0, 'skipped': 1, 'failed': 2, 'error': 3}
    outcomes = {}
    for case in ET.fromstring(text).iter('testcase'):
        name = case.get('classname', '')+'::'+case.get('name', '')
        state = next((state for tag, state in [('error', 'error'), ('failure', 'failed'),
                     ('skipped', 'skipped')] if case.find(tag) is not None), 'passed')
        if priorities[state] > priorities.get(outcomes.get(name), -1):
            outcomes[name] = state
    return outcomes


HASH_SCRIPT = r'''
# VALIDATION_HIDDEN_HASH
import hashlib,json,os,stat
paths=json.loads(__PAYLOAD__)
result={}
for path in paths:
 try:
  info=os.lstat(path)
  if stat.S_ISREG(info.st_mode):
   with open(path,'rb') as handle: digest=hashlib.sha256(handle.read()).hexdigest()
   result[path]={'kind':'file','sha256':digest}
  elif stat.S_ISLNK(info.st_mode):
   result[path]={'kind':'symlink','target':os.readlink(path)}
  else: result[path]={'kind':'other'}
 except FileNotFoundError: result[path]={'kind':'absent'}
print(json.dumps(result,sort_keys=True))
'''


def python_command(runtime, script):
    return shlex.quote(runtime['python_executable'])+' -c '+shlex.quote(script)


def checked(manager, cid, command, timeout=120):
    result = manager.exec(cid, command, timeout=timeout)
    if result.exit_code != 0:
        raise RuntimeError('Trusted verification setup failed: '+result.stderr+' '+result.stdout)
    return result


def snapshot(manager, cid, runtime, paths):
    script = HASH_SCRIPT.replace('__PAYLOAD__', repr(json.dumps(sorted(paths))))
    return json.loads(checked(manager, cid, python_command(runtime, script)).stdout)


def send_bytes(manager, cid, data, destination):
    temporary = manager.tmp/('transfer-'+uuid.uuid4().hex)
    temporary.write_bytes(data)
    try:
        manager.copy_to(cid, temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def install_hidden(manager, cid, runtime, plan):
    cwd = runtime['repo_cwd']
    make = "import os,stat; mode=stat.S_IMODE(os.stat('/').st_mode); os.chmod('/',mode|stat.S_IWUSR); "
    make += "os.makedirs('/probe',exist_ok=True); os.makedirs('/out',exist_ok=True); os.chmod('/',mode)"
    checked(manager, cid, python_command(runtime, make))
    hidden = []
    for item in plan:
        destination = item['destination']
        checked(manager, cid, 'mkdir -p -- '+shlex.quote(str(PurePosixPath(destination).parent)))
        send_bytes(manager, cid, regular_bytes(item['source'], Path(runtime['payload_dir'])), destination)
        if item['kind'] == 'patch':
            text = regular_bytes(item['source'], Path(runtime['payload_dir'])).decode()
            hidden.extend(cwd+'/'+path for path in patch_paths(text))
            checked(manager, cid, 'cd '+shlex.quote(cwd)+' && git apply --whitespace=nowarn '+shlex.quote(destination))
            checked(manager, cid, 'rm -f -- '+shlex.quote(destination))
        else:
            hidden.append(destination)
    return sorted(set(hidden))


def execute_test(runtime, patch_file, output, *, trusted_control=False):
    """Fresh container for any control/candidate. No filename-based gold bypass."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manager = None
    phase = 'trusted_payload'
    outcomes = {}
    result = None
    try:
        plan = hidden_plan(runtime)
        phase = 'candidate_validate'
        try:
            patch = b'' if patch_file is None else regular_bytes(Path(patch_file))
            text = patch.decode('utf-8')
        except (ValueError, UnicodeDecodeError, PermissionError) as error:
            result = assess_result(runtime, 1, {}, failure_kind='candidate_invalid_patch', error=str(error))
            return result
        cwd = runtime['repo_cwd']
        prospective_hidden = ([item['destination'] for item in plan if item['kind'] == 'file']
                              + [cwd+'/'+path for item in plan if item['kind'] == 'patch'
                                 for path in patch_paths(regular_bytes(item['source']).decode())])
        try:
            changed = patch_paths(text) if trusted_control else validate_patch(text, prospective_hidden, cwd)
        except ValueError as error:
            result = assess_result(runtime, 1, {}, failure_kind='candidate_invalid_patch', error=str(error))
            return result
        phase = 'container_start'
        manager = PodmanManager(runtime, output/'sandbox')
        cid = manager.start()
        phase = 'hidden_install'
        hidden = install_hidden(manager, cid, runtime, plan)
        before = snapshot(manager, cid, runtime, hidden)
        save(output/'injected-hidden.json', [{'payload_relative': str(item['source'].relative_to(
            Path(runtime['payload_dir']))), 'destination': item['destination'],
            'sha256': item['sha256'], 'kind': item['kind']} for item in plan])
        phase = 'candidate_apply'
        if patch.strip():
            stage = '/tmp/validation-candidate.patch'
            send_bytes(manager, cid, patch, stage)
            applied = manager.exec(cid, 'cd '+shlex.quote(cwd)+' && git apply --whitespace=nowarn '+stage,
                                   timeout=60)
            checked(manager, cid, 'rm -f -- '+stage)
            if applied.exit_code:
                (output/'apply.log').write_text(applied.stdout+applied.stderr)
                result = assess_result(runtime, applied.exit_code, {},
                    failure_kind='candidate_invalid_patch', patch_apply_failed=True)
                return result
        after_apply = snapshot(manager, cid, runtime, hidden)
        if before != after_apply:
            result = assess_result(runtime, 1, {}, failure_kind='candidate_changed_hidden_tests')
            save(output/'hidden-test-hashes.json', {'before': before, 'after_apply': after_apply})
            return result
        # Explicit leakage proof immediately before candidate code executes.
        clean = "import os; assert not any(os.path.exists(p) for p in "
        clean += repr(['/tmp/validation-candidate.patch', '/tmp/validation-hidden.patch',
                       '/probe/gold.patch', '/probe/old-files.json', '/probe/expected-output.json'])+')'
        checked(manager, cid, python_command(runtime, clean))
        phase = 'test_running'
        command = 'cd '+shlex.quote(cwd)+' && '+runtime['test_command']
        save(output/'executed-command.json', {'command': runtime['test_command'],
             'repo_cwd': cwd, 'timeout_seconds': 240, 'trusted_control': bool(trusted_control)})
        executed = manager.exec(cid, command, timeout=240)
        (output/'test.log').write_text(executed.stdout+executed.stderr)
        phase = 'test_finished'
        if executed.timed_out and 'owned sandbox removed' in executed.stderr:
            result = assess_result(runtime, 124, {}, failure_kind='candidate_test_timeout')
            save(output/'hidden-test-hashes.json', {'before': before, 'after_apply': after_apply,
                                                   'after_tests': None, 'unchanged': None})
            return result
        report = '/out/junit.xml' if runtime['task_id'].startswith('r2e_') else '/tmp/validation-result.xml'
        temporary = manager.tmp/('junit-'+uuid.uuid4().hex)
        junit_error = None
        try:
            manager.copy_from(cid, report, temporary)
            report_text = regular_bytes(temporary).decode('utf-8', 'replace')
            (output/'junit.xml').write_text(report_text)
            outcomes = (r2e_junit(report_text) if runtime['task_id'].startswith('r2e_')
                        else parse_junit(report_text, runtime['test_files']))
        except (RuntimeError, ValueError, ET.ParseError, FileNotFoundError) as error:
            junit_error = str(error)
        finally:
            temporary.unlink(missing_ok=True)
        after_tests = snapshot(manager, cid, runtime, hidden)
        unchanged = before == after_apply == after_tests
        save(output/'hidden-test-hashes.json', {'before': before, 'after_apply': after_apply,
                                               'after_tests': after_tests, 'unchanged': unchanged})
        kind = ('candidate_changed_hidden_tests' if not unchanged else
                'candidate_test_timeout' if executed.timed_out else
                'candidate_missing_test_results' if not outcomes else None)
        result = assess_result(runtime, executed.exit_code, outcomes, failure_kind=kind,
            hidden_payload_unchanged=unchanged, changed_paths=changed, patch_sha256=sha(patch),
            junit_error=junit_error, junit=str(output/'junit.xml'), log=str(output/'test.log'),
            image=runtime['image'], base_commit=runtime['resolved_base_commit'],
            manager_proofs=[str(path) for path in (output/'sandbox').glob('sandbox-proof-*.json')])
        if not unchanged:
            result['resolved'] = False
            result['expected_tests_covered'] = False
            result['failure_kind'] = 'candidate_changed_hidden_tests'
        return result
    except Exception as error:
        # Candidate failures are handled explicitly above by return code, timeout,
        # patch parser, or JUnit parser. Exceptions reaching here are control-plane
        # or trusted-artifact failures, even when they occur during test execution.
        result = assess_result(runtime, 1, outcomes, infra_error=True,
            failure_kind='runtime_artifact_or_setup',
            phase=phase, error=repr(error))
        return result
    finally:
        if manager is not None:
            try:
                manager.cleanup_all()
            except Exception as error:
                if result is not None:
                    result.update(infra_error=True, infra_failed=True, resolved=False,
                                  cleanup_error=repr(error))
        if result is not None:
            save(output/'verification.json', result)


def verify_patch(runtime, patch_file, output):
    if not runtime.get('passed'):
        raise ValueError('Task lacks a passing immutable gold/empty runtime calibration')
    return execute_test(runtime, patch_file, output, trusted_control=False)
