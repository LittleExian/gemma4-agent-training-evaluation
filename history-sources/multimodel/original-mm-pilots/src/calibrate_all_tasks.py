"""Calibrate every dataset task using gold and empty patches in fresh CPU sandboxes.

Run with the existing harness interpreter. Each verification uses 240 seconds for
its test command, an isolated dependency environment and no model service.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_shard(tasks, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid shard index/count')
    return [task for position, task in enumerate(tasks) if position % count == index]


def task_directory(root, instance_id):
    return Path(root) / hashlib.sha256(instance_id.encode()).hexdigest()


def checkpoint_valid(path, fingerprint, instance_id):
    if not Path(path).exists():
        return False
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    required = {'instance_id', 'fingerprint', 'resolved', 'test_output', 'test_exit_code'}
    return (required.issubset(value) and value['fingerprint'] == fingerprint
            and value['instance_id'] == instance_id)


def pytest_counts(text):
    counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}
    summaries = []
    for line in text.splitlines():
        matches = re.findall(r'\b(\d+) (passed|failed|skipped|errors?)\b', line)
        if matches:
            summaries.append(matches)
    if summaries:
        for number, label in summaries[-1]:
            counts['errors' if label.startswith('error') else label] = int(number)
    return counts


def resolve_profile(mapping, instance_id):
    if mapping.get('schema_version') != 1:
        raise ValueError('Unsupported profile map schema')
    profile_id = mapping.get('tasks', {}).get(instance_id)
    profile = mapping.get('profiles', {}).get(profile_id)
    if not profile or not profile.get('environment') or not isinstance(profile.get('requirements'), list):
        raise ValueError(f'Missing or invalid dependency profile for {instance_id}')
    return {**profile, 'profile_id': profile_id}


def diagnose(record):
    text = '\n'.join(str(record.get(key) or '') for key in ('test_output', 'error', 'error_message', 'exception'))
    lower = text.lower()
    patterns = {
        'collection_error': ('error collecting', 'errors during collection', 'error during collection', 'collectionerror', 'importerror while loading conftest'),
        'missing_dependency_or_import': ('modulenotfounderror', 'importerror', 'no module named'),
        'permission_error': ('permission denied', 'permissionerror', 'landlock guard failed', 'unexpected subprocess sandbox'),
        'verification_timeout': ('timed out', 'timeout expired', 'timeoutexpired', 'command timed out'),
        'patch_application_error': ('failed to apply', 'patch does not apply', 'patch file missing'),
        'test_runner_error': ('no tests ran', 'no tests collected', 'collected 0 items', 'pytest: command not found', 'missing test specification', 'missing or empty junit'),
        'import_origin_error': ('target package import origin',),
        'setup_error': ('containersetuperror', 'failed to set up', 'snapshot file not found', 'no space left on device'),
    }
    categories = [key for key, needles in patterns.items() if any(needle in lower for needle in needles)]
    missing = sorted(set(re.findall(r"No module named [\"']?([\w.]+)", text)))
    counts = pytest_counts(str(record.get('test_output') or ''))
    real_failure = record.get('test_exit_code') == 1 and counts['failed'] > 0 and not counts['errors'] and not categories
    return {'categories': categories, 'missing_modules': missing, 'real_test_failure': real_failure, 'pytest_counts': counts,
            'interpretation': 'Import/missing-module signals are conservative environment candidates; an intended code change can also cause them.'}


def classify_pair(gold, empty):
    gold_diagnostic, empty_diagnostic = diagnose(gold), diagnose(empty)
    gold_pass = gold.get('resolved') is True and gold.get('test_exit_code') == 0 and gold_diagnostic['pytest_counts']['passed'] > 0 and not gold_diagnostic['categories']
    empty_pass = empty.get('resolved') is True and empty.get('test_exit_code') == 0 and empty_diagnostic['pytest_counts']['passed'] > 0 and not empty_diagnostic['categories']
    status, reason = 'uncalibrated', 'gold_did_not_pass'
    if gold_pass and empty_pass:
        status, reason = 'nondiscriminating', 'empty_patch_also_passes'
    elif gold_pass and empty_diagnostic['real_test_failure']:
        status, reason = 'calibrated', 'gold_passes_and_empty_has_real_test_failure'
    elif gold_pass:
        reason = 'empty_verification_did_not_prove_real_test_failure'
    return {'status': status, 'reason': reason, 'gold': gold_diagnostic, 'empty': empty_diagnostic}


def summarize(records, dataset_count):
    return {'dataset_count': dataset_count, 'completed': len(records),
            'pending': dataset_count - len(records),
            **{name: sum(record['status'] == name for record in records)
               for name in ('calibrated', 'nondiscriminating', 'uncalibrated')}}


def install_confinement(base, shard, test_envs):
    import swegemma.sandbox.subprocess as backend
    original = backend.execute_subprocess_command
    expected = (shard / 'tmp').resolve()

    def confined(command, cwd, env, timeout=240, **kwargs):
        workspace = Path(os.path.abspath(cwd))
        sandbox = workspace.parent
        if (workspace.is_symlink() or sandbox.is_symlink() or sandbox.resolve().parent != expected
                or not sandbox.name.startswith('swegemma_sandbox_')):
            raise RuntimeError(f'Unexpected subprocess sandbox: {workspace}')
        source = workspace / 'src'
        if source.exists() and not source.resolve().is_relative_to(workspace):
            raise RuntimeError('Repository src escapes workspace')
        launcher = [sys.executable, '-I', str(base / 'landlock_guard.py')]
        read_paths = ['/usr', '/etc', '/opt/anaconda3', '/sys', '/proc/cpuinfo', '/proc/meminfo',
                      '/proc/stat', '/proc/sys', '/dev/urandom', '/dev/random', *map(str, test_envs),
                      str(base / 'runtime'), str(base / 'landlock_guard.py'),
                      str(base / 'data/wheels'), str(base / 'wheelhouse')]
        for path in read_paths:
            if Path(path).exists():
                launcher.extend(['--read', path])
        launcher.extend(['--write', str(sandbox), '--write', '/dev/null', '--'])
        allowed = {'PATH', 'LANG', 'LC_ALL', 'VIRTUAL_ENV', 'PIP_NO_INDEX', 'PIP_FIND_LINKS'}
        clean = {key: value for key, value in env.items() if key in allowed}
        roots = [source, workspace] if source.is_dir() else [workspace]
        clean.update(HOME=str(sandbox / 'tmp'), TMPDIR=str(sandbox / 'tmp'),
                     PYTHONPATH=os.pathsep.join(map(str, roots)), PYTHONNOUSERSITE='1')
        shell = ['/bin/bash', '-c', command] if isinstance(command, str) else list(command)
        return original(command=launcher + shell, cwd=workspace, env=clean, timeout=timeout,
                        shell=False, max_output_chars=kwargs.get('max_output_chars', 10 * 1024 * 1024))

    backend.execute_subprocess_command = confined


def environment_probe_code(target_package):
    dependencies = ['pytest', 'pydantic', 'starlette', 'Pygments', 'inline-snapshot', 'pytest-httpbin']
    return ('import importlib.util, importlib.metadata, json, site, sys\n'
            'versions = {}\n'
            f'for name in {dependencies!r}:\n'
            '    try: versions[name] = importlib.metadata.version(name)\n'
            '    except importlib.metadata.PackageNotFoundError: versions[name] = None\n'
            f'spec = importlib.util.find_spec({target_package!r})\n'
            f'print(json.dumps({{"package": {target_package!r}, "origin": spec.origin if spec else None, '
            '"executable": sys.executable, "python_version": sys.version, '
            '"site_packages": site.getsitepackages(), "sys_path": sys.path, "dependencies": versions}))')


def isolated_manager(manager_type, shard, test_env, transcript, target_package):
    manager = manager_type(timeout_seconds=240, base_dir=shard / 'tmp', system_site_packages=False)
    original_start, original_exec = manager.start, manager.exec
    sites = list(test_env.glob('lib/python*/site-packages'))
    if len(sites) != 1:
        raise ValueError('Expected exactly one test environment site-packages directory')

    def start(*args, **kwargs):
        identifier = original_start(*args, **kwargs)
        sandbox_venv = manager.sandboxes[identifier]['venv']
        sandbox_sites = list(sandbox_venv.glob('lib/python*/site-packages'))
        if len(sandbox_sites) == 1 and sandbox_sites[0].parent.name != sites[0].parent.name:
            subprocess.run([str(test_env / 'bin/python'), '-m', 'venv', '--clear', str(sandbox_venv)],
                           check=True, capture_output=True, text=True, timeout=120)
            sandbox_sites = list(sandbox_venv.glob('lib/python*/site-packages'))
        if len(sandbox_sites) != 1 or sandbox_sites[0].parent.name != sites[0].parent.name:
            raise ValueError('Expected one isolated sandbox site-packages directory matching profile Python')
        (sandbox_sites[0] / '_preflight_test_env.pth').write_text(str(sites[0]) + '\n')
        return identifier

    def execute(identifier, command, **kwargs):
        if 'python3 -s -m pytest ' in command:
            workspace = manager.sandboxes[identifier]['workspace'].resolve()
            probe_code = environment_probe_code(target_package)
            probe = original_exec(identifier, 'PYTHONSAFEPATH=1 python3 -s -c ' + shlex.quote(probe_code), timeout=30)
            transcript.append({'phase': 'target_package_origin', 'exit_code': probe.exit_code,
                               'stdout': probe.stdout, 'stderr': probe.stderr})
            try:
                probe_data = json.loads(probe.stdout.strip())
                transcript[-1]['environment'] = probe_data
                origin = probe_data['origin']
                valid = probe.exit_code == 0 and origin and Path(origin).resolve().is_relative_to(workspace)
            except (ValueError, TypeError, KeyError):
                valid = False
            if not valid:
                raise RuntimeError(f'Target package import origin outside workspace: {probe.stdout} {probe.stderr}')
        result = original_exec(identifier, command, **kwargs)
        transcript.append({'command': command, 'exit_code': result.exit_code,
                           'stdout': result.stdout, 'stderr': result.stderr})
        return result

    manager.start, manager.exec = start, execute
    return manager


def freeze(interpreter):
    return subprocess.check_output([str(interpreter), '-m', 'pip', 'freeze', '--all'], text=True, timeout=60)


def fingerprint_inputs(base):
    import swegemma.harness.verification as verification
    import swegemma.harness.container_setup as setup
    import swegemma.sandbox.subprocess as backend
    return {'dataset_sha256': sha(base / 'data/tasks.jsonl'),
            'dataset_manifest_sha256': sha(base / 'data/manifest.json'),
            'runner_sha256': sha(__file__), 'landlock_sha256': sha(base / 'landlock_guard.py'),
            'harness_sources': {str(Path(module.__file__).name): sha(module.__file__)
                                for module in (verification, setup, backend)},
            'runtime_freeze': freeze(sys.executable),
            'command_timeout_seconds': 240, 'system_site_packages': False}


async def verify_one(task, variant, base, shard, profile, fingerprint, config):
    from swegemma.harness.verification import verify_task
    from swegemma.sandbox.subprocess import SubprocessManager
    transcript = []
    manager = isolated_manager(SubprocessManager, shard, Path(profile['environment']), transcript, task.repo.rsplit('/', 1)[-1])
    started = time.perf_counter()
    try:
        result = await verify_task(manager, config, task, base / 'data/snapshots' / f'{task.instance_id}.tgz',
                                   agent_patch=task.patch if variant == 'gold' else '', start_time=started)
        data = result.model_dump(mode='json', exclude={'agent_patch', 'trace'})
    except Exception as exc:
        data = {'resolved': False, 'test_exit_code': None, 'test_output': '',
                'exception': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
    finally:
        manager.cleanup_all()
    return {**data, 'instance_id': task.instance_id, 'repo': task.repo, 'variant': variant,
            'fingerprint': fingerprint, 'dependency_profile': profile, 'saved_at': stamp(), 'commands': transcript,
            'pytest_counts': pytest_counts(str(data.get('test_output') or '')),
            'environment_probes': [entry['environment'] for entry in transcript if 'environment' in entry],
            'duration_seconds': time.perf_counter() - started}


async def run_tasks(args):
    base, root = (Path(value).resolve() for value in (args.base, args.run_dir))
    shard = root / f'generation-{args.generation}' / f'shard-{args.shard_index}'
    for name in ('tmp', 'home', 'results', 'harness-results', 'agent'):
        (shard / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(HOME=str(shard / 'home'), TMPDIR=str(shard / 'tmp'),
                      PYTHONNOUSERSITE='1', OTEL_SDK_DISABLED='true',
                      LITELLM_LOCAL_MODEL_COST_MAP='True')
    tempfile.tempdir = str(shard / 'tmp')
    import fcntl
    from swegemma.config import EvalConfig
    from swegemma.models.task import load_tasks
    lock = (shard / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = load_tasks(base / 'data/tasks.jsonl')
    by_id = {task.instance_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError('Duplicate dataset identifiers')
    selected = tasks
    if args.task_ids:
        unknown = set(args.task_ids) - set(by_id)
        if unknown:
            raise ValueError(f'Unknown task IDs: {sorted(unknown)}')
        selected = [task for task in tasks if task.instance_id in args.task_ids]
    assigned = select_shard(selected, args.shard_index, args.shard_count)
    mapping = json.loads(Path(args.profile_map).read_text()) if args.profile_map else {
        'schema_version': 1, 'profiles': {'default': {'environment': str(Path(args.test_env).resolve()), 'requirements': []}},
        'tasks': {task.instance_id: 'default' for task in tasks}}
    profiles = {task.instance_id: resolve_profile(mapping, task.instance_id) for task in assigned}
    freeze_cache = {}
    for task_id, profile in profiles.items():
        environment = Path(profile['environment']).resolve()
        if environment not in freeze_cache:
            freeze_cache[environment] = freeze(environment / 'bin/python')
        interpreter_version = subprocess.check_output([str(environment / 'bin/python'), '--version'], text=True, timeout=30).strip()
        site_versions = sorted(path.parent.name for path in environment.glob('lib/python*/site-packages'))
        profiles[task_id] = {**profile, 'environment': str(environment), 'actual_freeze': freeze_cache[environment],
                             'interpreter_version': interpreter_version, 'site_versions': site_versions}
    inputs = fingerprint_inputs(base)
    inputs.update(profile_map_sha256=sha(args.profile_map) if args.profile_map else None,
                  dependency_profiles=profiles)
    inputs.update(shard_index=args.shard_index, shard_count=args.shard_count,
                  assigned_ids=[task.instance_id for task in assigned])
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    state_path = shard / 'summary.json'
    if state_path.exists() and json.loads(state_path.read_text())['fingerprint'] != digest:
        raise ValueError('Fingerprint changed; use a new generation instead of mixing environments')
    state = {'status': 'running', 'fingerprint': digest, 'fingerprint_inputs': inputs,
             'started_at': stamp(), 'assigned_count': len(assigned), 'dataset_count': len(tasks)}
    config = EvalConfig(tasks_path=base / 'data/tasks.jsonl', snapshots_dir=base / 'data/snapshots',
                        results_dir=shard / 'harness-results', submission_dir=shard / 'agent',
                        models=None, sandbox='subprocess', wheels_dir=base / 'data/wheels', timeout_seconds=240)
    if config.harness.command_timeout_seconds != 240:
        raise ValueError(f'Unexpected verification timeout: {config.harness.command_timeout_seconds}')
    install_confinement(base, shard, freeze_cache)
    pairs = []
    atomic_json(state_path, state)
    for task in assigned:
        directory = task_directory(shard / 'results', task.instance_id)
        snapshot_sha = sha(base / 'data/snapshots' / f'{task.instance_id}.tgz')
        task_fp = hashlib.sha256((digest + snapshot_sha).encode()).hexdigest()
        state.update(current_task=task.instance_id, updated_at=stamp())
        atomic_json(state_path, state)
        variants = {}
        for variant in ('gold', 'empty'):
            path = directory / f'{variant}.json'
            if checkpoint_valid(path, task_fp, task.instance_id):
                variants[variant] = json.loads(path.read_text())
                print(f'CACHED {task.instance_id} {variant}', flush=True)
                continue
            if path.exists():
                raise ValueError(f'Invalid or stale checkpoint: {path}')
            print(f'START {task.instance_id} {variant}', flush=True)
            result = await verify_one(task, variant, base, shard, profiles[task.instance_id], task_fp, config)
            variants[variant] = result
            atomic_json(path, result)
            print(f'DONE {task.instance_id} {variant} resolved={result.get("resolved")} exit={result.get("test_exit_code")}', flush=True)
        pair = {'instance_id': task.instance_id, 'repo': task.repo, 'fingerprint': task_fp,
                'snapshot_sha256': snapshot_sha, 'dependency_profile': profiles[task.instance_id], **classify_pair(variants['gold'], variants['empty'])}
        atomic_json(directory / 'calibration.json', pair)
        pairs.append(pair)
        state.update(**summarize(pairs, len(tasks)), assigned_pending=len(assigned) - len(pairs), updated_at=stamp())
        atomic_json(state_path, state)
        print(f'PAIR {task.instance_id} {pair["status"]} completed={len(pairs)}/{len(assigned)}', flush=True)
    state.update(status='completed', finished_at=stamp(), current_task=None)
    atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('base', 'run-dir'):
        parser.add_argument('--' + option, required=True)
    environment = parser.add_mutually_exclusive_group(required=True)
    environment.add_argument('--test-env')
    environment.add_argument('--profile-map')
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, default=8)
    parser.add_argument('--task-ids', nargs='+')
    parser.add_argument('--generation', type=int, default=1)
    args = parser.parse_args()
    select_shard([], args.shard_index, args.shard_count)
    if args.generation < 1:
        parser.error('generation must be positive')
    asyncio.run(run_tasks(args))


if __name__ == '__main__':
    main()
