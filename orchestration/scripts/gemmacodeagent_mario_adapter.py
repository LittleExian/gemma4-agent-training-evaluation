"""Explicit Mario non-root sandbox adapter; original submission/scheduling stay intact.

Deploy beside calibrate_all_tasks.py and calibrate_all_tasks_v2.py. Replace only
``from autodl_chroot import ChrootManager`` in the upstream runner with this
module's equivalent import. This is NOT the author's Python 3.13 chroot runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from gemmacodeagent_native_guard import install_native_guards


def strict_calibration(gold, empty):
    """Smoke tasks must prove a real negative failure under the current runtime.

    This deliberately does not generalize reference-controlled import/timeout
    exceptions from the full dataset: the selected three tasks need neither.
    """
    import calibrate_all_tasks as calibration
    return calibration.classify_pair(gold, empty)['status'] == 'calibrated'


def require_supported_verification(command):
    """Do not silently lose the origin probe if official pytest syntax changes."""
    if ('--junit' in command and '_swegemma_junit_' in command
            and 'python3 -s -m pytest ' not in command):
        raise RuntimeError('Official verification launcher changed; origin probe compatibility must be reviewed')


def runner_arguments(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--run')
    args, _ = parser.parse_known_args(arguments)
    if not args.ids or len(args.ids) != 1:
        raise ValueError('Mario adapter requires exactly one --ids task per process')
    if not args.run or Path(args.run).is_absolute() or '..' in Path(args.run).parts:
        raise ValueError('Mario adapter requires a relative --run inside repository/runs')
    return args.ids[0], args.run


def validated_profile(gate, task_id, actual_freeze):
    if task_id not in gate.get('eligible_ids', []):
        raise ValueError(f'Task is not in the calibrated eligible set: {task_id}')
    binding = gate.get('task_bindings', {}).get(task_id, {})
    if binding.get('required_profile_ids'):
        raise ValueError('Smoke adapter does not support tasks requiring a profile matrix')
    profile = binding.get('profile', {})
    if not profile.get('environment') or profile.get('actual_freeze') != actual_freeze:
        raise ValueError('Dependency profile is missing or its installed versions changed')
    return dict(profile)


def confined_invocation(command, cwd, env, base, shard, runtime, profile):
    workspace = Path(os.path.abspath(cwd))
    sandbox = workspace.parent
    expected = (shard / 'tmp').resolve()
    if (workspace.is_symlink() or sandbox.is_symlink()
            or sandbox.resolve().parent != expected
            or not sandbox.name.startswith('swegemma_sandbox_')):
        raise RuntimeError(f'Unexpected subprocess sandbox: {workspace}')
    source = workspace / 'src'
    if source.exists() and not source.resolve().is_relative_to(workspace):
        raise RuntimeError('Repository src escapes workspace')
    launcher = [sys.executable, '-I', str(base / 'landlock_guard.py')]
    reads = ['/usr', '/etc', '/opt/anaconda3', '/sys', '/proc/cpuinfo',
             '/proc/meminfo', '/proc/stat', '/proc/sys', '/dev/urandom', '/dev/random',
             str(base / 'runtime'), str(runtime), str(profile),
             str(base / 'landlock_guard.py'), str(base / 'data/wheels'), str(base / 'wheelhouse')]
    for path in dict.fromkeys(reads):
        if Path(path).exists():
            launcher.extend(['--read', path])
    launcher.extend(['--write', str(sandbox), '--write', '/dev/null', '--'])
    allowed = {'PATH', 'LANG', 'LC_ALL', 'VIRTUAL_ENV', 'PIP_NO_INDEX', 'PIP_FIND_LINKS'}
    clean = {key: value for key, value in env.items() if key in allowed}
    roots = [source, workspace] if source.is_dir() else [workspace]
    clean.update(HOME=str(sandbox / 'tmp'), TMPDIR=str(sandbox / 'tmp'),
                 PYTHONPATH=os.pathsep.join(map(str, roots)), PYTHONNOUSERSITE='1')
    shell = ['/bin/bash', '-c', command] if isinstance(command, str) else list(command)
    return launcher + shell, clean


def install_confinement(backend, base, shard, runtime, profile):
    original = backend.execute_subprocess_command

    def confined(command, cwd, env, timeout=240, **kwargs):
        arguments, environment = confined_invocation(command, cwd, env, base, shard, runtime, profile)
        return original(command=arguments, cwd=cwd, env=environment, timeout=timeout,
                        shell=False, max_output_chars=kwargs.get('max_output_chars', 10 * 1024 * 1024))

    backend.execute_subprocess_command = confined


def ChrootManager(timeout_seconds=300, **kwargs):
    """Factory preserving the upstream constructor name, with explicit metadata."""
    import calibrate_all_tasks as calibration
    import calibrate_all_tasks_v2 as calibration_v2
    import swegemma.sandbox.subprocess as backend

    task_id, run = runner_arguments(sys.argv[1:])
    base = Path(os.environ['GEMMA_BASE']).resolve()
    gate_path = Path(os.environ['GEMMA_PREFLIGHT_GATE']).resolve()
    runtime = Path(os.environ['GEMMA_EVAL_RUNTIME']).resolve()
    repo = Path(os.environ['GEMMA_REPO']).resolve()
    shard = repo / 'runs' / run / 'mario-adapter'
    (shard / 'tmp').mkdir(parents=True, exist_ok=True)
    gate = json.loads(gate_path.read_text())
    raw = gate.get('task_bindings', {}).get(task_id, {}).get('profile', {})
    if not raw.get('environment'):
        raise ValueError(f'No dependency profile for {task_id}')
    profile = validated_profile(gate, task_id, calibration.freeze(Path(raw['environment']) / 'bin/python'))
    environment = Path(profile['environment']).resolve()
    if Path(sys.prefix).resolve() != runtime:
        raise ValueError('Runner interpreter is not the configured independent eval runtime')
    tasks = [json.loads(line) for line in (base / 'data/tasks.jsonl').read_text().splitlines() if line.strip()]
    matching = [task for task in tasks if task['instance_id'] == task_id]
    if len(matching) != 1:
        raise ValueError('Expected one matching task in the verified dataset')
    if calibration.sha(base / 'data/tasks.jsonl') != gate['dataset_sha256']:
        raise ValueError('Task dataset differs from the calibration gate')
    transcript = []
    install_confinement(backend, base, shard, runtime, environment)
    manager = calibration_v2.with_declared_async_backend(
        backend.SubprocessManager, shard, environment, transcript,
        matching[0]['repo'].rsplit('/', 1)[-1])
    manager.timeout_seconds = timeout_seconds
    manifest = {'environment': 'mario-landlock-profile-adaptation-not-upstream-chroot',
                'task_id': task_id, 'profile': profile, 'eval_runtime': str(runtime),
                'gate_sha256': calibration.sha(gate_path),
                'adapter_sha256': calibration.sha(__file__),
                'native_guard_sha256': calibration.sha(Path(__file__).with_name('gemmacodeagent_native_guard.py')),
                'calibration_sources': {Path(module.__file__).name: calibration.sha(module.__file__)
                                        for module in (calibration, calibration_v2)},
                'upstream_constructor_kwargs': kwargs}
    calibration.atomic_json(shard / 'manifest.json', manifest)
    execute = manager.exec

    def logged_exec(identifier, command, **options):
        try:
            require_supported_verification(command)
            return execute(identifier, command, **options)
        finally:
            calibration.atomic_json(shard / 'commands.json', transcript)

    manager.exec = logged_exec
    install_native_guards(manager, shard)
    return manager
