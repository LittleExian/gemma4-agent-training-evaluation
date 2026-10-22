"""Run one resumable shard of the unchanged budget submission on a single GPU."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def select_shard(tasks, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Require shard-count > 0 and 0 <= shard-index < shard-count')
    return [task for position, task in enumerate(tasks) if position % count == index]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def result_path(root, instance_id):
    # Dataset identifiers are external input; hash names cannot traverse directories.
    return Path(root) / (hashlib.sha256(instance_id.encode()).hexdigest() + '.json')


def has_result(root, instance_id):
    path = result_path(root, instance_id)
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    required = {'instance_id', 'status', 'resolved', 'agent_patch', 'test_output',
                'test_exit_code', 'duration_seconds'}
    if (not required.issubset(value) or value.get('instance_id') != instance_id
            or value.get('status') not in {'evaluated', 'infrastructure_error'}):
        raise ValueError(f'Invalid checkpoint: {path}')
    return True


def save_result(root, record):
    path = result_path(root, record['instance_id'])
    if path.exists():
        raise FileExistsError(f'Existing task result must not be overwritten: {path}')
    atomic_json(path, record)


def classify_result(result):
    error = str(result.get('error_message') or result.get('error') or '')
    budget_errors = ('time budget', 'timed out', 'timeout', 'tool call limit', 'max_tool_calls',
                     'max turns', 'max_llm_calls', 'budget exceeded', 'maximum number of',
                     'max turn', 'llm call limit')
    infrastructure = ('snapshot file not found', 'permission denied', 'connection refused',
                      'connection error', 'apierror', 'out of memory', 'no space left',
                      'failed to install',
                      'failed to set up', 'unexpected subprocess', 'authenticationerror',
                      'missing test specification', 'failed to apply test_patch', 'evaluation error:')
    lower = error.lower()
    diagnostic = verification_diagnostic(result)
    if diagnostic['definite_infrastructure']:
        return 'infrastructure_error'
    if any(token in lower for token in infrastructure):
        return 'infrastructure_error'
    if error and any(token in lower for token in budget_errors):
        return 'evaluated'
    if error and (result.get('test_exit_code') is None or result.get('test_exit_code', -1) < 0):
        return 'infrastructure_error'
    return 'evaluated'


def verification_diagnostic(result):
    """Flag environment candidates without excusing imports broken by an agent patch."""
    text = str(result.get('test_output') or '').lower()
    patterns = {
        'collection_error': ('error collecting', 'errors during collection', 'interrupted: 1 error'),
        'import_or_missing_dependency': ('modulenotfounderror', 'importerror', 'no module named'),
        'permission_error': ('permission denied', 'permissionerror', 'landlock'),
        'verification_timeout': ('timed out', 'timeout expired', 'timeoutexpired'),
    }
    categories = [name for name, tokens in patterns.items() if any(token in text for token in tokens)]
    definite = any(token in text for token in [
        'no module named pytest', "no module named 'pytest'", 'pytest: command not found',
        'landlock guard failed', 'unexpected subprocess sandbox',
        'permission denied', 'permissionerror'])
    return {'categories': categories, 'suspected_environment': bool(categories),
            'definite_infrastructure': definite,
            'interpretation': 'Candidate only: agent changes can also cause import or collection errors.'}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resume_state(current, previous):
    if previous.get('fingerprint') != current['fingerprint']:
        raise ValueError('Resume configuration fingerprint changed')
    # Preserve original GPU/start metadata even when every task is already done.
    merged = {**previous, **current}
    merged['started_at'] = previous.get('started_at', current.get('started_at'))
    return merged


def result_counts(records, total_dataset):
    return {'dataset_count': total_dataset, 'record_count': len(records),
            'evaluated': sum(r['status'] == 'evaluated' for r in records),
            'infrastructure_errors': sum(r['status'] == 'infrastructure_error' for r in records),
            'resolved': sum(r['status'] == 'evaluated' and r.get('resolved') is True for r in records),
            'suspected_environment_verifications': sum(
                bool(r.get('verification_diagnostic', {}).get('suspected_environment')) for r in records)}


def stop_owned_server(server, shard, process_api=None):
    """Capture the server's descendants, preserving PID identity before parent stop."""
    if process_api is None:
        import psutil as process_api
    report = {'captured_child_pids': [], 'terminated': [], 'killed': [], 'notes': []}
    children = []
    process = server.process
    if process is not None:
        try:
            descendants = process_api.Process(process.pid).children(recursive=True)
            children = [(child, child.create_time()) for child in descendants]
            report['captured_child_pids'] = [child.pid for child, _ in children]
        except Exception as exc:
            report['notes'].append(f'Capture children: {type(exc).__name__}: {exc}')
    try:
        source = Path(server.log_path)
        if source.exists():
            shutil.copy2(source, shard / 'server.log')
    finally:
        server.stop()
    if not children:
        return report
    process_api.wait_procs([child for child, _ in children], timeout=2)
    for operation in ('terminate', 'kill'):
        acted = []
        for child, created in children:
            try:
                if child.is_running() and child.create_time() == created:
                    getattr(child, operation)()
                    acted.append(child)
                    report['terminated' if operation == 'terminate' else 'killed'].append(child.pid)
            except Exception as exc:
                report['notes'].append(f'{child.pid}: {type(exc).__name__}: {exc}')
        if acted:
            process_api.wait_procs(acted, timeout=2)
    return report


def workspace_pythonpath(workspace):
    workspace = Path(workspace)
    source = workspace / 'src'
    if source.exists() or source.is_symlink():
        if not source.resolve().is_relative_to(workspace.resolve()):
            raise ValueError('Repository src path escapes its workspace')
    roots = [source, workspace] if source.is_dir() else [workspace]
    return os.pathsep.join(str(path) for path in roots)


def install_confinement(base, shard):
    import swegemma.sandbox.subprocess as backend
    original = backend.execute_subprocess_command
    expected = (shard / 'tmp').resolve()

    def confined(command, cwd, env, timeout=300, **kwargs):
        workspace = Path(os.path.abspath(cwd))
        sandbox = workspace.parent
        if (workspace.is_symlink() or sandbox.is_symlink()
                or sandbox.resolve().parent != expected
                or not sandbox.name.startswith('swegemma_sandbox_')):
            raise RuntimeError(f'Unexpected subprocess sandbox: {workspace}')
        launcher = [sys.executable, '-I', str(base / 'landlock_guard.py')]
        read_paths = ['/usr', '/etc', '/opt/anaconda3', '/sys', '/proc/cpuinfo',
                      '/proc/meminfo', '/proc/stat', '/proc/sys', '/dev/urandom', '/dev/random',
                      str(base / 'runtime'), str(base / 'landlock_guard.py'),
                      str(base / 'data/wheels'), str(base / 'wheelhouse')]
        for path in read_paths:
            if Path(path).exists():
                launcher.extend(['--read', path])
        launcher.extend(['--write', str(sandbox), '--write', '/dev/null', '--'])
        shell_command = ['/bin/bash', '-c', command] if isinstance(command, str) else list(command)
        allowed = {'PATH', 'LANG', 'LC_ALL', 'VIRTUAL_ENV', 'PYTHONPATH', 'TMPDIR',
                   'PIP_NO_INDEX', 'PIP_FIND_LINKS', 'PYTHONNOUSERSITE'}
        clean_env = {key: value for key, value in env.items() if key in allowed}
        clean_env.update(HOME=str(sandbox / 'tmp'), TMPDIR=str(sandbox / 'tmp'),
                         PYTHONPATH=workspace_pythonpath(workspace), PYTHONNOUSERSITE='1')
        return original(command=launcher + shell_command, cwd=workspace, env=clean_env,
                        timeout=timeout, shell=False,
                        max_output_chars=kwargs.get('max_output_chars', 10 * 1024 * 1024))

    backend.execute_subprocess_command = confined


def unpack_submission(archive, directory):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for name in names:
            target = directory / name
            if not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError(f'Unsafe ZIP member: {name}')
            content = bundle.read(name)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f'Existing agent differs from submitted ZIP: {name}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    return names


def configure_environment(shard, base):
    for name in ('tmp', 'home', 'results', 'agent'):
        (shard / name).mkdir(parents=True, exist_ok=True)
    values = {'HOME': str(shard / 'home'), 'TMPDIR': str(shard / 'tmp'),
              'HF_HOME': str(base / 'hf-cache'), 'HF_HUB_OFFLINE': '1',
              'TRANSFORMERS_OFFLINE': '1', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
              'TRANSFORMERS_NO_TF': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
              'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_NO_USAGE_STATS': '1',
              'OTEL_SDK_DISABLED': 'true', 'PYTHONNOUSERSITE': '1',
              'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
    os.environ.update(values)
    tempfile.tempdir = str(shard / 'tmp')


async def evaluate_all(evaluator, pending, output, state, state_path):
    for position, task in enumerate(pending, 1):
        state.update(status='running', current_task=task.instance_id, updated_at=stamp())
        atomic_json(state_path, state)
        print(f'TASK START {task.instance_id}', flush=True)
        started = time.monotonic()
        try:
            result = await evaluator.evaluate_task(task=task, task_index=position, total_tasks=len(pending))
            data = result.model_dump(mode='python', exclude={'trace'})
            trace = getattr(result, 'trace', None)
            if trace is not None:
                data['trace'] = trace.model_dump(mode='python') if hasattr(trace, 'model_dump') else str(trace)
            record = {**data, 'instance_id': task.instance_id, 'status': classify_result(data),
                      'verification_diagnostic': verification_diagnostic(data), 'saved_at': stamp()}
        except Exception as exc:
            record = {'instance_id': task.instance_id, 'repo': task.repo,
                      'status': 'infrastructure_error', 'resolved': None,
                      'error_message': f'{type(exc).__name__}: {exc}',
                      'exception_traceback': traceback.format_exc(), 'agent_patch': '',
                      'test_output': '', 'test_exit_code': None,
                      'duration_seconds': time.monotonic() - started, 'saved_at': stamp()}
            record['verification_diagnostic'] = verification_diagnostic(record)
        save_result(output, record)
        state.update(finished_this_invocation=position, last_task=task.instance_id)
        atomic_json(state_path, state)
        print(f'TASK DONE {task.instance_id} status={record["status"]} resolved={record.get("resolved")}', flush=True)


def run(args):
    base, run_dir = Path(args.base).resolve(), Path(args.run_dir).resolve()
    shard = run_dir / f'shard-{args.shard_index}'
    configure_environment(shard, base)
    import fcntl
    import litellm
    import torch
    import yaml
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models.discovery import validate_single_declared_model
    from swegemma.models.task import load_tasks

    lock = (shard / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = load_tasks(base / 'data/tasks.jsonl')
    assigned = select_shard(tasks, args.shard_index, args.shard_count)
    if len({t.instance_id for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task IDs')
    archive = Path(args.submission or run_dir / 'submission.zip')
    names = unpack_submission(archive, shard / 'agent')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    state_path = shard / 'summary.json'
    cache_config = {'min_tokens': 2048, 'ttl_seconds': 1800, 'cache_intervals': 10}
    compaction_config = {'compaction_interval': 15, 'overlap_size': 2,
                         'token_threshold': 14336, 'event_retention_size': 5}
    raw = yaml.safe_load((shard / 'agent/eval_config.yaml').read_text())
    section = raw.get('evaluation', raw)
    fingerprint = {'submission_sha256': digest,
                   'tasks_sha256': file_sha256(base / 'data/tasks.jsonl'),
                   'model_config_sha256': file_sha256(base / 'model/config.json'),
                   'runtime_freeze_sha256': file_sha256(run_dir / 'runtime-freeze.txt'),
                   'context_cache_config': cache_config, 'events_compaction_config': compaction_config,
                   'gpu_memory_fraction': args.gpu_memory_fraction, 'evaluation_config': section,
                   'shard_index': args.shard_index, 'shard_count': args.shard_count,
                   'assigned_ids': [t.instance_id for t in assigned]}
    state = {'status': 'started', 'started_at': stamp(), 'submission_sha256': digest,
             'fingerprint': fingerprint, 'evaluation_config': section,
             'submission_members': names, 'shard_index': args.shard_index, 'shard_count': args.shard_count,
             'dataset_count': len(tasks), 'assigned_ids': [t.instance_id for t in assigned],
             'context_cache_config': cache_config, 'events_compaction_config': compaction_config,
             'gpu_memory_fraction': args.gpu_memory_fraction,
             'versions': {p: importlib.metadata.version(p) for p in
                          ['torch', 'vllm', 'swegemma', 'adk-submission', 'google-adk', 'transformers']}}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        state = resume_state(state, previous)
    output = shard / 'task-results'
    pending = [t for t in assigned if not has_result(output, t.instance_id)]
    state['pending_at_start'] = len(pending)
    atomic_json(state_path, state)
    if not pending:
        records = [json.loads(result_path(output, t.instance_id).read_text()) for t in assigned]
        state.update(status='completed', finished_at=stamp(), current_task=None,
                     **result_counts(records, len(tasks)))
        atomic_json(state_path, state)
        return
    server = None
    try:
        assert torch.cuda.device_count() == 1, 'Launch each shard with one visible GPU'
        state['gpu_uuid'] = str(torch.cuda.get_device_properties(0).uuid)
        state['gpu_name'] = torch.cuda.get_device_name(0)
        atomic_json(state_path, state)
        litellm.drop_params = True
        agent = shard / 'agent'
        declared = validate_single_declared_model(agent)
        adapters = discover_adapters(str(agent), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        if getattr(adapters, 'adapters', None):
            raise ValueError('Expected unchanged budget submission without LoRA')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cfg = VllmConfig(model=str(base / 'model'), host='127.0.0.1', port=port,
                         tool_call_parser='gemma4', reasoning_parser='gemma4',
                         max_model_len=32768, dtype='bfloat16',
                         gpu_memory_utilization=args.gpu_memory_fraction,
                         enable_auto_tool_choice=True, enable_lora=False,
                         tensor_parallel_size=1, startup_timeout=1200,
                         extra_args=['--max-num-seqs', '1', '--limit-mm-per-prompt',
                                     '{"image":0,"audio":0,"video":0}', '--enforce-eager'])
        server = VllmServer(cfg, adapter_manifest=adapters)
        state.update(server_log=str(server.log_path), port=port)
        atomic_json(state_path, state)
        server.start()
        state['server_pid'] = server.process.pid
        atomic_json(state_path, state)
        models = server.create_model_registry(aliases=[declared, 'gemma-4-31b-it-qat-w4a16-ct'],
                                               model_prefix='openai/', api_key='EMPTY')
        limits, constraints = build_submission_limits()
        config = EvalConfig(tasks_path=base / 'data/tasks.jsonl', snapshots_dir=base / 'data/snapshots',
                            results_dir=shard / 'results', submission_dir=agent, models=models,
                            sandbox='subprocess', timeout_seconds=int(section['timeout_seconds']),
                            max_time_minutes=float(section['max_time_minutes']),
                            max_tool_calls=int(section['max_tool_calls']), max_turns=int(section['max_turns']),
                            limits=limits, generation_constraints=constraints, adapter_manifest=adapters,
                            context_cache_config=ContextCacheConfig(**cache_config),
                            events_compaction_config=EventsCompactionConfig(**compaction_config),
                            graph_dir=str(base / 'data/graphs'), embeddings_dir=str(base / 'data/embeddings'),
                            wheels_dir=base / 'data/wheels', verbose=False)
        install_confinement(base, shard)
        evaluator = Evaluator(config)
        evaluator.sandbox.base_dir = shard / 'tmp'
        asyncio.run(evaluate_all(evaluator, pending, output, state, state_path))
        records = [json.loads(result_path(output, t.instance_id).read_text()) for t in assigned]
        state.update(status='completed', finished_at=stamp(), current_task=None,
                     **result_counts(records, len(tasks)))
    except BaseException as exc:
        state.update(status='failed', finished_at=stamp(), error=f'{type(exc).__name__}: {exc}',
                     exception_traceback=traceback.format_exc())
        raise
    finally:
        try:
            if server is not None:
                state['cleanup'] = stop_owned_server(server, shard)
                state['cleanup']['gpu_uuid'] = state.get('gpu_uuid')
        finally:
            atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--submission')
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.65)
    args = parser.parse_args()
    select_shard([], args.shard_index, args.shard_count)
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('gpu-memory-fraction must be in (0, 1]')
    run(args)


if __name__ == '__main__':
    main()
