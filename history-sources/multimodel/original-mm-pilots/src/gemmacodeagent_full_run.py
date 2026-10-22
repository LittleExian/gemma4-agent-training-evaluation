"""Rolling, checkpointed full evaluation; model services are managed separately."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from finalize_preflight import atomic_json, classify_pair, classify_matrix, counts, junit_outcomes, node_matches, required_test_names


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


EXECUTION_DEPENDENCIES = ('gemmacodeagent_full_adapter.py', 'gemmacodeagent_native_guard.py',
                          'gemmacodeagent_mario_adapter.py', 'calibrate_all_tasks.py',
                          'calibrate_all_tasks_v2.py', 'finalize_preflight.py',
                          'run_eval_autodl_matched56.py', 'global_eval_slots.py')


def execution_sources(task_runner):
    return {name: digest(task_runner.parent / name) for name in EXECUTION_DEPENDENCIES}


def endpoint_slots(endpoints, workers):
    if (len(endpoints) != len(workers) or not endpoints or len(set(endpoints)) != len(endpoints)
            or any(n < 1 or n > 8 for n in workers)):
        raise ValueError('Each unique endpoint requires a worker count between 1 and 8')
    return [endpoint for endpoint, number in zip(endpoints, workers) for _ in range(number)]


def profiles_for(gate, task_id):
    binding = gate['task_bindings'][task_id]
    profiles = binding.get('required_profile_ids')
    if profiles:
        return profiles
    profile_id = (binding.get('profile') or {}).get('profile_id')
    if not profile_id:
        raise ValueError('Missing bound dependency profile: ' + task_id)
    return [profile_id]


def normal_output(record):
    verification = record.get('verification') or record
    exit_code = verification.get('test_exit_code', verification.get('exit_code'))
    budget_timeout = (record.get('agent_timed_out') is True
                      and record.get('error_code') == 'agent_budget_timeout')
    return (not record.get('infra_error')
            and (not record.get('error') or budget_timeout)
            and (exit_code in (0, 1) or (budget_timeout and exit_code in (None, -1, 124))))


def bind_run(root, manifest):
    path = root / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Resume fingerprint changed; use a new run directory')
    else:
        atomic_json(path, manifest)


def result_path(root, identifier):
    return root / 'results' / (hashlib.sha256(identifier.encode()).hexdigest() + '.json')


def commit_result(root, identifier, record):
    path = result_path(root, identifier)
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Refusing to replace a committed task result')
    else:
        atomic_json(path, record)


def control_verdict(task, pairs):
    if len(pairs) == 1:
        return classify_pair(task, pairs[0]['gold'], pairs[0]['empty'])
    return classify_matrix(task, pairs)


def score_patch(task, records):
    if len(records) == 1:
        return records[0].get('resolved') is True
    required, _ = required_test_names(task)
    covered = set()
    for record in records:
        verification = record.get('verification') or record
        totals = counts(verification)
        if (verification.get('test_exit_code', verification.get('exit_code')) != 0
                or not totals['passed'] or totals['failed'] or totals['errors']):
            return False
        outcomes = junit_outcomes(record)
        if not outcomes or any(node['status'] in ('failed', 'errors') for node in outcomes):
            return False
        covered.update(name for name in required if any(
            node_matches(node, name) and node['status'] == 'passed' for node in outcomes))
    return bool(required) and covered == set(required)


def evaluate_task(task, profiles, invoke):
    pairs = [{'generation': index + 1, 'gold': invoke('gold', profile),
              'empty': invoke('empty', profile)} for index, profile in enumerate(profiles)]
    verdict = control_verdict(task, pairs)
    base = {'instance_id': task['instance_id'], 'controls': verdict,
            'required_profile_ids': profiles, 'normal_output': False, 'resolved': False}
    if verdict['gate'] != 'eligible':
        return {**base, 'status': 'control_failed'}
    agent = invoke('agent', profiles[0])
    validations = [agent]
    if agent.get('patch_chars', 0) > 0 and agent.get('patch_file'):
        validations.extend(invoke('verify-patch', profile, agent['patch_file']) for profile in profiles[1:])
    matrix_complete = len(validations) == len(profiles)
    infrastructure_failed = any(record.get('infra_error') for record in validations)
    status = ('infra_error' if infrastructure_failed else
              'agent_timeout' if agent.get('agent_timed_out') else 'completed')
    return {**base, 'status': status, 'agent': agent, 'validations': validations,
            'normal_output': normal_output(agent)
                             and not infrastructure_failed,
            'resolved': not infrastructure_failed and matrix_complete and score_patch(task, validations)}


def rolling(tasks, slots, work, complete):
    """Each slot is reused immediately, while its endpoint capacity stays fixed."""
    pending = iter(tasks)
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        active = {}
        for endpoint in slots:
            task = next(pending, None)
            if task is not None:
                active[pool.submit(work, task, endpoint)] = endpoint
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                endpoint = active.pop(future)
                complete(future.result())
                task = next(pending, None)
                if task is not None:
                    active[pool.submit(work, task, endpoint)] = endpoint


def invocation(args, task_id, endpoint, task_root):
    def invoke(mode, profile, patch_file=None):
        directory = task_root / (mode + '-' + profile)
        if directory.exists():
            # Never reuse an interrupted individual sandbox invocation.
            directory = task_root / (mode + '-' + profile + '-retry-' + str(time.time_ns()))
        directory.mkdir(parents=True)
        command = [sys.executable, str(args.task_runner), '--base', str(args.base), '--repo', str(args.repo),
                   '--run-dir', str(args.run_dir), '--task-id', task_id, '--endpoint', endpoint,
                   '--mode', mode, '--profile-id', profile, '--output-dir', str(directory),
                   '--gate', str(args.gate)]
        if patch_file:
            command.extend(['--patch-file', patch_file])
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
                       'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONNOUSERSITE': '1'}
        # Upstream leases permit at most 8 workers per directory. Give each model
        # endpoint its own directory so three 4-way services really admit 12 tasks.
        environment.update(GEMMA_SLOT_DIR=str(args.run_dir / 'endpoint-slots' /
                                             hashlib.sha256(endpoint.encode()).hexdigest()),
                           GEMMA_EVAL_SLOTS=str(dict(zip(args.endpoints, args.workers))[endpoint]))
        with (directory / 'process.log').open('w') as log:
            try:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                           start_new_session=True)
                process.wait(timeout=args.invocation_timeout)
            except subprocess.TimeoutExpired:
                # Only this invocation's newly created session is eligible for cleanup.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError('Per-invocation watchdog expired: ' + str(directory))
        result_file = directory / 'result.json'
        if process.returncode != 0 or not result_file.exists():
            raise RuntimeError('Task invocation failed: ' + str(directory))
        result = json.loads(result_file.read_text())
        if result.get('instance_id', result.get('task_id')) != task_id:
            raise RuntimeError('Task result identifier mismatch: ' + str(directory))
        return {**result, 'result_file': str(result_file), 'result_sha256': digest(result_file)}
    return invoke


def load_inputs(args):
    source = args.base / 'data/tasks.jsonl'
    tasks = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len({task['instance_id'] for task in tasks}) != len(tasks):
        raise ValueError('Duplicate task identifiers')
    gate = json.loads(args.gate.read_text())
    if gate['dataset_sha256'] != digest(source):
        raise ValueError('Dataset does not match preflight gate')
    if args.ids:
        selected = set(args.ids)
        if selected - {task['instance_id'] for task in tasks}:
            raise ValueError('Unknown selected task identifiers')
        tasks = [task for task in tasks if task['instance_id'] in selected]
    return tasks, gate


def run(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / 'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args)


def run_locked(args):
    slots = endpoint_slots(args.endpoints, args.workers)
    tasks, gate = load_inputs(args)
    manifest = {'dataset_sha256': digest(args.base / 'data/tasks.jsonl'), 'gate_sha256': digest(args.gate),
                'task_runner_sha256': digest(args.task_runner), 'scheduler_sha256': digest(__file__),
                'execution_source_sha256': execution_sources(args.task_runner),
                'finalizer_sha256': digest(Path(__file__).with_name('finalize_preflight.py')),
                'endpoints': args.endpoints, 'workers': args.workers,
                'task_ids': [task['instance_id'] for task in tasks],
                'external_fingerprint': json.loads(args.fingerprint.read_text()) if args.fingerprint else {}}
    bind_run(args.run_dir, manifest)
    completed = {}
    eligible = set(gate['eligible_ids'])
    for task in tasks:
        identifier = task['instance_id']
        path = result_path(args.run_dir, identifier)
        if path.exists():
            completed[identifier] = json.loads(path.read_text())
        elif identifier not in eligible:
            record = {'instance_id': identifier, 'status': 'preflight_excluded', 'normal_output': False,
                      'resolved': False, 'reason': 'nondiscriminating' if identifier in gate.get('nondiscriminating_ids', []) else 'quarantined'}
            commit_result(args.run_dir, identifier, record)
            completed[identifier] = record

    def progress(status):
        normal = sum(record.get('normal_output') is True for record in completed.values())
        report = {'status': status, 'updated_unix': time.time(), 'total': len(tasks),
                  'completed': len(completed), 'normal_outputs': normal,
                  'resolved': sum(record.get('resolved') is True for record in completed.values()),
                  'agent_budget_timeouts': sum(record.get('agent', {}).get('agent_timed_out') is True
                                              for record in completed.values()),
                  'statuses': {name: sum(record['status'] == name for record in completed.values())
                               for name in sorted({record['status'] for record in completed.values()})}}
        atomic_json(args.run_dir / 'progress.json', report)
        milestone = args.run_dir / 'ten-normal-outputs.json'
        if normal >= 10 and not milestone.exists():
            atomic_json(milestone, {**report, 'task_ids': [key for key, record in completed.items() if record.get('normal_output')][:10]})

    def work(task, endpoint):
        started = time.time()
        task_root = args.run_dir / 'tasks' / hashlib.sha256(task['instance_id'].encode()).hexdigest()
        try:
            record = evaluate_task(task, profiles_for(gate, task['instance_id']),
                                   invocation(args, task['instance_id'], endpoint, task_root))
        except Exception as error:
            record = {'instance_id': task['instance_id'], 'status': 'infra_error', 'normal_output': False,
                      'resolved': False, 'error': f'{type(error).__name__}: {error}'}
        return {**record, 'endpoint': endpoint, 'started_unix': started, 'finished_unix': time.time()}

    def complete(record):
        commit_result(args.run_dir, record['instance_id'], record)
        completed[record['instance_id']] = record
        progress('running')

    progress('running')
    rolling([task for task in tasks if task['instance_id'] not in completed], slots, work, complete)
    progress('completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'repo', 'run-dir', 'gate', 'task-runner'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fingerprint', type=Path)
    parser.add_argument('--endpoints', nargs='+', required=True)
    parser.add_argument('--workers', nargs='+', type=int, required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--invocation-timeout', type=int, default=1800)
    args = parser.parse_args()
    for name in ('base', 'repo', 'run_dir', 'gate', 'task_runner'):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == '__main__':
    main()
