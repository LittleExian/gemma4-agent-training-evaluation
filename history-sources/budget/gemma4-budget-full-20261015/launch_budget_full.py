"""Wait for verified inputs, supervise four GPU workers, and publish progress."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path('/workspace/gemma4-notebook-run-20261008')
RUN = Path(__file__).resolve().parent
PYTHON = BASE / 'runtime/bin/python'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def log(message):
    print(f'[{now()}] {message}', flush=True)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def diagnostic_counts(records):
    counts = Counter()
    for record in records:
        diagnostic = record.get('verification_diagnostic') or {}
        categories = diagnostic.get('categories', []) if isinstance(diagnostic, dict) else [str(diagnostic)]
        counts.update(categories or ['none'])
    return dict(counts)


class ExistingWorker:
    def __init__(self, entry):
        self.entry = entry
        self.returncode = None

    def poll(self):
        import psutil
        try:
            process = psutil.Process(self.entry['pid'])
            alive = process.create_time() == self.entry['create_time'] and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            summary = read_json(RUN / f'shard-{self.entry["index"]}/summary.json') or {}
            self.returncode = 0 if summary.get('status') == 'completed' else 1
        return self.returncode


def input_status():
    manifest = read_json(BASE / 'data/manifest.json')
    valid = sum((BASE / 'data' / x['name']).is_file()
                and (BASE / 'data' / x['name']).stat().st_size == x['size'] for x in manifest)
    ids = ['fastapi_15661', 'rich_4070', 'requests_7505', 'httpx_3672']
    verified = [x for x in ids if (read_json(RUN / 'preflight/results' / f'{x}.json') or {}).get('resolved') is True]
    return valid, len(manifest), verified


def aggregate(processes=None):
    tasks = [json.loads(line) for line in (BASE / 'data/tasks.jsonl').read_text().splitlines()]
    ids = {x['instance_id'] for x in tasks}
    records = [read_json(p) for p in RUN.glob('shard-*/task-results/*.json')]
    records = [r for r in records if r is not None]
    found = [r['instance_id'] for r in records]
    if len(found) != len(set(found)) or not set(found) <= ids:
        raise ValueError('Duplicate or unknown task result')
    resolved = sum(r.get('resolved') is True and r['status'] == 'evaluated' for r in records)
    infra = sum(r['status'] == 'infrastructure_error' for r in records)
    state = {'updated_at': now(), 'total_tasks': len(tasks), 'finished': len(records),
             'resolved': resolved, 'infrastructure_errors': infra,
             'verification_diagnostics': diagnostic_counts(records),
             'resolved_fraction_all_tasks': resolved / len(tasks),
             'by_repo': {}, 'workers': []}
    for repo in sorted({x['repo'] for x in tasks}):
        subset = [r for r in records if r.get('repo') == repo]
        state['by_repo'][repo] = {'total': sum(x['repo'] == repo for x in tasks),
                                 'finished': len(subset), 'resolved': sum(r.get('resolved') is True for r in subset)}
    for index in range(4):
        summary = read_json(RUN / f'shard-{index}/summary.json') or {}
        state['workers'].append({'gpu': index, 'status': summary.get('status', 'starting'),
                                 'current_task': summary.get('current_task'),
                                 'exit_code': processes[index].poll() if processes else None})
    if processes and all(p.poll() is not None for p in processes):
        state['status'] = ('completed' if len(records) == len(tasks)
                           and all(p.returncode == 0 for p in processes) else 'failed')
    else:
        state['status'] = 'running'
    write_json(RUN / 'progress.json', state)
    write_json(RUN / 'all-results.json', records)
    columns = ['instance_id', 'repo', 'status', 'resolved', 'test_exit_code',
               'duration_seconds', 'tool_calls', 'verification_diagnostic', 'error_message']
    with (RUN / 'results.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    return state


def main():
    lock = (RUN / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if '--monitor-only' in sys.argv:
        entries = read_json(RUN / 'worker-processes.json')
        if not entries or len(entries) != 4:
            raise ValueError('Four original worker identities are required')
        log('MONITOR: reattached to existing workers; no new GPU jobs launched')
        monitor([ExistingWorker(entry) for entry in entries])
        return
    log('PREPARING: waiting for all input files and four repository preflights')
    while True:
        valid, total, verified = input_status()
        log(f'PREPARING: input_files={valid}/{total}; golden_preflights={len(verified)}/4')
        write_json(RUN / 'progress.json', {'status': 'preparing', 'updated_at': now(),
                   'input_files': valid, 'total_input_files': total, 'preflights_passed': verified,
                   'total_tasks': 129, 'finished': 0})
        if valid == total and len(verified) == 4:
            break
        time.sleep(30)
    subprocess.run([str(PYTHON), '-m', 'pip', 'check'], check=True)
    with (RUN / 'runtime-freeze.txt').open('w') as handle:
        subprocess.run([str(PYTHON), '-m', 'pip', 'freeze'], stdout=handle, check=True)
    processes = []
    handles = []
    for index in range(4):
        shard = RUN / f'shard-{index}'
        shard.mkdir(exist_ok=True)
        handle = (shard / 'worker.log').open('a')
        handles.append(handle)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(index), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
               'PYTHONUNBUFFERED': '1', 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'PATH': str(BASE / 'runtime/bin') + ':/usr/bin:/bin'}
        args = [str(PYTHON), str(RUN / 'budget_full_run.py'), '--base', str(BASE),
                '--run-dir', str(RUN), '--shard-index', str(index), '--shard-count', '4',
                '--gpu-memory-fraction', '0.65']
        process = subprocess.Popen(args, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        processes.append(process)
        log(f'STARTED: GPU={index}; worker_pid={process.pid}; assigned_tasks={33 if index == 0 else 32}')
    import psutil
    write_json(RUN / 'worker-processes.json', [
        {'index': index, 'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time()}
        for index, process in enumerate(processes)])
    monitor(processes)
    for handle in handles:
        handle.close()


def monitor(processes):
    last = None
    while True:
        status = aggregate(processes)
        message = (f'PROGRESS: {status["finished"]}/{status["total_tasks"]}; resolved={status["resolved"]}; '
                   f'infrastructure_errors={status["infrastructure_errors"]}; '
                   + '; '.join(f'GPU{x["gpu"]}={x["status"]}:{x["current_task"]}' for x in status['workers']))
        if message != last:
            log(message)
            last = message
        if status['status'] != 'running':
            log(f'FINAL: status={status["status"]}; resolved={status["resolved"]}/{status["total_tasks"]}; '
                f'finished={status["finished"]}; see progress.json, all-results.json and results.csv')
            break
        time.sleep(20)


if __name__ == '__main__':
    main()
