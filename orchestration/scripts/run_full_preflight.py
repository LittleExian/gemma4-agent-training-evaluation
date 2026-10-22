"""Run all gold/empty checks with CPU workers and export an explicit task gate."""
import collections
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/workspace/gemma4-notebook-run-20261008')
PYTHON = BASE / 'runtime/bin/python'


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def aggregate(generation, workers, profile_map):
    pairs = [load(p) for p in (ROOT / f'generation-{generation}').glob('shard-*/results/*/calibration.json')]
    pairs = [p for p in pairs if p]
    identifiers = [p['instance_id'] for p in pairs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Duplicate calibration task IDs')
    counts = collections.Counter(p['status'] for p in pairs)
    summary = {'generation': generation, 'total_tasks': 129, 'completed': len(pairs),
               'calibrated': counts['calibrated'], 'nondiscriminating': counts['nondiscriminating'],
               'uncalibrated': counts['uncalibrated'], 'pending': 129-len(pairs),
               'workers': [p.poll() for p in workers]}
    summary['status'] = ('running' if any(p.poll() is None for p in workers) else
                         'completed' if len(pairs) == 129 and all(p.returncode == 0 for p in workers) else 'failed')
    suffix = '' if generation == 1 else f'-g{generation}'
    save(ROOT / f'progress{suffix}.json', summary)
    save(ROOT / f'all-calibration{suffix}.json', pairs)
    gate = {'complete_preflight': summary['status'] == 'completed',
            'profile_map_sha256': hashlib.sha256(profile_map.read_bytes()).hexdigest(),
            'eligible_task_ids': sorted(p['instance_id'] for p in pairs if p['status'] == 'calibrated'),
            'quarantined': {p['instance_id']: {'status':p['status'],'reason':p['reason']}
                            for p in pairs if p['status'] != 'calibrated'},
            'policy': 'Only eligible tasks may be scored with these profiles. Keep all 129 in the audit; report quarantined tasks separately.'}
    save(ROOT / f'calibration-gate{suffix}.json', gate)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=int, default=1)
    parser.add_argument('--runner', default='calibrate_all_tasks.py')
    parser.add_argument('--profile-map', default=str(ROOT/'task-profiles.json'))
    args = parser.parse_args()
    profile_map = Path(args.profile_map)
    import fcntl
    lock = (ROOT / f'supervisor-g{args.generation}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mapping = load(profile_map)
    while True:
        statuses = [load(ROOT/'installation'/(key+'.json')) for key in mapping['profiles']]
        failures = [x['profile'] for x in statuses if x and not x['success']]
        if failures:
            raise RuntimeError(f'Profile installation failed: {failures}')
        ready = sum(bool(x and x['success']) for x in statuses)
        print(f'PROFILES {ready}/{len(statuses)} ready', flush=True)
        if ready == len(statuses):
            break
        time.sleep(15)
    workers, logs = [], []
    generation = args.generation
    for index in range(8):
        log = (ROOT / f'worker-g{generation}-{index}.log').open('a')
        logs.append(log)
        command = [str(PYTHON), '-u', str(ROOT/args.runner), '--base', str(BASE),
                '--run-dir', str(ROOT), '--profile-map', str(profile_map),
                '--generation', str(generation), '--shard-index', str(index), '--shard-count', '8']
        environment = {**os.environ, 'LITELLM_LOCAL_MODEL_COST_MAP':'True', 'PYTHONNOUSERSITE':'1',
                       'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
                       'CUDA_VISIBLE_DEVICES':''}
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment, start_new_session=True)
        workers.append(process)
        print(f'CPU WORKER {index} pid={process.pid}', flush=True)
    last = None
    while True:
        summary = aggregate(generation, workers, profile_map)
        text = json.dumps(summary, sort_keys=True)
        if text != last:
            print(text, flush=True)
            last = text
        if summary['status'] != 'running':
            break
        time.sleep(15)
    for log in logs:
        log.close()


if __name__ == '__main__':
    main()
