"""Keep three owned model services alive until the full evaluation terminates."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from gemmacodeagent_full_server import validate_options
from gemmacodeagent_smoke_driver import release_owned_model


def validate_config(config, run):
    models = config['models']
    if len(models) != 3 or {m['gpu'] for m in models} != {0, 2, 3}:
        raise ValueError('Exactly three PRO services on physical GPU 0, 2, 3 are required')
    if len({m['port'] for m in models}) != 3 or len({str(Path(m['state_dir']).resolve()) for m in models}) != 3:
        raise ValueError('Model ports and state directories must be distinct')
    for model in models:
        validate_options(model['gpu'], model['slots'], model['memory_utilization'], model['port'])
        state = Path(model['state_dir'])
        if not state.is_absolute() or run.resolve() not in state.resolve().parents:
            raise ValueError('Each service state must be an absolute child of this run directory')
    if not Path(config['runner']).is_absolute() or not isinstance(config['runner_args'], list):
        raise ValueError('Runner must be absolute and runner_args must be an argv list')
    if any(not isinstance(arg, str) for arg in config['runner_args']):
        raise ValueError('Runner arguments must be strings')


def bind_config(run, config):
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    target = run/'supervisor-config.json'
    if target.exists() and json.loads(target.read_text())['sha256'] != fingerprint:
        raise RuntimeError('Refusing to reuse run directory with a changed supervisor config')
    target.write_text(json.dumps({'sha256': fingerprint, 'config': config}, indent=2))
    return fingerprint


def assert_live(record):
    import psutil
    process = psutil.Process(record['pid'])
    if (process.create_time() != record['create_time'] or process.uids().real != os.getuid()
            or process.status() == psutil.STATUS_ZOMBIE):
        raise RuntimeError('Owned service identity changed or service exited')


def wait_healthy(record, port, state):
    import requests
    for _ in range(120):
        assert_live(record)
        try:
            response = requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=3)
            if response.status_code == 200:
                (state/'health.json').write_text(json.dumps(response.json(), indent=2))
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError('Model startup exceeded its health-check budget')


def write_status(run, status):
    temporary = run/'supervisor-status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(run/'supervisor-status.json')


def monitor_runner(process, records, models, interval=15):
    """Identity loss fails immediately; transient HTTP failures get three attempts."""
    import requests
    failures = {model['port']: 0 for model in models}
    while process.poll() is None:
        for record, model in zip(records, models):
            assert_live(record)
            try:
                response = requests.get(f"http://127.0.0.1:{model['port']}/v1/models", timeout=3)
                healthy = response.status_code == 200
            except requests.RequestException:
                healthy = False
            failures[model['port']] = 0 if healthy else failures[model['port']] + 1
            if failures[model['port']] >= 3:
                raise RuntimeError(f"Model port {model['port']} failed three consecutive health checks")
        time.sleep(interval)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def run_evaluation(python, config, run, status, environment, started):
    import psutil
    record = None
    with (run/'runner.log').open('a') as log:
        process = subprocess.Popen([python, config['runner'], *config['runner_args']],
                                   env=environment, stdout=log, stderr=log,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            record = {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
                      'uid': os.getuid(), 'command': process.args}
            (run/'runner-process.json').write_text(json.dumps(record, indent=2))
            write_status(run, {**status, 'runner': record})
            monitor_runner(process, status['models'], config['models'])
        finally:
            if record is not None and process.poll() is None:
                cleanup = release_owned_model(record, started)
                (run/'runner-cleanup.json').write_text(json.dumps(cleanup, indent=2))
                process.wait(timeout=30)


def orchestrate(repo, run, config):
    started = time.time()
    python = str(repo/'envs/eval/bin/python')
    status = {'status': 'preflight', 'started_unix': started, 'models': []}
    write_status(run, status)
    try:
        subprocess.run([python, '-m', 'pip', 'check'], check=True)
        subprocess.run([python, str(repo/'scripts/verify_environment.py')], check=True)
        for model in config['models']:
            state = Path(model['state_dir'])
            status = {**status, 'status': 'starting_models', 'starting_gpu': model['gpu']}
            write_status(run, status)
            subprocess.run([python, str(Path(__file__).with_name('gemmacodeagent_full_server.py')),
                            '--repo', str(repo), '--state-dir', str(state), '--gpu', str(model['gpu']),
                            '--port', str(model['port']), '--concurrency', str(model['slots']),
                            '--memory-utilization', str(model['memory_utilization'])], check=True)
            record = json.loads((state/'process.json').read_text())
            if record['create_time'] < started:
                raise RuntimeError('Refusing to adopt a service predating this supervisor')
            status = {**status, 'models': [*status['models'], record]}
            write_status(run, status)
            wait_healthy(record, model['port'], state)
        status = {**status, 'status': 'evaluating', 'model_ready_unix': time.time()}
        write_status(run, status)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1',
                       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        run_evaluation(python, config, run, status, environment, started)
        status = {**status, 'status': 'completed', 'finished_unix': time.time()}
    except BaseException as error:
        status = {**status, 'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                  'finished_unix': time.time()}
        raise
    finally:
        cleanup = []
        # Read every state: a launcher can succeed before a later health or record check fails.
        for model in config['models']:
            try:
                state = Path(model['state_dir'])/'process.json'
                if not state.exists():
                    continue
                record = json.loads(state.read_text())
                if record['create_time'] < started:
                    continue
                cleanup.append({'gpu': model['gpu'], **release_owned_model(record, started)})
            except Exception as error:
                cleanup.append({'gpu': model['gpu'], 'error': f'{type(error).__name__}: {error}'})
        write_status(run, {**status, 'cleanup': cleanup})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_dir.is_absolute():
        raise ValueError('Run directory must be absolute')
    repo, run = args.repo.resolve(), args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    validate_config(config, run)
    with (run/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_config(run, config)
        from gemmacodeagent_full_server import refuse_live_record
        import psutil
        for model in config['models']:
            refuse_live_record(Path(model['state_dir']), psutil)
        orchestrate(repo, run, config)


if __name__ == '__main__':
    main()
