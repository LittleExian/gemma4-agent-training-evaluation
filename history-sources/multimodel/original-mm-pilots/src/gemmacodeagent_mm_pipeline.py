"""Seven pinned model arms: finish all fixed pilots before admitting full runs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gemmacodeagent_smoke_driver import release_owned_model

ARMS = ('gemma4_kaggle', 'qwen35_base', 'qwen35_swegym', 'qwen38_base',
        'qwen38_coder390', 'gemma4_teich_agent', 'gemma4_capybara')
GROUPS = ((ARMS[0],), ARMS[1:3], (ARMS[3],), (ARMS[4],), (ARMS[5],), (ARMS[6],))
GPUS = (0, 2, 3)
PORTS = (8140, 8142, 8143)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError('Protocol drift; use a new run directory: ' + str(path))
    atomic_json(path, value)


def assert_identity(record, psutil_module=None):
    if psutil_module is None:
        import psutil as psutil_module
    process = psutil_module.Process(record['pid'])
    if (process.create_time() != record['create_time']
            or process.uids().real != record['uid']
            or process.status() == psutil_module.STATUS_ZOMBIE):
        raise RuntimeError('Owned process identity changed')


def require_pilots(pilots):
    if set(pilots) != set(ARMS) or not all(pilots[a].get('passed') is True for a in ARMS):
        raise RuntimeError('Every model must finish its fixed pilot before full evaluation')


def validate_records(records, identifiers, allow_excluded=False):
    ids = [row.get('instance_id') for row in records]
    if len(ids) != len(set(ids)) or set(ids) != set(identifiers):
        raise RuntimeError('Missing, duplicate, or unexpected task result')
    for row in records:
        if allow_excluded and row.get('status') == 'preflight_excluded':
            continue
        if row.get('status') not in ('completed', 'agent_timeout'):
            raise RuntimeError('Non-evaluation terminal state: ' + str(row.get('instance_id')))
        validations = row.get('validations', [])
        if not validations or any(v.get('infra_error') for v in validations):
            raise RuntimeError('Missing verification or infrastructure failure')


def records_at(directory):
    return [json.loads(p.read_text()) for p in sorted((directory/'results').glob('*.json'))]


def process_record(process):
    import psutil
    return {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
            'uid': os.getuid()}


def checked_process(command, log_path, environment, models=(), timeout=None):
    """Continuously check model identities; cleanup is restricted to this process tree."""
    started = time.time()
    with log_path.open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=environment,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        record = process_record(process)
        atomic_json(log_path.with_suffix('.process.json'), {**record, 'command': command})
        try:
            while process.poll() is None:
                for model in models:
                    assert_identity(model)
                if timeout is not None and time.time() - started > timeout:
                    raise TimeoutError('Subprocess exceeded its bounded runtime')
                time.sleep(2)
            if process.returncode:
                raise RuntimeError('Subprocess failed; inspect ' + str(log_path))
        except BaseException:
            release_owned_model(record, started)
            raise


def wait_downloads(root, experiment, cache, group, environment, directory):
    specs = json.loads((experiment/'models.json').read_text())
    for arm in group:
        if arm == 'gemma4_kaggle':
            from gemmacodeagent_mm_server import digest as streaming_digest
            model = root/'models/gemma-4-31b-it-qat-w4a16-ct'
            expected = '1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20'
            weights = list(model.glob('*.safetensors'))
            if len(weights) != 1 or streaming_digest(weights[0]) != expected:
                raise RuntimeError('Original Kaggle checkpoint hash changed')
            atomic_json(directory/'original-model-verified.json',
                        {'sha256': expected, 'config_sha256': digest(model/'config.json')})
            continue
        # The downloader verifies every cached file, including completed downloads.
        checked_process([sys.executable, str(root/'src/gemmacodeagent_mm_download.py'),
            'download', '--experiment-dir', str(experiment), '--cache-dir', str(cache),
            '--arm', arm, '--workers', '16'], directory/(arm+'-download.log'),
            environment, timeout=48*3600)
    return specs


def start_models(root, experiment, cache, group, specs, directory, environment):
    models = []
    started = time.time()
    try:
        for gpu, port in zip(GPUS, PORTS):
            state = directory/('gpu-'+str(gpu)+'-'+str(time.time_ns()))
            state.mkdir(parents=True)
            arm = group[0]
            model = (root/'models/gemma-4-31b-it-qat-w4a16-ct' if arm == 'gemma4_kaggle'
                     else cache/specs[arm]['directory'])
            command = [sys.executable, str(root/'src/gemmacodeagent_mm_server.py'), 'launch',
                '--root', str(root), '--models-json', str(experiment/'models.json'),
                '--arm', arm, '--model', str(model), '--state', str(state),
                '--gpu', str(gpu), '--port', str(port), '--slots', '4']
            if arm == 'qwen35_base':
                command += ['--adapter', str(cache/specs['qwen35_swegym']['directory'])]
            # The launcher records ownership before returning; retain the record even
            # when its exit status is nonzero so partially started children are covered.
            try:
                checked_process(command, state/'launch.log', environment, models, timeout=180)
            finally:
                if (state/'process.json').exists():
                    models.append(json.loads((state/'process.json').read_text()))
            checked_process([sys.executable, str(root/'src/gemmacodeagent_mm_server.py'),
                'health', '--state', str(state), '--timeout', '1200'],
                state/'health.log', environment, models, timeout=1800)
        return models, started
    except BaseException:
        for record in reversed(models):
            release_owned_model(record, started)
        raise


def evaluate(root, base, gate, arm, scope, ids, directory, models, environment, fingerprint):
    output = directory/scope/arm
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(root/'src/gemmacodeagent_mm_run.py'),
        '--base', str(base), '--repo', str(root), '--run-dir', str(output), '--gate', str(gate),
        '--task-runner', str(root/'src/gemmacodeagent_mm_task.py'),
        '--endpoints', *[f'http://127.0.0.1:{port}/v1' for port in PORTS],
        '--workers', '4', '4', '4', '--fingerprint', str(fingerprint), '--ids', *ids]
    checked_process(command, output/'runner.log', {**environment, 'G4_MM_ARM': arm}, models)
    records = records_at(output)
    validate_records(records, ids, allow_excluded=scope == 'full')
    report = {'passed': True, 'arm': arm, 'scope': scope, 'total': len(records),
        'resolved': sum(row.get('resolved') is True for row in records),
        'timeouts': sum(row.get('status') == 'agent_timeout' for row in records),
        'excluded': sum(row.get('status') == 'preflight_excluded' for row in records),
        'output': str(output), 'finished_unix': time.time()}
    atomic_json(output/'compatibility.json', report)
    return report


def build_protocol(root, base, gate, experiment):
    files = [p for p in (root/'src').glob('*.py') if not p.name.startswith('test_')]
    files += [p for p in experiment.rglob('*') if p.is_file()]
    files += [p for p in (root/'submissions/mm15_6000').rglob('*') if p.is_file()]
    return {'commit': '29984e6638ef6825ad2636164d59df972c19d0f5',
        'arms': list(ARMS), 'gpus': list(GPUS), 'ports': list(PORTS), 'slots_per_gpu': 4,
        'dataset_sha256': digest(base/'data/tasks.jsonl'), 'gate_sha256': digest(gate),
        'runtime_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True),
        'critical_source_manifest_sha256': digest(root/'environment/installed_code_hashes.json'),
        'original_model_config_sha256': digest(root/'models/gemma-4-31b-it-qat-w4a16-ct/config.json'),
        'source_sha256': {str(p.relative_to(root)): digest(p) for p in sorted(set(files))},
        'protocol': {'minutes': 15, 'thinking': 6000, 'context': 32768,
            'tools': 56, 'turns': 160, 'output': 8192, 'temperature': .2, 'top_p': .95},
        'calibration': 'Fresh gold/empty per eligible task; 3 known invalid tasks reported separately',
        'pilot_reuse': 'Full evaluation runs afresh, including the pilot tasks',
        'policy': 'All seven fixed16 pilots required; never silently skip an incompatible model'}


def run(args):
    root, directory = args.root.resolve(), args.run_dir.resolve()
    if not directory.is_relative_to(root/'runs'):
        raise ValueError('Pipeline directory must be inside repository/runs')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experiment = root/'experiments/distributed_model_comparison'
        subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
        subprocess.run([sys.executable, str(root/'scripts/verify_environment.py')], check=True)
        protocol = build_protocol(root, args.base, args.gate, experiment)
        bind_json(directory/'protocol.json', protocol)
        cache = root/'research/distributed_model_comparison/models'
        pilot_ids = json.loads((experiment/'pilot16.json').read_text())['tasks']
        full_ids = json.loads((experiment/'full129.json').read_text())['tasks']
        if len(set(pilot_ids)) != 16 or len(set(full_ids)) != 129 or not set(pilot_ids) <= set(full_ids):
            raise ValueError('Pinned task scope is invalid')
        environment = {**os.environ, 'PYTHONNOUSERSITE': '1', 'OTEL_SDK_DISABLED': 'true',
                       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        state = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
                 'pilots': {}, 'full': {}, 'protocol_sha256': digest(directory/'protocol.json')}
        def save():
            atomic_json(directory/'pipeline-status.json', {**state, 'updated_unix': time.time()})
        save()
        try:
            for scope, ids in [('pilot16', pilot_ids), ('full', full_ids)]:
                if scope == 'full':
                    require_pilots(state['pilots'])
                    atomic_json(directory/'all-seven-pilots-passed.json', state['pilots'])
                for group in GROUPS:
                    state.update(scope=scope, model_group=list(group), phase='downloading')
                    save()
                    specs = wait_downloads(root, experiment, cache, group, environment, directory)
                    state['phase'] = 'model_health'
                    save()
                    models, started = start_models(root, experiment, cache, group, specs,
                                                  directory/scope/('services-'+group[0]), environment)
                    try:
                        state.update(phase='evaluating', models=models)
                        save()
                        for arm in group:
                            state['arm'] = arm
                            save()
                            report = evaluate(root, args.base, args.gate, arm, scope, ids,
                                directory, models, environment, directory/'protocol.json')
                            state['pilots' if scope == 'pilot16' else 'full'][arm] = report
                            save()
                    finally:
                        state['cleanup'] = [release_owned_model(record, started) for record in reversed(models)]
                        state.pop('models', None)
                        save()
            state.update(status='completed', phase='complete', finished_unix=time.time())
            save()
        except BaseException as error:
            state.update(status='blocked', phase='error', error=f'{type(error).__name__}: {error}')
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'base', 'gate', 'run-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__': main()
