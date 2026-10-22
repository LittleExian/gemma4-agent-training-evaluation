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

from gemmacodeagent_owned_cleanup import release_owned_model

ARMS = ('gemma4_kaggle', 'qwen35_base', 'qwen35_swegym', 'qwen38_base',
        'qwen38_coder390', 'gemma4_teich_agent', 'gemma4_capybara')
GROUPS = ((ARMS[0],), ARMS[1:3], (ARMS[3],), (ARMS[4],), (ARMS[5],), (ARMS[6],))
GPUS = (0, 2, 3)
PORTS = (8140, 8142, 8143)
SERVING_ADAPTATIONS = {'gemma4_capybara': {
    'change': 'append <|channel>thought\n to assistant generation prefix only when enable_thinking=True and add_generation_prompt=True',
    'reasoning_parser': 'capybara_budget (prefilled thought-channel parsing)',
    'reason': 'fine-tune can omit channel marker, bypassing thinking-token accounting',
    'weights_quantization_sampling_evaluator': 'unchanged'}}



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


def process_record(process, launch_boottime):
    import psutil
    return {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
            'uid': os.getuid(), 'launch_boottime': launch_boottime}


def checked_process(command, log_path, environment, models=(), timeout=None):
    """Continuously check model identities; cleanup is restricted to this process tree."""
    started = time.time()
    with log_path.open('a') as log:
        launch_boottime = time.clock_gettime(time.CLOCK_BOOTTIME)
        process = subprocess.Popen(command, stdout=log, stderr=log, env=environment,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        record = process_record(process, launch_boottime)
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
            cleanup_secondary([record], started, log_path.with_suffix('.cleanup-secondary.json'))
            raise


def cleanup_secondary(records, started, path):
    """Attempt every owned cleanup and retain errors without hiding an active failure."""
    outcomes = []
    for record in reversed(records):
        try:
            outcomes.append(release_owned_model(record, started))
        except BaseException as error:
            outcomes.append({'cleanup_error': f'{type(error).__name__}: {error}',
                             'record': record})
    try:
        atomic_json(path, outcomes)
    except Exception as error:
        outcomes.append({'cleanup_error': f'Unable to save cleanup evidence: {error}'})
        print(outcomes[-1]['cleanup_error'], file=sys.stderr)
    return outcomes


def recover_pilots(source, target, current_protocol, identifiers):
    """Explicitly import immutable evidence; never transplant mutable runner checkpoints."""
    source, target = source.resolve(), target.resolve()
    if source == target or source.parent.name != 'runs':
        raise RuntimeError('Recovery requires a distinct source repository run')
    old = json.loads((source/'protocol.json').read_text())
    if ('serving_adaptations' in current_protocol
            and current_protocol['serving_adaptations'] != SERVING_ADAPTATIONS):
        raise RuntimeError('Recovery serving adaptation drift')
    old_fields = {k: v for k, v in old.items() if k != 'source_sha256'}
    new_fields = {k: v for k, v in current_protocol.items()
                  if k not in ('source_sha256', 'recovery', 'serving_adaptations')}
    if old_fields != new_fields:
        raise RuntimeError('Recovery execution protocol drift')
    allowed_changes = {'src/gemmacodeagent_mm_pipeline.py', 'src/gemmacodeagent_mm_server.py'}
    allowed_additions = {'src/gemmacodeagent_owned_cleanup.py',
                         'src/gemmacodeagent_capybara_template.py',
                         'src/gemmacodeagent_capybara_parser.py'}
    previous, current = old['source_sha256'], current_protocol['source_sha256']
    for name, sha in previous.items():
        if name not in allowed_changes and current.get(name) != sha:
            raise RuntimeError('Recovery source drift: ' + name)
        if name not in current:
            raise RuntimeError('Recovery missing source: ' + name)
    if set(current) - set(previous) - allowed_additions:
        raise RuntimeError('Recovery unexpected source drift')
    # The old run must still contain the exact code bound when its pilots were run.
    source_root = source.parent.parent
    for name, sha in previous.items():
        if digest(source_root/name) != sha:
            raise RuntimeError('Recovery historical source drift: ' + name)
    state = json.loads((source/'pipeline-status.json').read_text())
    if ('gemma4_capybara' in state.get('pilots', {})
            and old.get('serving_adaptations') != SERVING_ADAPTATIONS):
        raise RuntimeError('Recovery changed Capybara serving requires fresh pilot')
    pilots, evidence = {}, {}
    for group in GROUPS:
        available = [arm for arm in group if arm in state.get('pilots', {})]
        if not available:
            continue
        if available != list(group):
            raise RuntimeError('Recovery incomplete shared model pilot group')
        health_hashes = {}
        for gpu in GPUS:
            matches = list((source/'pilot16'/('services-'+group[0])).glob('gpu-'+str(gpu)+'-*/health.json'))
            if len(matches) != 1:
                raise RuntimeError('Recovery missing or ambiguous model health evidence')
            path = matches[0]
            health = json.loads(path.read_text())
            if (health.get('passed') is not True or health.get('concurrency') != 4
                    or set(health.get('models', [])) != set(group)):
                raise RuntimeError('Recovery failed or mismatched model health')
            health_hashes[str(path)] = digest(path)
        for arm in available:
            output = source/'pilot16'/arm
            report_path = output/'compatibility.json'
            report = json.loads(report_path.read_text())
            records = records_at(output)
            validate_records(records, identifiers)
            expected = {'passed': True, 'arm': arm, 'scope': 'pilot16', 'total': len(identifiers),
                'resolved': sum(row.get('resolved') is True for row in records),
                'timeouts': sum(row.get('status') == 'agent_timeout' for row in records),
                'excluded': 0, 'output': str(output)}
            if any(report.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Recovery compatibility report mismatch')
            if state['pilots'][arm] != report:
                raise RuntimeError('Recovery pipeline report mismatch')
            pilots[arm] = {**report, 'imported': True, 'source_run': str(source),
                           'source_output': str(output)}
            evidence[arm] = {'compatibility_sha256': digest(report_path),
                'result_sha256': {str(path): digest(path) for path in sorted((output/'results').glob('*.json'))},
                'health_sha256': health_hashes}
    return pilots, {'source_run': str(source), 'source_protocol_sha256': digest(source/'protocol.json'),
                   'source_status_sha256': digest(source/'pipeline-status.json'), 'pilots': evidence,
                   'policy': 'Only completed fixed pilots imported; every full evaluation runs fresh'}


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
        cleanup_secondary(models, started, directory/'cleanup-secondary.json')
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
        'serving_adaptations': SERVING_ADAPTATIONS,
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
        cache = (root/'research/distributed_model_comparison/models').resolve()
        pilot_ids = json.loads((experiment/'pilot16.json').read_text())['tasks']
        full_ids = json.loads((experiment/'full129.json').read_text())['tasks']
        if len(set(pilot_ids)) != 16 or len(set(full_ids)) != 129 or not set(pilot_ids) <= set(full_ids):
            raise ValueError('Pinned task scope is invalid')
        environment = {**os.environ, 'PYTHONNOUSERSITE': '1', 'OTEL_SDK_DISABLED': 'true',
                       'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        imported, recovery = ({}, None)
        if getattr(args, 'resume_from', None) is not None:
            imported, recovery = recover_pilots(args.resume_from, directory, protocol, pilot_ids)
            protocol = {**protocol, 'recovery': recovery}
        bind_json(directory/'protocol.json', protocol)
        state = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
                 'pilots': imported, 'full': {}, 'protocol_sha256': digest(directory/'protocol.json')}
        def save():
            atomic_json(directory/'pipeline-status.json', {**state, 'updated_unix': time.time()})
        save()
        try:
            for scope, ids in [('pilot16', pilot_ids), ('full', full_ids)]:
                if scope == 'full':
                    require_pilots(state['pilots'])
                    atomic_json(directory/'all-seven-pilots-passed.json', state['pilots'])
                for group in GROUPS:
                    if scope == 'pilot16' and all(arm in state['pilots'] for arm in group):
                        continue
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
                        active_error = sys.exc_info()[0] is not None
                        state['cleanup'] = cleanup_secondary(models, started,
                            directory/scope/('services-'+group[0])/'cleanup-secondary.json')
                        state.pop('models', None)
                        save()
                        if not active_error and any('cleanup_error' in row for row in state['cleanup']):
                            raise RuntimeError('Owned service cleanup incomplete; inspect cleanup-secondary.json')
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
    parser.add_argument('--resume-from', type=Path, help='Explicit completed pilot evidence from a separate prior run')
    run(parser.parse_args())


if __name__ == '__main__': main()
