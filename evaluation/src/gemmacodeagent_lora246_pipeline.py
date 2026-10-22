"""Evaluate the final Gemma W4 adapter with a fixed pilot and fresh calibrated 129-task run."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gemmacodeagent_mm_pipeline import (assert_identity, atomic_json, bind_json,
    checked_process, cleanup_secondary, digest, records_at, validate_records)
from gemmacodeagent_mm_server import digest as streaming_digest

ARM = 'gemma4_w4_lora246'
BASE_ALIAS = 'gemma-4-31b-it-qat-w4a16-ct'
BASE_SHA = '1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20'
GPUS, PORTS = (0, 2), (8140, 8142)


def validate_scope(pilot, full):
    if (len(pilot) != 16 or len(set(pilot)) != 16 or len(full) != 129
            or len(set(full)) != 129 or not set(pilot) <= set(full)):
        raise ValueError('Expected the pinned 16-task pilot and all 129 unique tasks')


def validate_package(root, manifest):
    adapter = manifest.get('adapter', {})
    packaged = root/'submissions/mm15_6000/adapters'/ARM
    discovered = manifest.get('discovery', {}).get('adapters', [])
    if (manifest.get('checkpoint') != 246 or adapter.get('name') != ARM
            or adapter.get('path') != str(packaged)
            or manifest.get('base_model', {}).get('path') != str(root/'models'/BASE_ALIAS)
            or manifest.get('base_model', {}).get('weights_sha256') != BASE_SHA
            or not any(item.get('name') == ARM and item.get('path') == str(packaged/'adapter_model.safetensors')
                       for item in discovered)):
        raise ValueError('Package must bind checkpoint 246, original W4 and official adapter discovery')
    source = Path(adapter.get('source_path', ''))
    for filename, key in [('adapter_config.json', 'config_sha256'),
                          ('adapter_model.safetensors', 'weights_sha256')]:
        path = packaged/filename
        if (path.is_symlink() or not path.is_file() or not adapter.get(key)
                or streaming_digest(path) != adapter[key]
                or streaming_digest(source/filename) != adapter[key]):
            raise ValueError('Adapter must be a verified real copy of the final checkpoint: '+filename)
    return packaged


def validate_deployment_gate(path, manifest):
    proof = manifest.get('deployment_gate', {})
    gate = json.loads(path.read_text())
    exported = gate.get('export', {})
    if (manifest.get('checkpoint') != 246 or proof.get('path') != str(path)
            or proof.get('sha256') != digest(path)
            or proof.get('adapter_weights_sha256') != manifest.get('adapter', {}).get('weights_sha256')
            or gate.get('passed') is not True or gate.get('zero_adapter_max_logit_delta') != 0
            or gate.get('trained_adapter_max_logit_delta', 0) <= 1e-6
            or exported.get('rank') != 16 or exported.get('tensor_count') != 820):
        raise ValueError('Final checkpoint 246 must pass the bound original-W4 deployment gate')


def validate_health(report):
    rows = report.get('rows', [])
    if (report.get('passed') is not True or report.get('concurrency') != 4
            or report.get('models') != [ARM] or ARM not in report.get('thinking_tokens', {})
            or not any(row.get('request', {}).get('model') == ARM
                       and row['request'].get('tools') for row in rows)):
        raise RuntimeError('Adapter health must validate tools, thinking budget, and four-way serving')


def start_models(root, adapter, directory, environment):
    models, started = [], time.time()
    wrapper = str(root/'src/gemmacodeagent_lora246_server.py')
    try:
        for gpu, port in zip(GPUS, PORTS):
            state = directory/('gpu-'+str(gpu)+'-'+str(time.time_ns()))
            state.mkdir(parents=True)
            command = [sys.executable, wrapper, 'launch', '--root', str(root),
                '--model', str(root/'models'/BASE_ALIAS), '--adapter', str(adapter),
                '--state', str(state), '--gpu', str(gpu), '--port', str(port), '--slots', '4']
            try:
                checked_process(command, state/'launch.log', environment, models, timeout=600)
            finally:
                if (state/'process.json').exists():
                    models.append(json.loads((state/'process.json').read_text()))
            checked_process([sys.executable, wrapper, 'health', '--state', str(state),
                '--timeout', '1200'], state/'health.log', environment, models, timeout=1800)
            validate_health(json.loads((state/'health.json').read_text()))
        return models, started
    except BaseException:
        cleanup_secondary(models, started, directory/'cleanup-secondary.json')
        raise


def summarize(records, identifiers, scope, excluded_ids):
    validate_records(records, identifiers, allow_excluded=scope == 'full')
    excluded = sorted(row['instance_id'] for row in records if row.get('status') == 'preflight_excluded')
    expected = sorted(set(identifiers) & set(excluded_ids)) if scope == 'full' else []
    if excluded != expected or any(row.get('resolved') for row in records if row['instance_id'] in excluded):
        raise RuntimeError('Known exceptional tasks must be reported separately and never resolved')
    return {'passed': True, 'arm': ARM, 'scope': scope, 'total': len(records),
        'eligible': len(records)-len(excluded), 'excluded': len(excluded), 'excluded_ids': excluded,
        'resolved': sum(row.get('resolved') is True for row in records),
        'timeouts': sum(row.get('status') == 'agent_timeout' for row in records)}


def evaluate(root, base, gate, arm, scope, ids, directory, models, environment, fingerprint):
    output = directory/scope/arm
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(root/'src/gemmacodeagent_mm_run.py'),
        '--base', str(base), '--repo', str(root), '--run-dir', str(output), '--gate', str(gate),
        '--task-runner', str(root/'src/gemmacodeagent_mm_task.py'),
        '--endpoints', *[f'http://127.0.0.1:{port}/v1' for port in PORTS],
        '--workers', *['4' for _ in PORTS], '--fingerprint', str(fingerprint), '--ids', *ids]
    checked_process(command, output/'runner.log', {**environment, 'G4_MM_ARM': arm}, models)
    gate_data = json.loads(gate.read_text())
    excluded = gate_data['quarantined_ids']+gate_data['nondiscriminating_ids']
    report = {**summarize(records_at(output), ids, scope, excluded),
              'output': str(output), 'finished_unix': time.time()}
    atomic_json(output/'compatibility.json', report)
    return report


def verify_runtime(root):
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    subprocess.run([sys.executable, str(root/'scripts/verify_environment.py')], check=True)


def build_protocol(root, base, gate, experiment, manifest_path, deployment_gate):
    model = root/'models'/BASE_ALIAS
    if streaming_digest(model/'model.safetensors') != BASE_SHA:
        raise ValueError('Original Kaggle packed W4 checkpoint hash changed')
    files = [p for p in (root/'src').glob('*.py') if not p.name.startswith('test_')]
    files += [p for p in experiment.rglob('*') if p.is_file()]
    files += [p for p in (root/'submissions/mm15_6000').rglob('*') if p.is_file()]
    return {'upstream_commit': '29984e6638ef6825ad2636164d59df972c19d0f5',
        'arm': ARM, 'checkpoint': 246, 'gpus': list(GPUS), 'ports': list(PORTS), 'slots_per_gpu': 4, 'total_slots': 8,
        'dataset_sha256': digest(base/'data/tasks.jsonl'), 'gate_sha256': digest(gate),
        'adapter_package_manifest_sha256': digest(manifest_path),
        'deployment_gate_sha256': digest(deployment_gate), 'base_weight_sha256': BASE_SHA,
        'base_metadata_sha256': {p.name: streaming_digest(p) for p in sorted(model.iterdir())
                                if p.is_file() and p.suffix != '.safetensors'},
        'runtime_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True),
        'critical_source_manifest_sha256': digest(root/'environment/installed_code_hashes.json'),
        'source_sha256': {str(p.relative_to(root)): streaming_digest(p) for p in sorted(set(files))},
        'protocol': {'minutes': 15, 'thinking': 6000, 'context': 32768, 'tools': 56,
            'turns': 160, 'output': 8192, 'temperature': .2, 'top_p': .95, 'top_k': 64},
        'calibration': 'Fresh gold/empty controls; three known exceptional tasks reported separately',
        'pilot_reuse': 'None; full129 reruns every pilot task with a fresh invocation',
        'hardware': 'Two PRO 6000, TP1, four requests per GPU; differs from official TP4 L4 timing'}


def run(args):
    root, directory = args.root.resolve(), args.run_dir.resolve()
    if not directory.is_relative_to(root/'runs') or directory == root/'runs':
        raise ValueError('Pipeline directory must remain inside isolated repository/runs')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/'pipeline-status.json').exists():
            raise ValueError('Use a fresh pipeline directory; completed evidence is never transplanted')
        manifest = json.loads(args.manifest.read_text())
        validate_package(root, manifest)
        validate_deployment_gate(args.deployment_gate, manifest)
        verify_runtime(root)
        experiment = root/'experiments/distributed_model_comparison'
        pilot_ids = json.loads((experiment/'pilot16.json').read_text())['tasks']
        full_ids = json.loads((experiment/'full129.json').read_text())['tasks']
        validate_scope(pilot_ids, full_ids)
        protocol = build_protocol(root, args.base, args.gate, experiment, args.manifest, args.deployment_gate)
        bind_json(directory/'protocol.json', protocol)
        environment = {**os.environ, 'PYTHONNOUSERSITE': '1', 'OTEL_SDK_DISABLED': 'true',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        state = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
                 'protocol_sha256': digest(directory/'protocol.json'), 'reports': {}}
        def save(**changes):
            nonlocal state
            state = {**state, **changes, 'updated_unix': time.time()}
            atomic_json(directory/'pipeline-status.json', state)
        save()
        try:
            for scope, ids in [('pilot16', pilot_ids), ('full', full_ids)]:
                if scope == 'full' and not state['reports'].get('pilot16', {}).get('passed'):
                    raise RuntimeError('Fixed pilot must pass before full129')
                save(scope=scope, phase='model_health')
                services = directory/scope/'services'
                models, started = start_models(root, Path(manifest['adapter']['source_path']), services, environment)
                try:
                    save(phase='evaluating', models=models)
                    report = evaluate(root, args.base, args.gate, ARM, scope, ids, directory,
                                      models, environment, directory/'protocol.json')
                    save(reports={**state['reports'], scope: report})
                finally:
                    active_error = sys.exc_info()[0] is not None
                    cleanup = cleanup_secondary(models, started, services/'cleanup-secondary.json')
                    save(models=[], cleanup=cleanup)
                    if not active_error and any('cleanup_error' in row for row in cleanup):
                        raise RuntimeError('Owned service cleanup incomplete; inspect evidence')
            save(status='completed', phase='complete', finished_unix=time.time())
        except BaseException as error:
            save(status='blocked', phase='error', error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'base', 'gate', 'run-dir', 'manifest', 'deployment-gate'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
