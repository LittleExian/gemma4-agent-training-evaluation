"""Queue one calibrated held-out validation job after the completed Kaggle job."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import gemmacodeagent_lora246_pipeline as serving
from gemmacodeagent_mm_pipeline import (atomic_json, bind_json, checked_process,
    cleanup_secondary, digest)
from gemmacodeagent_mm_run import fail_fast_rolling
from gemmacodeagent_mm_server import digest as streaming_digest

GPUS, PORTS, ARM = serving.GPUS, serving.PORTS, serving.ARM


def endpoint_slots(workers_per_gpu=4):
    if workers_per_gpu not in (1, 2, 4):
        raise ValueError('Explicit workers per GPU must be 1, 2, or 4')
    return [f'http://127.0.0.1:{port}/v1' for port in PORTS for _ in range(workers_per_gpu)]


def validate_inputs(selection, dataset, gate_path):
    identifiers = json.loads(selection.read_text())['tasks']
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    actual = [row.get('instance_id', row.get('task_id')) for row in rows]
    gate = json.loads(gate_path.read_text())
    bindings = gate.get('tasks', {})
    artifacts = gate.get('artifact_sha256', {})
    if (len(identifiers) != 20 or len(set(identifiers)) != 20
            or any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
            or len(actual) != 20 or set(actual) != set(identifiers)
            or gate.get('passed') is not True or set(bindings) != set(identifiers)
            or gate.get('selection_sha256') != digest(selection)
            or gate.get('dataset_sha256') != digest(dataset) or not artifacts):
        raise ValueError('All 20 held-out tasks must have a matching complete CPU calibration gate')
    for identifier in identifiers:
        proof = bindings[identifier]
        if (proof.get('eligible') is not True or proof.get('gold_passed') is not True
                or proof.get('empty_passed') is not False):
            raise ValueError('Validation task lacks discriminating gold/empty controls: '+identifier)
    for name, sha in artifacts.items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or streaming_digest(path) != sha:
            raise ValueError('CPU calibration artifact drift: '+name)
    return identifiers


def _owned_process_alive(record, psutil_module):
    try:
        process = psutil_module.Process(record['pid'])
        return (process.create_time() == record['create_time']
                and process.uids().real == record['uid']
                and process.status() != psutil_module.STATUS_ZOMBIE)
    except psutil_module.NoSuchProcess:
        return False


def _cleanup_targets_alive(cleanup, records, psutil_module):
    earliest = min(record['create_time'] for record in records)
    targets = {pid for row in cleanup for key in ('terminated', 'killed') for pid in row.get(key, [])}
    for pid in targets:
        try:
            process = psutil_module.Process(pid)
            if (process.uids().real == os.getuid() and process.create_time() >= earliest
                    and process.status() != psutil_module.STATUS_ZOMBIE):
                return True
        except psutil_module.NoSuchProcess:
            continue
    return False


def verify_predecessor_services(directory, psutil_module=None):
    if psutil_module is None:
        import psutil as psutil_module
    ready = True
    for scope in ('pilot16', 'full'):
        services = directory/scope/'services'
        cleanup = json.loads((services/'cleanup-secondary.json').read_text())
        records = list(services.glob('gpu-*/process.json'))
        if len(records) != 2 or len(cleanup) != 2 or any('cleanup_error' in row for row in cleanup):
            raise RuntimeError('Kaggle owned-service cleanup evidence is incomplete')
        observed, identities = set(), []
        for path in records:
            record = json.loads(path.read_text())
            identities.append(record)
            gpu = record['physical_gpu_index_at_start']
            observed.add(gpu)
            if (gpu not in GPUS or record['port'] != PORTS[GPUS.index(gpu)]
                    or record['models'] != [ARM] or record['uid'] != os.getuid()
                    or 'launch_boottime' not in record):
                raise RuntimeError('Kaggle service identity or GPU scope differs')
            ready = not _owned_process_alive(record, psutil_module) and ready
        if observed != set(GPUS):
            raise RuntimeError('Kaggle services do not describe the expected two cards')
        ready = not _cleanup_targets_alive(cleanup, identities, psutil_module) and ready
    return ready


def predecessor_ready(path):
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state.get('status') in ('blocked', 'failed', 'error', 'cancelled'):
        raise RuntimeError('Kaggle predecessor is '+state['status']+': '+str(state.get('error', '')))
    if state.get('status') != 'completed':
        return False
    reports = state.get('reports', {})
    cleanup = state.get('cleanup', [])
    if (state.get('models') or len(cleanup) != 2 or any('cleanup_error' in row for row in cleanup)
            or reports.get('pilot16', {}).get('passed') is not True
            or reports.get('pilot16', {}).get('total') != 16
            or reports.get('full', {}).get('passed') is not True
            or reports.get('full', {}).get('total') != 129):
        raise RuntimeError('Kaggle completion or owned cleanup is not valid')
    return verify_predecessor_services(path.parent)


def gpu_readiness(check_output=subprocess.check_output, psutil_module=None):
    """Read existing processes without modifying them; insufficient capacity stays queued."""
    if psutil_module is None:
        import psutil as psutil_module
    raw = check_output(['nvidia-smi', '--id=0,2',
        '--query-gpu=index,uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True)
    inventory = []
    for fields in csv.reader(raw.splitlines(), skipinitialspace=True):
        gpu, uuid, name, total, free = fields
        item = {'gpu': int(gpu), 'uuid': uuid, 'name': name, 'total_mib': int(total),
                'free_mib': int(free), 'required_free_mib': .65*int(total)+8192}
        if item['gpu'] not in GPUS or 'PRO 6000' not in name:
            raise RuntimeError('Requested physical GPU inventory differs from the two PRO 6000 cards')
        inventory.append(item)
    if len(inventory) != 2 or {item['gpu'] for item in inventory} != set(GPUS):
        raise RuntimeError('Missing or duplicate requested GPU inventory')
    uuids = {item['uuid'] for item in inventory}
    raw = check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                        '--format=csv,noheader,nounits'], text=True)
    foreign, identity_ready = [], True
    for fields in csv.reader(raw.splitlines(), skipinitialspace=True):
        if not fields or fields[0] not in uuids:
            continue
        uuid, pid, name, memory = fields
        item = {'gpu_uuid': uuid, 'pid': int(pid), 'name': name, 'memory_mib': memory}
        try:
            process = psutil_module.Process(item['pid'])
            item = {**item, 'uid': process.uids().real, 'create_time': process.create_time()}
        except psutil_module.NoSuchProcess:
            item = {**item, 'already_exited': True}
        except psutil_module.AccessDenied:
            item = {**item, 'identity_unavailable': True}
            identity_ready = False
        foreign.append(item)
    return {'ready': identity_ready and all(item['free_mib'] >= item['required_free_mib'] for item in inventory),
            'gpus': inventory, 'existing_compute_processes': foreign, 'policy': 'Read only; never terminate existing processes'}


def host_memory_readiness(workers_per_gpu, meminfo=None):
    endpoint_slots(workers_per_gpu)
    if meminfo is None:
        meminfo = Path('/proc/meminfo').read_text()
    values = {line.split(':', 1)[0]: int(line.split()[1]) for line in meminfo.splitlines()}
    available = values['MemAvailable']/2**20
    required = 16+len(GPUS)*workers_per_gpu*4
    return {'ready': available >= required, 'available_gib': available, 'required_gib': required,
            'total_gib': values['MemTotal']/2**20,
            'swap_used_gib': (values.get('SwapTotal', 0)-values.get('SwapFree', 0))/2**20,
            'source': '/proc/meminfo MemAvailable; swap excluded', 'workers_per_gpu': workers_per_gpu}


def wait_until_ready(args, save):
    remaining = args.wait_timeout
    status_path = getattr(args, 'run_dir', Path('/nonexistent'))/'pipeline-status.json'
    if status_path.is_file():
        started = json.loads(status_path.read_text())['started_unix']
        remaining = min(remaining, max(0., started+args.wait_timeout-time.time()))
    if remaining <= 0:
        raise TimeoutError('Readiness deadline expired before resources became available')
    deadline = time.monotonic()+remaining
    required = [args.selection, args.dataset, args.gate, args.manifest, args.deployment_gate, args.task_runner]
    validated_signature = None
    while True:
        predecessor = predecessor_ready(args.wait_for)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            save(phase='waiting_inputs', missing_inputs=missing)
        else:
            signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                              for path in (args.selection, args.dataset, args.gate))
            if signature != validated_signature:
                validate_inputs(args.selection, args.dataset, args.gate)
                validated_signature = signature
            if not predecessor:
                save(phase='waiting_kaggle_cleanup', missing_inputs=[])
            else:
                memory = host_memory_readiness(getattr(args, 'workers_per_gpu', 4))
                save(phase='memory_ready' if memory['ready'] else 'waiting_host_memory',
                     host_memory_readiness=memory, missing_inputs=[])
                if memory['ready']:
                    gpu = gpu_readiness()
                    save(phase='gpu_ready' if gpu['ready'] else 'waiting_gpu', gpu_readiness=gpu, missing_inputs=[])
                    if gpu['ready']:
                        return {**gpu, 'host_memory': memory}
        if time.monotonic() >= deadline:
            raise TimeoutError('Readiness deadline expired; see pipeline-status.json for the blocking phase')
        time.sleep(10)


def validate_result(record, identifier):
    verification = record.get('verification') or {}
    code = verification.get('test_exit_code')
    if (record.get('instance_id') != identifier
            or record.get('status') not in ('completed', 'agent_timeout')
            or record.get('infra_error') or type(record.get('normal_output')) is not bool
            or type(record.get('resolved')) is not bool or type(code) is not int
            or any(type(verification[key]) is not int or verification[key] < 0
                   for key in ('passed', 'failed', 'errors') if key in verification)
            or (record['resolved'] and (code != 0 or verification.get('expected_tests_covered') is not True
                or type(verification.get('passed')) is not int or verification['passed'] <= 0
                or verification.get('failed') != 0 or verification.get('errors') != 0))):
        raise ValueError('Task returned missing/invalid identity, status, or verification: '+identifier)


def invoke_task(args, identifier, endpoint, models, environment):
    key = hashlib.sha256(identifier.encode()).hexdigest()
    output, logdir = args.run_dir/'tasks'/key, args.run_dir/'task-logs'/key
    started = time.time()
    try:
        if output.exists():
            raise RuntimeError('Refusing to reuse an existing validation task output')
        logdir.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(args.task_runner), '--task-id', identifier,
                   '--endpoint', endpoint, '--output', str(output), '--root', str(args.root)]
        checked_process(command, logdir/'task.log', environment, models, timeout=1800)
        path = output/'result.json'
        record = json.loads(path.read_text())
        validate_result(record, identifier)
        return {**record, 'endpoint': endpoint, 'started_unix': started, 'finished_unix': time.time(),
                'result_file': str(path), 'result_sha256': digest(path)}
    except Exception as error:
        return {'instance_id': identifier, 'status': 'infra_error', 'normal_output': False, 'resolved': False,
                'endpoint': endpoint, 'started_unix': started, 'finished_unix': time.time(),
                'task_directory': str(output), 'log_directory': str(logdir), 'error': f'{type(error).__name__}: {error}'}


def evaluate(args, identifiers, models, environment):
    completed = {}
    def progress(status):
        report = {'status': status, 'total': 20, 'completed': len(completed),
            'normal_outputs': sum(row.get('normal_output') is True for row in completed.values()),
            'resolved': sum(row.get('resolved') is True for row in completed.values()),
            'agent_timeouts': sum(row.get('status') == 'agent_timeout' for row in completed.values()),
            'infrastructure_errors': sum(row.get('status') == 'infra_error' for row in completed.values()),
            'updated_unix': time.time()}
        atomic_json(args.run_dir/'progress.json', report)
        return report
    def complete(record):
        identifier = record['instance_id']
        path = args.run_dir/'results'/(hashlib.sha256(identifier.encode()).hexdigest()+'.json')
        if identifier in completed or path.exists():
            raise RuntimeError('Refusing to replace an existing validation task result')
        atomic_json(path, record)
        completed[identifier] = record
        progress('running')
    progress('running')
    try:
        fail_fast_rolling(identifiers, endpoint_slots(getattr(args, 'workers_per_gpu', 4)),
            lambda identifier, endpoint: invoke_task(args, identifier, endpoint, models, environment), complete)
        if set(completed) != set(identifiers):
            raise RuntimeError('Validation dispatch did not produce every selected task')
    except BaseException:
        progress('blocked')
        raise
    report = progress('completed')
    atomic_json(args.run_dir/'summary.json', report)
    return report


def build_protocol(args):
    main_protocol = args.wait_for.parent/'protocol.json'
    previous = json.loads(main_protocol.read_text())
    for name, sha in previous['source_sha256'].items():
        if streaming_digest(args.service_root/name) != sha:
            raise ValueError('Completed Kaggle source/package drift: '+name)
    model = args.service_root/'models'/serving.BASE_ALIAS
    if (previous.get('base_weight_sha256') != serving.BASE_SHA
            or streaming_digest(model/'model.safetensors') != serving.BASE_SHA):
        raise ValueError('Original W4 model drift since the completed Kaggle evaluation')
    metadata = previous.get('base_metadata_sha256', {})
    if not metadata or any(streaming_digest(model/name) != sha for name, sha in metadata.items()):
        raise ValueError('Original W4 model metadata drift since Kaggle evaluation')
    sources = [path for path in (args.root/'src').glob('*.py') if not path.name.startswith('test_')]
    if args.task_runner not in sources:
        sources.append(args.task_runner)
    return {'scope': 'heldout_validation20', 'checkpoint': 246, 'arm': ARM,
        'gpus': list(GPUS), 'ports': list(PORTS), 'slots_per_gpu': getattr(args, 'workers_per_gpu', 4),
        'total_slots': len(endpoint_slots(getattr(args, 'workers_per_gpu', 4))),
        'resource_readiness': json.loads((args.run_dir/'pipeline-status.json').read_text()).get('resource_readiness')
            if hasattr(args, 'run_dir') else None,
        'selection_sha256': digest(args.selection), 'dataset_sha256': digest(args.dataset),
        'cpu_gate_sha256': digest(args.gate), 'adapter_manifest_sha256': digest(args.manifest),
        'deployment_gate_sha256': digest(args.deployment_gate),
        'kaggle_completed_status_sha256': digest(args.wait_for), 'kaggle_protocol_sha256': digest(main_protocol),
        'source_sha256': {str(path): streaming_digest(path) for path in sorted(set(sources))},
        'runtime_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True),
        'protocol': {'minutes': 15, 'tools': 56, 'turns': 160, 'output': 8192,
                     'thinking': 6000, 'context': 32768, 'temperature': .2, 'top_p': .95, 'top_k': 64},
        'task_output_policy': 'Task creates fresh run/tasks/<sha>; controller audit logs stay in separate run/task-logs/<sha>',
        'policy': 'All 20 require gold-pass/empty-fail; infrastructure errors stop new dispatch and drain in-flight tasks'}


def run_ready(args, save):
    identifiers = validate_inputs(args.selection, args.dataset, args.gate)
    manifest = json.loads(args.manifest.read_text())
    serving.validate_package(args.service_root, manifest)
    serving.validate_deployment_gate(args.deployment_gate, manifest)
    serving.verify_runtime(args.service_root)
    protocol = build_protocol(args)
    memory = host_memory_readiness(getattr(args, 'workers_per_gpu', 4))
    if not memory['ready']:
        save(phase='waiting_host_memory', host_memory_readiness=memory)
        resources = wait_until_ready(args, save)
        save(resource_readiness=resources)
        identifiers = validate_inputs(args.selection, args.dataset, args.gate)
        serving.validate_package(args.service_root, manifest)
        protocol = build_protocol(args)
    bind_json(args.run_dir/'protocol.json', protocol)
    environment = {**os.environ, 'PYTHONNOUSERSITE': '1', 'OTEL_SDK_DISABLED': 'true',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'G4_VALIDATION_ROOT': str(args.root),
        'SUBMISSION_DIR': str(args.service_root/'submissions/mm15_6000'), 'G4_MM_ARM': ARM}
    save(phase='model_health', protocol_sha256=digest(args.run_dir/'protocol.json'))
    services = args.run_dir/'services'
    models, started = serving.start_models(args.service_root, Path(manifest['adapter']['source_path']), services, environment)
    try:
        save(phase='evaluating', models=models)
        report = evaluate(args, identifiers, models, environment)
        save(report=report)
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup = cleanup_secondary(models, started, services/'cleanup-secondary.json')
        save(models=[], cleanup=cleanup)
        if not active_error and any('cleanup_error' in row for row in cleanup):
            raise RuntimeError('Validation owned-service cleanup is incomplete')


def run(args):
    args.root, args.service_root, args.run_dir = args.root.resolve(), args.service_root.resolve(), args.run_dir.resolve()
    if not args.run_dir.is_relative_to(args.root/'runs') or args.run_dir == args.root/'runs':
        raise ValueError('Validation run directory must remain inside its isolated repository/runs')
    if not 0 < args.wait_timeout <= 48*3600:
        raise ValueError('Readiness timeout must be positive and at most 48 hours')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.run_dir/'pipeline-status.json').exists():
            raise ValueError('Use a fresh validation run directory')
        state = {'status': 'waiting', 'phase': 'queued', 'pid': os.getpid(),
                 'started_unix': time.time(), 'wait_timeout_seconds': args.wait_timeout,
                 'workers_per_gpu': getattr(args, 'workers_per_gpu', 4),
                 'total_slots': len(endpoint_slots(getattr(args, 'workers_per_gpu', 4)))}
        def save(**changes):
            nonlocal state
            state = {**state, **changes, 'updated_unix': time.time()}
            atomic_json(args.run_dir/'pipeline-status.json', state)
        save()
        try:
            resources = wait_until_ready(args, save)
            save(status='running', resource_readiness=resources)
            run_ready(args, save)
            save(status='completed', phase='complete', finished_unix=time.time())
        except BaseException as error:
            save(status='blocked', phase='error', error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'service-root', 'wait-for', 'selection', 'dataset', 'gate',
                 'manifest', 'deployment-gate', 'task-runner', 'run-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wait-timeout', type=float, default=36*3600)
    parser.add_argument('--workers-per-gpu', type=int, choices=(1, 2, 4), default=4)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
