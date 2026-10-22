"""Fresh, bounded, three-PRO-GPU weighted LoRA training supervisor."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

from gemmacodeagent_owned_cleanup import release_owned_model

REVISION = '52f3f65bc7a02d555763bc923bd1d9094898219d'
GPU_NAME = 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
TOTAL_STEPS = {'smoke': 1, 'full': 246}
TIMEOUT_SECONDS = {'smoke': 7200, 'full': 72 * 3600}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def fingerprint(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def validate_recipe_data(path, scope):
    with path.open() as stream:
        samples = sum(bool(line.strip()) for line in stream)
    if samples == 0 or (scope == 'full' and samples != 2951):
        raise ValueError('Dataset does not match the fixed training recipe')
    return samples


def verify_manifest(provenance, manifest):
    expected = manifest.get('source_sha256')
    if (manifest.get('training_data_sha256') != provenance['data_sha256']
            or not isinstance(expected, dict)
            or {name: sha for name, sha in expected.items() if name.endswith('.py')} != provenance['source_sha256']):
        raise ValueError('Source or training data differs from the immutable manifest')


def validate_host_memory(available):
    if not isinstance(available, int) or available < 24 * 2**30:
        raise ValueError('At least 24 GiB of available host memory is required')
    return available


def validate_paths(args):
    if not args.root.is_absolute() or not args.root.is_dir():
        raise ValueError('An absolute existing root is required')
    root = args.root.resolve()
    for value in (args.source, args.output, args.data, args.hf_home, args.manifest):
        if value is not None and (not value.is_absolute() or not value.resolve().is_relative_to(root)):
            raise ValueError('Training paths must stay inside root')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('A fresh output directory is required')
    for parent in args.output.parents:
        if parent == args.root:
            break
        if parent.is_symlink():
            raise ValueError('Output parent symlinks are forbidden')
    if not (args.source/'train.py').is_file() or not args.data.is_file() or not args.hf_home.is_dir():
        raise ValueError('Training source, data and HF cache must exist')
    if args.manifest is not None and not args.manifest.is_file():
        raise ValueError('Fingerprint manifest is missing')
    credential_root = Path('/workspace/.credentials')
    if (not args.credential_file.is_absolute()
            or not args.credential_file.is_relative_to(credential_root)
            or not args.credential_file.resolve().is_relative_to(credential_root)):
        raise ValueError('Credential must be inside the private credential directory')
    if args.scope not in TOTAL_STEPS or not args.run_name.strip() or any(ord(c) < 32 for c in args.run_name):
        raise ValueError('Invalid scope or run name')


def read_credential(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError('Credential must be an owned regular file with mode 0600')
        if not 0 < info.st_size <= 4096:
            raise ValueError('Invalid credential size')
        with os.fdopen(descriptor, 'r', closefd=False) as stream:
            value = stream.read(4097).strip()
        if not value or any(c.isspace() or ord(c) < 32 for c in value):
            raise ValueError('Invalid credential format')
        return value
    finally:
        os.close(descriptor)


def select_gpus(inventory, applications):
    rows = {}
    for line in inventory.splitlines():
        index, uuid, name, total, free = [value.strip() for value in line.split(',')]
        if index in rows:
            raise ValueError('Duplicate GPU index')
        rows[index] = (uuid, name, int(total), int(free))
    selected = []
    for index in ('0', '2', '3'):
        uuid, name, total, free = rows.get(index, ('', '', 0, 0))
        if name != GPU_NAME or total < 90 * 1024 or free < 90 * 1024 or not uuid.startswith('GPU-'):
            raise ValueError('Selected GPU is not an available 96GB PRO 6000')
        selected.append(uuid)
    if len(set(selected)) != 3:
        raise ValueError('GPU UUIDs must be distinct')
    for line in applications.splitlines():
        if line.strip() and line.split(',')[0].strip() in selected:
            raise ValueError('A selected GPU already has a compute process')
    return selected


def build_command(args, entity):
    if args.scope not in TOTAL_STEPS:
        raise ValueError('Unknown training scope')
    command = [str(args.root/'.venv-w4/bin/python'), '-m', 'torch.distributed.run',
        '--standalone', '--nnodes=1', '--nproc_per_node=3', '--max_restarts=0', str(args.source/'train.py'),
        '--data', str(args.data), '--output', str(args.output), '--revision', REVISION,
        '--accum', '4', '--lr', '2e-5', '--seed', '42', '--attention', 'sdpa',
        '--wandb-mode', 'online', '--wandb-project', 'repo-agent-lora', '--wandb-entity', entity,
        '--wandb-name', args.run_name]
    return command + (['--smoke', '--save-every', '1'] if args.scope == 'smoke'
                      else ['--steps', '0', '--save-every', '50'])


def training_environment(args, uuids, key, entity, inherited):
    return {**{k: v for k, v in inherited.items() if k not in ('WANDB_RUN_ID', 'WANDB_RESUME')},
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': ','.join(uuids),
        'PYTHONNOUSERSITE': '1', 'HF_HOME': str(args.hf_home), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'NCCL_DEBUG': 'WARN',
        'WANDB_API_KEY': key, 'WANDB_ENTITY': entity, 'WANDB_PROJECT': 'repo-agent-lora',
        'WANDB_NAME': args.run_name, 'WANDB_MODE': 'online', 'WANDB_CONSOLE': 'off',
        'WANDB_DISABLE_CODE': 'true', 'WANDB_SAVE_CODE': 'false', 'WANDB_DIR': str(args.output)}


def last_metrics(output):
    rows = {}
    for rank in range(3):
        path = output/f'metrics.rank{rank}.jsonl'
        if path.exists():
            for line in reversed(path.read_bytes().splitlines(keepends=True)):
                if line.endswith(b'\n') and line.strip():
                    row = json.loads(line)
                    rows[str(rank)] = {k: v for k, v in row.items() if k != 'sample_ids'}
                    break
    return rows


def verify_completion(output, scope):
    expected = TOTAL_STEPS[scope]
    metrics = last_metrics(output)
    if set(metrics) != {'0', '1', '2'} or any(row.get('step') != expected for row in metrics.values()):
        raise ValueError('Every rank must finish the final optimizer step')
    for row in metrics.values():
        for name in ('loss', 'gradient_norm', 'adapter_delta_max'):
            value = row.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or (name != 'loss' and value <= 0):
                raise ValueError('Invalid final optimization metric')
    checkpoint = output/f'checkpoint-{expected}'
    complete = json.loads((checkpoint/'COMPLETE.json').read_text())
    if complete != {'step': expected, 'world_size': 3}:
        raise ValueError('Final checkpoint is not complete for all three ranks')
    for rank in range(3):
        if not (checkpoint/f'rng.rank{rank}.pt').is_file() or not (checkpoint/f'rng.rank{rank}.pt').stat().st_size:
            raise ValueError('Final checkpoint is missing a rank RNG state')
    for path in (checkpoint/'adapter_audit.json', output/'base_audit.json'):
        if json.loads(path.read_text()).get('passed') is not True:
            raise ValueError('Base or adapter audit failed')
    if scope == 'smoke':
        smoke = json.loads((output/'SMOKE_PASS.json').read_text())
        before = json.loads((output/'base_audit.json').read_text())
        after = json.loads((output/'base_audit_after.json').read_text())
        if (after != before or after.get('passed') is not True
                or smoke.get('passed') is not True or smoke.get('world_size') != 3
                or smoke.get('steps') != 1 or smoke.get('reload_max_abs_logit_difference') != 0):
            raise ValueError('Three-rank smoke or roundtrip verification failed')
    return {'passed': True, 'steps': expected, 'world_size': 3, 'checkpoint': str(checkpoint)}


def run(args):
    import psutil
    validate_paths(args)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started, record, process = time.time(), None, None
    state = {'scope': args.scope, 'state': 'running', 'phase': 'preflight',
             'total_steps': TOTAL_STEPS[args.scope], 'started_unix': started}
    def progress():
        try:
            rows = last_metrics(args.output)
            json.dumps(rows, allow_nan=False)
        except (ValueError, OSError) as error:
            rows = {}
            state['metrics_error_type'] = type(error).__name__
        atomic_json(args.output/'progress.json', {**state, 'elapsed_seconds': time.time()-started, 'ranks': rows})
    try:
        progress()
        connectivity = json.loads((args.root/'work/wandb-connectivity.json').read_text())
        if connectivity.get('passed') is not True or not isinstance(connectivity.get('entity'), str) or not connectivity['entity'].strip():
            raise ValueError('W&B connectivity gate is missing or failed')
        entity = connectivity['entity']
        validate_recipe_data(args.data, args.scope)
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
        uuids = select_gpus(inventory, apps)
        command = build_command(args, entity)
        provenance = {'command': command, 'gpu_uuids': uuids, 'data_sha256': fingerprint(args.data),
            'source_sha256': {str(p.relative_to(args.source)): fingerprint(p) for p in sorted(args.source.rglob('*.py'))},
            'supervisor_sha256': fingerprint(Path(__file__)), 'run_name': args.run_name}
        if args.manifest is not None:
            raw = args.manifest.read_bytes()
            verify_manifest(provenance, json.loads(raw))
            (args.output/'input-manifest.json').write_bytes(raw)
            provenance['external_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
        provenance['host_available_bytes'] = validate_host_memory(psutil.virtual_memory().available)
        atomic_json(args.output/'launch.json', provenance)
        environment = training_environment(args, uuids, read_credential(args.credential_file), entity, os.environ)
        state['phase'] = 'training'
        with (args.output/'train.log').open('a') as log:
            boottime = time.clock_gettime(time.CLOCK_BOOTTIME)
            process = subprocess.Popen(command, env=environment, cwd=args.source, stdout=log, stderr=log,
                stdin=subprocess.DEVNULL, start_new_session=True)
            record = {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
                      'uid': os.getuid(), 'launch_boottime': boottime}
            atomic_json(args.output/'process.json', record)
            while process.poll() is None:
                progress()
                if time.time()-started > TIMEOUT_SECONDS[args.scope]:
                    raise TimeoutError('Training exceeded its wall time bound')
                time.sleep(10)
        if process.returncode != 0:
            raise ChildProcessError('Training process exited unsuccessfully')
        state['phase'] = 'verifying'
        state['verification'] = verify_completion(args.output, args.scope)
        state.update(state='completed', phase='complete', exit_code=0)
    except BaseException as error:
        state.update(state='failed', error_type=type(error).__name__, exit_code=130 if isinstance(error, KeyboardInterrupt) else 1)
        if record is not None:
            try:
                state['cleanup'] = release_owned_model(record, started)
            except BaseException as cleanup_error:
                state['cleanup_error_type'] = type(cleanup_error).__name__
    finally:
        state['finished_unix'] = time.time()
        progress()
        atomic_json(args.output/'exit.json', {**state, 'child_exit_code': process.returncode if process else None})
    return state['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'output', 'data', 'hf-home', 'credential-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--scope', choices=tuple(TOTAL_STEPS), required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        return run(args)
    except BaseException as error:
        print('Supervisor failed: '+type(error).__name__, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
