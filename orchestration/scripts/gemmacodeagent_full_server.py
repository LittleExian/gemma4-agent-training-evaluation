"""Launch one isolated, bounded PRO 6000 service; never stop existing services."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile


def validate_options(gpu, concurrency, memory, port):
    if gpu not in (0, 2, 3) or concurrency not in range(1, 5):
        raise ValueError('Require an available PRO GPU index 0/2/3 and 1..4 sequences')
    if not math.isfinite(memory) or not .35 <= memory <= .70 or not 1024 <= port <= 65535:
        raise ValueError('Require memory utilization .35.. .70 and port 1024..65535')


def validate_gpu(line, utilization):
    values = next(csv.reader([line], skipinitialspace=True))
    if len(values) != 4:
        raise RuntimeError('Unexpected GPU inventory response')
    uuid, name, total, free = values
    total, free = float(total), float(free)
    if 'RTX PRO 6000' not in name or free < total * utilization + 8192:
        raise RuntimeError('Selected GPU type/free memory cannot satisfy budget plus 8 GiB margin')
    return {'uuid': uuid, 'name': name, 'total_mib': total, 'free_mib': free}


def validate_uuid(expected, actual):
    if expected.removeprefix('GPU-').lower() != actual.removeprefix('GPU-').lower():
        raise RuntimeError('CUDA device UUID differs from selected physical GPU')


def config_kwargs(root, port, concurrency, utilization):
    return dict(
        model=str(root/'models/gemma-4-31b-it-qat-w4a16-ct'),
        host='127.0.0.1', port=port, tool_call_parser='gemma4', reasoning_parser='gemma4',
        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=utilization,
        enable_auto_tool_choice=True, enable_lora=False, scheduler_reserve_full_isl=False,
        tensor_parallel_size=1, limit_mm_per_prompt='{"image":0,"audio":0,"video":0}',
        extra_args=['--served-model-name', 'gemma-4-31b-it-qat-w4a16-ct',
                    '--max-num-seqs', str(concurrency), '--max-num-batched-tokens', '4096',
                    '--enforce-eager'])


def launch_environment(original, state, gpu, temporary_dir):
    return {**original, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': str(gpu),
            'HF_HOME': str(state/'cache/huggingface'), 'XDG_CACHE_HOME': str(state/'cache'),
            'TMPDIR': str(temporary_dir), 'TORCHINDUCTOR_CACHE_DIR': str(state/'cache/inductor'),
            'TRITON_CACHE_DIR': str(state/'cache/triton'),
            'VLLM_CACHE_ROOT': str(state/'cache/vllm'), 'PYTHONNOUSERSITE': '1',
            'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}


def verify_records(root):
    for filename, key in [('environment-verified.json', 'official_six_source_hashes_match'),
                          ('model-sha256.json', 'matches')]:
        if json.loads((root/'runs'/filename).read_text()).get(key) is not True:
            raise RuntimeError('Fresh official source and checkpoint verification are required')


def refuse_live_record(state, psutil):
    record_path = state/'process.json'
    if not record_path.exists():
        return
    previous = json.loads(record_path.read_text())
    try:
        process = psutil.Process(previous['pid'])
        if process.create_time() == previous['create_time'] and process.status() != psutil.STATUS_ZOMBIE:
            raise RuntimeError('This state directory already owns a live model service')
    except psutil.NoSuchProcess:
        pass


def launch(args):
    validate_options(args.gpu, args.concurrency, args.memory_utilization, args.port)
    if not args.state_dir.is_absolute():
        raise ValueError('State directory must be an absolute, independent run path')
    root, state = args.repo.resolve(), args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    import psutil
    # The launch lock serializes the check/start/record transaction, not the service lifetime.
    with (state/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refuse_live_record(state, psutil)
        verify_records(root)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        gpu = validate_gpu(subprocess.check_output([
            'nvidia-smi', f'--id={args.gpu}',
            '--query-gpu=uuid,name,memory.total,memory.free',
            '--format=csv,noheader,nounits'], text=True).strip(), args.memory_utilization)
        foreign_before = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
            '--format=csv'], text=True)
        # Unix-domain sockets append a UUID; the long run path exceeds sun_path.
        temporary_dir = Path(tempfile.mkdtemp(prefix=f"g4-{os.getuid()}-", dir="/tmp"))
        environment = launch_environment(os.environ, state, args.gpu, temporary_dir)
        for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'TORCHINDUCTOR_CACHE_DIR',
                    'TRITON_CACHE_DIR', 'VLLM_CACHE_ROOT'):
            Path(environment[key]).mkdir(parents=True, exist_ok=True)
        os.environ.update(environment)
        import torch
        validate_uuid(gpu['uuid'], str(torch.cuda.get_device_properties(0).uuid))
        from adk_submission import VllmConfig, VllmServer
        server = VllmServer(VllmConfig(**config_kwargs(
            root, args.port, args.concurrency, args.memory_utilization)))
        command = server.build_cmd()
        with (state/'vllm.log').open('a') as log:
            process = subprocess.Popen(command, env=server.build_env(), stdout=log, stderr=log,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        record = {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
                  'uid': os.getuid(), 'gpu_uuid': gpu['uuid'], 'gpu_inventory': gpu,
                  'physical_gpu_index_at_start': args.gpu, 'port': args.port, 'command': command,
                  'memory_utilization': args.memory_utilization, 'max_num_seqs': args.concurrency,
                  'enforce_eager': True, 'temporary_dir': str(temporary_dir), 'foreign_gpu_processes_before': foreign_before}
        temporary = state/'process.json.tmp'
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(state/'process.json')
        print(json.dumps(record))
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--memory-utilization', type=float, default=.65)
    launch(parser.parse_args())


if __name__ == '__main__':
    main()
