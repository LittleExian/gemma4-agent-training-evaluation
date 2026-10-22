"""Launch a bounded, separately logged model service for the Mario smoke test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--port', type=int, default=8106)
    args = parser.parse_args()
    root = args.repo.resolve()
    state = root/'runs/smoke-model'
    state.mkdir(exist_ok=True)
    lock = (state/'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import psutil
    if (state/'process.json').exists():
        previous = json.loads((state/'process.json').read_text())
        try:
            existing = psutil.Process(previous['pid'])
            if existing.create_time() == previous['create_time'] and existing.status() != psutil.STATUS_ZOMBIE:
                raise RuntimeError('The previously recorded smoke model service is still alive')
        except psutil.NoSuchProcess:
            pass
    if not json.loads((root/'runs/environment-verified.json').read_text()).get('official_six_source_hashes_match'):
        raise RuntimeError('Official runtime source verification is required')
    if not json.loads((root/'runs/model-sha256.json').read_text()).get('matches'):
        raise RuntimeError('Model checkpoint hash does not match upstream')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('Smoke-test port is already occupied')
    values = subprocess.check_output([
        'nvidia-smi', f'--id={args.gpu}',
        '--query-gpu=uuid,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    gpu_uuid, name, total, free = values
    if 'RTX PRO 6000' not in name or float(free) < float(total)*.35 + 8192:
        raise RuntimeError('GPU type or current free-memory margin does not satisfy the smoke budget')
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.gpu),
                      HF_HOME=str(root/'cache/huggingface'), XDG_CACHE_HOME=str(root/'cache'),
                      TMPDIR=str(root/'tmp'), OMP_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    import torch
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid.removeprefix('GPU-').lower() != gpu_uuid.removeprefix('GPU-').lower():
        raise RuntimeError('CUDA-visible device UUID does not match the selected physical GPU')
    from adk_submission import VllmConfig, VllmServer
    config = VllmConfig(
        model=str(root/'models/gemma-4-31b-it-qat-w4a16-ct'),
        host='127.0.0.1', port=args.port, tool_call_parser='gemma4',
        reasoning_parser='gemma4', max_model_len=32768, dtype='bfloat16',
        gpu_memory_utilization=.35, enable_auto_tool_choice=True, enable_lora=False,
        scheduler_reserve_full_isl=False, tensor_parallel_size=1,
        limit_mm_per_prompt='{"image":0,"audio":0,"video":0}',
        extra_args=['--served-model-name','gemma-4-31b-it-qat-w4a16-ct',
                    '--max-num-seqs','1','--max-num-batched-tokens','4096','--enforce-eager'])
    server = VllmServer(config)
    command = server.build_cmd()
    with (state/'vllm.log').open('a') as log:
        process = subprocess.Popen(command, env=server.build_env(), stdout=log,
                                   stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    record = {'pid':process.pid,'create_time':psutil.Process(process.pid).create_time(),
              'gpu_uuid':gpu_uuid,'physical_gpu_index_at_start':args.gpu,
              'free_mib_before_start':float(free),'memory_utilization':.35,
              'max_num_seqs':1,'enforce_eager':True,'port':args.port,'command':command,
              'condition':'Mario shared GPU, single-worker compatibility smoke; differs from upstream .90 chroot capacity experiment'}
    (state/'process.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'pid':process.pid,'gpu_uuid':gpu_uuid,'port':args.port}))


if __name__ == '__main__':
    main()
