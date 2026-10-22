"""Run three calibrated upstream jobs and release only our recorded model process."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def release_owned_model(record, started):
    import psutil
    report = {'terminated': [], 'killed': []}
    if record['create_time'] < started:
        raise RuntimeError('Refusing cleanup of a model service predating this driver')
    try:
        root = psutil.Process(record['pid'])
        if root.create_time() != record['create_time'] or root.uids().real != os.getuid():
            raise RuntimeError('Recorded model process identity changed')
        targets = [(root, record['create_time'])]
        targets.extend((p, p.create_time()) for p in root.children(recursive=True)
                       if p.uids().real == os.getuid())
    except psutil.NoSuchProcess:
        return report
    for process, created in targets:
        try:
            if process.create_time() == created and process.uids().real == os.getuid():
                process.terminate()
                report['terminated'].append(process.pid)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([p for p, _ in targets], timeout=15)
    identities = {p.pid: created for p, created in targets}
    for process in alive:
        try:
            if process.create_time() == identities[process.pid] and process.uids().real == os.getuid():
                process.kill()
                report['killed'].append(process.pid)
        except psutil.NoSuchProcess:
            pass
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run', default='mario-smoke3-hardened1')
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not args.run or not all(c.isalnum() or c in '-_' for c in args.run):
        raise ValueError('Invalid run name')
    started = time.time()
    lock = (repo/'runs/smoke-driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ids = ['rich_3506', 'httpx_3672', 'fastapi_13786']
    controls = json.loads((repo/'runs'/f'{args.run}-controls.json').read_text())
    if {r['task_id'] for r in controls if r['calibrated']} != set(ids):
        raise RuntimeError('All three fresh reference controls must pass before GPU launch')
    adapter_sha = hashlib.sha256((repo/'src/gemmacodeagent_mario_adapter.py').read_bytes()).hexdigest()
    if any(r.get('adapter_sha256') != adapter_sha for r in controls):
        raise RuntimeError('Reference controls must use the current sandbox adapter')
    native_sha = hashlib.sha256((repo/'src/gemmacodeagent_native_guard.py').read_bytes()).hexdigest()
    if any(r.get('native_guard_sha256') != native_sha for r in controls):
        raise RuntimeError('Reference controls must use the current native-file guard')
    python = str(repo/'envs/eval/bin/python')
    subprocess.run([python,'-m','pip','check'],check=True)
    subprocess.run([python,str(repo/'scripts/verify_environment.py')],check=True)
    environment = {**os.environ,
        'GEMMA_BASE':'/workspace/gemma4-notebook-run-20261008',
        'GEMMA_PREFLIGHT_GATE':'/workspace/gemma4-env-preflight-20261016/final-gate.json',
        'GEMMA_EVAL_RUNTIME':str(repo/'envs/eval'), 'GEMMA_REPO':str(repo),
        'GEMMA_SLOT_DIR':str(repo/'runs/local-slots'), 'GEMMA_EVAL_SLOTS':'1',
        'ADP_CONCURRENCY':'1', 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
        'MKL_NUM_THREADS':'2', 'PYTHONNOUSERSITE':'1',
        'GEMMA_RUNTIME_PROFILE':'mario-landlock-pro6000-smoke35-eager'}
    (repo/'runs/smoke-gpu-before.txt').write_text(subprocess.check_output([
        'nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True))
    status = {'started_unix':started, 'status':'starting_model', 'task_ids':ids,'run_name':args.run}
    state_path = repo/'runs/smoke-driver-status.json'
    state_path.write_text(json.dumps(status,indent=2))
    try:
        subprocess.run([python,str(repo/'scripts/mario_smoke_server.py'),
                        '--repo',str(repo),'--gpu','0','--port','8106'],env=environment,check=True)
        import requests
        for attempt in range(120):
            record = json.loads((repo/'runs/smoke-model/process.json').read_text())
            import psutil
            process = psutil.Process(record['pid'])
            if process.create_time() != record['create_time'] or process.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError('Model server exited during startup; inspect vllm.log')
            try:
                response = requests.get('http://127.0.0.1:8106/v1/models',timeout=3)
                if response.status_code == 200:
                    (repo/'runs/smoke-model/health.json').write_text(json.dumps(response.json(),indent=2))
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise RuntimeError('Model health check did not succeed within startup budget')
        status.update(status='evaluating',model_ready_unix=time.time())
        state_path.write_text(json.dumps(status,indent=2))
        command = [python,str(repo/'src/benchmark.py'),'--cohort','valid55',
                   '--shards','1','--shard','0','--workers','1','--ids',*ids,
                   '--arms','v5_6000','--run',args.run]
        with (repo/'logs/smoke-benchmark.log').open('a') as log:
            subprocess.run(command,env={**environment,'CUDA_VISIBLE_DEVICES':''},
                           stdout=log,stderr=log,check=True)
        status.update(status='completed',finished_unix=time.time())
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
        raise
    finally:
        process_record = repo/'runs/smoke-model/process.json'
        if process_record.exists():
            record = json.loads(process_record.read_text())
            if record['create_time'] >= started:
                status['model_cleanup'] = release_owned_model(record,started)
        status['gpu_after'] = subprocess.check_output([
            'nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
        state_path.write_text(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
