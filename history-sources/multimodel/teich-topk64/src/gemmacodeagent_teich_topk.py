"""Fresh Teich full129 run with only the vLLM top-k sampling default changed."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import gemmacodeagent_mm_pipeline as pipeline

ARM = 'gemma4_teich_agent'


def sampling_override(command):
    for flag in ('--generation-config', '--override-generation-config'):
        if command.count(flag) != 1:
            raise ValueError('Missing or ambiguous sampling configuration')
    if command[command.index('--generation-config') + 1] != 'vllm':
        raise ValueError('Checkpoint sampling defaults must remain disabled')
    override = json.loads(command[command.index('--override-generation-config') + 1])
    if override != {'top_k': 64}:
        raise ValueError('Only top_k=64 may change')
    return override


def verify_sampling(record, state):
    """Exercise the installed server's default merge and chat-request conversion."""
    from vllm.config import ModelConfig
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    override = sampling_override(record['command'])
    config = SimpleNamespace(generation_config='vllm', override_generation_config=override)
    defaults = ModelConfig.get_diff_sampling_param(config)
    request = ChatCompletionRequest(model=ARM,
        messages=[{'role': 'user', 'content': 'test'}],
        temperature=.2, top_p=.95, max_tokens=8192, thinking_token_budget=6000,
        chat_template_kwargs={'enable_thinking': True})
    params = request.to_sampling_params(8192, defaults)
    if (params.top_k, params.temperature, params.top_p, params.max_tokens) != (64, .2, .95, 8192):
        raise RuntimeError('Installed chat sampling conversion disagrees with requested protocol')
    pipeline.atomic_json(state/'sampling-proof.json', {
        'passed': True, 'server_pid': record['pid'], 'defaults': defaults,
        'request_top_k': request.top_k, 'effective_top_k': params.top_k,
        'temperature': params.temperature, 'top_p': params.top_p,
        'max_tokens': params.max_tokens,
        'method': 'Recorded live server argv + installed ModelConfig and ChatCompletionRequest conversion'})


def start_models(root, experiment, cache, specs, directory, environment):
    models, started = [], time.time()
    try:
        for gpu, port in zip(pipeline.GPUS, pipeline.PORTS):
            state = directory/('gpu-'+str(gpu))
            state.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, str(root/'src/gemmacodeagent_mm_server.py'), 'launch',
                '--root', str(root), '--models-json', str(experiment/'models.json'),
                '--arm', ARM, '--model', str(cache/specs[ARM]['directory']),
                '--state', str(state), '--gpu', str(gpu), '--port', str(port),
                '--slots', '4', '--vllm-topk', '64']
            try:
                pipeline.checked_process(command, state/'launch.log', environment, models, timeout=180)
            finally:
                if (state/'process.json').exists():
                    models.append(json.loads((state/'process.json').read_text()))
            verify_sampling(models[-1], state)
            pipeline.checked_process([sys.executable, str(root/'src/gemmacodeagent_mm_server.py'),
                'health', '--state', str(state), '--timeout', '1200'],
                state/'health.log', environment, models, timeout=1800)
        return models, started
    except BaseException:
        pipeline.cleanup_secondary(models, started, directory/'cleanup-secondary.json')
        raise


def run(args):
    root, directory = args.root.resolve(), args.run_dir.resolve()
    if not directory.is_relative_to(root/'runs'):
        raise ValueError('Run directory must remain inside repository/runs')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory/'pipeline-status.json').exists():
            raise RuntimeError('Fresh rerun required; refusing to overwrite existing run')
        state = {'status': 'running', 'phase': 'preflight', 'pid': os.getpid(),
                 'started_unix': time.time(), 'arm': ARM, 'top_k': 64}
        def save():
            pipeline.atomic_json(directory/'pipeline-status.json', {**state, 'updated_unix': time.time()})
        save()
        models, started = [], time.time()
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
            subprocess.run([sys.executable, str(root/'scripts/verify_environment.py')], check=True)
            experiment = root/'experiments/distributed_model_comparison'
            ids = json.loads((experiment/'full129.json').read_text())['tasks']
            if len(ids) != 129 or len(set(ids)) != 129:
                raise ValueError('Full task scope changed')
            old = pipeline.build_protocol(root, args.base, args.gate, experiment)
            protocol = {**old, 'arms': [ARM], 'ports': list(pipeline.PORTS),
                'serving_adaptations': {}, 'protocol': {**old['protocol'], 'top_k': 64},
                'policy': 'Fresh full129 Teich rerun; existing model compatibility established in prior full run; fresh three-GPU health gates required',
                'comparison_run': str(args.previous_run.resolve())}
            pipeline.bind_json(directory/'protocol.json', protocol)
            environment = {**os.environ, 'PYTHONNOUSERSITE': '1', 'OTEL_SDK_DISABLED': 'true',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
            cache = (root/'research/distributed_model_comparison/models').resolve()
            state['phase'] = 'verifying_model_files'
            save()
            specs = pipeline.wait_downloads(root, experiment, cache, (ARM,), environment, directory)
            state['phase'] = 'model_health'
            save()
            models, started = start_models(root, experiment, cache, specs, directory/'services', environment)
            state.update(phase='evaluating', models=models)
            save()
            report = pipeline.evaluate(root, args.base, args.gate, ARM, 'full', ids,
                directory, models, environment, directory/'protocol.json')
            state.update(status='completed', phase='complete', result=report, finished_unix=time.time())
        except BaseException as error:
            state.update(status='blocked', phase='error', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            state['cleanup'] = pipeline.cleanup_secondary(models, started, directory/'cleanup-secondary.json')
            if state['status'] == 'completed' and any('cleanup_error' in item for item in state['cleanup']):
                state.update(status='blocked', phase='cleanup_error')
            state.pop('models', None)
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'run-dir', 'base', 'gate', 'previous-run'):
        parser.add_argument('--'+name, required=True, type=Path)
    run(parser.parse_args())
