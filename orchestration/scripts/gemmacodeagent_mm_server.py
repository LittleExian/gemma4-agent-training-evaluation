"""Isolated four-sequence multimodel service and recorded compatibility gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

from gemmacodeagent_full_server import (launch_environment, refuse_live_record,
    validate_gpu, validate_uuid, validate_options as validate_base)


def validate_options(gpu, slots, memory, port):
    validate_base(gpu, slots, memory, port)
    if slots != 4:
        raise ValueError('This experiment requires exactly four sequences per physical GPU')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def remap_key(key):
    prefix = 'base_model.model.model.layers.'
    if not key.startswith(prefix):
        raise ValueError('Unexpected LoRA key prefix: ' + key)
    return 'base_model.model.model.language_model.layers.' + key[len(prefix):]


def stage_adapter(source, state):
    from safetensors import safe_open
    from safetensors.torch import save_file
    destination = state/'adapter-vllm'
    destination.mkdir(exist_ok=False)
    config = json.loads((source/'adapter_config.json').read_text())
    rank = config.get('r')
    if not isinstance(rank, int) or not 0 < rank <= 512:
        raise ValueError('Unsupported LoRA rank')
    tensors, mapping = {}, {}
    with safe_open(source/'adapter_model.safetensors', framework='pt', device='cpu') as f:
        for key in f.keys():
            mapped = remap_key(key)
            tensor = f.get_tensor(key)
            if str(tensor.dtype) != 'torch.bfloat16':
                raise ValueError('Expected published BF16 LoRA; refusing dtype conversion')
            tensors[mapped] = tensor
            mapping[key] = mapped
    if not tensors:
        raise ValueError('Empty adapter')
    save_file(tensors, str(destination/'adapter_model.safetensors'))
    (destination/'adapter_config.json').write_text(json.dumps(config, indent=2))
    with safe_open(destination/'adapter_model.safetensors', framework='pt', device='cpu') as f:
        if set(f.keys()) != set(tensors) or not all(f.get_tensor(k).equal(v) for k, v in tensors.items()):
            raise RuntimeError('Adapter remapping changed tensor values')
    save_json(state/'adapter_mapping.json', {'mapping': mapping, 'bitwise_equal': True,
        'source_sha256': digest(source/'adapter_model.safetensors'),
        'mapped_sha256': digest(destination/'adapter_model.safetensors')})
    return destination, rank


def validate_top_k(value):
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError('vLLM top_k must be a positive integer')
    return value


def positive_top_k(value):
    try:
        return validate_top_k(int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_command(root, model, arm, spec, port, memory, adapter=None, rank=None, top_k=None):
    validate_top_k(top_k)
    is_gemma = arm.startswith('gemma4_')
    parser = 'capybara_budget' if arm == 'gemma4_capybara' else ('gemma4' if is_gemma else 'qwen3')
    command = [str(root/'envs/eval/bin/python'), '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(model), '--served-model-name', 'qwen35_base' if arm == 'qwen35_swegym' else arm,
        '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
        '--max-model-len', '32768', '--max-num-seqs', '4', '--max-num-batched-tokens', '4096',
        '--tensor-parallel-size', '1', '--gpu-memory-utilization', str(memory), '--enforce-eager',
        '--reasoning-parser', parser, '--tool-call-parser', 'gemma4' if is_gemma else 'qwen3_coder',
        '--enable-auto-tool-choice', '--reasoning-config',
        json.dumps({'reasoning_start_str': '<|channel>' if is_gemma else '<think>',
                    'reasoning_end_str': '<channel|>' if is_gemma else '</think>'}),
        '--no-scheduler-reserve-full-isl']
    if arm != 'gemma4_kaggle':
        command += ['--generation-config', 'vllm']
    if arm == 'gemma4_capybara':
        command += ['--reasoning-parser-plugin', str(root/'src/gemmacodeagent_capybara_parser.py')]
    if is_gemma:
        command += ['--limit-mm-per-prompt', '{"image":0,"audio":0,"video":0}']
    else:
        command += ['--language-model-only']
    if spec['quantization'] in ('fp8', 'shared_fp8_base_plus_bf16_lora'):
        command += ['--quantization', 'fp8']
    if adapter is not None:
        command += ['--enable-lora', '--max-lora-rank', str(rank), '--lora-dtype', 'bfloat16',
                    '--lora-modules', 'qwen35_swegym=' + str(adapter)]
    if top_k is not None:
        command += ['--override-generation-config', json.dumps({'top_k': top_k})]
    return command


def prepare_template(model, arm, state):
    if arm != 'gemma4_capybara':
        return None
    from gemmacodeagent_capybara_template import adapt_template
    source = model/'chat_template.jinja'
    destination = state/'capybara-budget-template.jinja'
    destination.write_text(adapt_template(source.read_text()))
    save_json(state/'serving-adaptation.json', {'arm': arm,
        'reason': 'Make enabled thought channel explicit for token-budget accounting',
        'source_sha256': digest(source), 'override_sha256': digest(destination),
        'weights_quantization_sampling': 'unchanged'})
    return destination


def launch(args):
    validate_options(args.gpu, args.slots, args.memory_utilization, args.port)
    top_k = validate_top_k(getattr(args, 'vllm_topk', None))
    for value in (args.root, args.model, args.state, args.models_json):
        if not value.is_absolute():
            raise ValueError('All input and state paths must be absolute')
    root, model, state = args.root.resolve(), args.model.resolve(), args.state.resolve()
    specs = json.loads(args.models_json.read_text())
    if args.arm not in specs:
        raise ValueError('Unknown experiment arm')
    config_path = model/'config.json'
    if not config_path.is_file():
        raise ValueError('Complete local model config is required')
    config = json.loads(config_path.read_text())
    quant = specs[args.arm]['quantization']
    if quant == 'published_fp8' and 'fp8' not in json.dumps(config.get('quantization_config', {})).lower():
        raise ValueError('Published FP8 checkpoint lacks its quantization metadata')
    if quant == 'original_kaggle_w4a16' and 'quantization_config' not in config:
        raise ValueError('Original Kaggle checkpoint lacks quantization metadata')
    shared = args.arm in ('qwen35_base', 'qwen35_swegym')
    if shared and (args.adapter is None or not args.adapter.is_absolute()):
        raise ValueError('Shared Qwen35 service requires an absolute adapter directory')
    state.mkdir(parents=True, exist_ok=True)
    import psutil
    with (state/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refuse_live_record(state, psutil)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        gpu = validate_gpu(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
            '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).strip(), args.memory_utilization)
        foreign = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv'], text=True)
        temporary = Path(tempfile.mkdtemp(prefix=f'mm-{os.getuid()}-', dir='/tmp'))
        environment = launch_environment(os.environ, state, args.gpu, temporary)
        environment['TOKENIZERS_PARALLELISM'] = 'false'
        for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'VLLM_CACHE_ROOT'):
            Path(environment[key]).mkdir(parents=True, exist_ok=True)
        # No torch import may precede selecting the physical CUDA device.
        os.environ.update(environment)
        import torch
        validate_uuid(gpu['uuid'], str(torch.cuda.get_device_properties(0).uuid))
        adapter, rank = stage_adapter(args.adapter.resolve(), state) if shared else (None, None)
        command = build_command(root, model, args.arm, specs[args.arm], args.port, args.memory_utilization, adapter, rank, top_k=top_k)
        template = prepare_template(model, args.arm, state)
        if template is not None:
            command += ['--chat-template', str(template)]
        models = ['qwen35_base', 'qwen35_swegym'] if shared else [args.arm]
        provenance = {'arm': args.arm, 'model_path': str(model), 'model_config_sha256': digest(config_path),
            'models_json': str(args.models_json.resolve()), 'models_json_sha256': digest(args.models_json),
            'spec': specs[args.arm], 'adapter_source': str(args.adapter.resolve()) if shared else None}
        save_json(state/'launch-inputs.json', provenance)
        with (state/'vllm.log').open('a') as log:
            launch_boottime = time.clock_gettime(time.CLOCK_BOOTTIME) if hasattr(time, 'CLOCK_BOOTTIME') else None
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        record = {**provenance, 'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
            'uid': os.getuid(), 'gpu_uuid': gpu['uuid'], 'gpu_inventory': gpu, 'physical_gpu_index_at_start': args.gpu,
            'port': args.port, 'models': models, 'command': command, 'temporary_dir': str(temporary),
            'foreign_gpu_processes_before': foreign, 'memory_utilization': args.memory_utilization, 'max_num_seqs': 4,
            'launch_boottime': launch_boottime}
        save_json(state/'process.json', record)
        save_json(state/'routes.json', {name: {'model': name, 'port': args.port} for name in models})
        return record


def request_json(url, body=None, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_identity(data, expected):
    actual = {entry['id'] for entry in data.get('data', [])}
    if not set(expected).issubset(actual):
        raise RuntimeError(f'Requested model names missing: expected={expected}, actual={sorted(actual)}')


def message(data):
    return data['choices'][0]['message']


def validate_tool(data):
    calls = message(data).get('tool_calls', [])
    if not calls:
        raise RuntimeError('Tool parser produced no call')
    for call in calls:
        function = call.get('function', {})
        arguments = json.loads(function.get('arguments', '{}'))
        if call.get('type') != 'function' or function.get('name') != 'run_command' or not isinstance(arguments, dict) or set(arguments) != {'command'} or not isinstance(arguments['command'], str) or not arguments['command'].strip():
            raise RuntimeError('Tool name or arguments failed the declared JSON schema')


def validate_thinking(data, tokenizer):
    msg = message(data)
    reason = msg.get('reasoning') or msg.get('reasoning_content') or ''
    count = len(tokenizer.encode(reason, add_special_tokens=False))
    if not 0 < count <= 24 or not msg.get('content'):
        raise RuntimeError(f'Thinking budget failed: tokens={count}, content={bool(msg.get("content"))}')
    return count


def verify_service_identity(record):
    import psutil
    try:
        process = psutil.Process(record['pid'])
        if (process.create_time() != record['create_time'] or
                process.uids().real != record['uid'] or record['uid'] != os.getuid() or
                not process.is_running() or process.status() == psutil.STATUS_ZOMBIE):
            raise RuntimeError('Model service PID/create_time/UID identity changed')
    except psutil.NoSuchProcess as error:
        raise RuntimeError('Recorded model service is no longer alive') from error


def health(state, tokenizer=None, timeout=1200):
    record = json.loads((state/'process.json').read_text())
    url = f'http://127.0.0.1:{record["port"]}/v1'
    rows, evidence_lock = [], threading.Lock()
    report = {'passed': False, 'concurrency': 4, 'models': record['models'], 'rows': rows}
    def req(model, body):
        payload = {**body, 'model': model}
        started = time.monotonic()
        try:
            data = request_json(url+'/chat/completions', payload)
            row = {'request': payload, 'response': data, 'seconds': time.monotonic()-started}
        except Exception as error:
            with evidence_lock:
                rows.append({'request': payload, 'error': repr(error)})
            raise
        with evidence_lock:
            rows.append(row)
        return data
    try:
        verify_service_identity(record)
        deadline = time.monotonic()+timeout
        while True:
            try:
                inventory = request_json(url+'/models', timeout=10)
                check_identity(inventory, record['models'])
                report['inventory'] = inventory
                break
            except (urllib.error.URLError, OSError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Model readiness timeout')
                time.sleep(min(5, max(0, deadline-time.monotonic())))
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(record['model_path'], local_files_only=True, trust_remote_code=False)
        tool = {'messages': [{'role': 'user', 'content': 'Use run_command to execute pwd.'}],
            'tools': [{'type': 'function', 'function': {'name': 'run_command', 'description': 'Run a shell command',
                'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command'], 'additionalProperties': False}}}],
            'temperature': .2, 'top_p': .95, 'max_tokens': 1024, 'thinking_token_budget': 6000,
            'chat_template_kwargs': {'enable_thinking': True}}
        for model in record['models']:
            validate_tool(req(model, tool))
        thinking = {'messages': [{'role': 'user', 'content': 'Prove that there are infinitely many primes, with detailed reasoning.'}],
            'max_tokens': 256, 'temperature': 0, 'thinking_token_budget': 16, 'chat_template_kwargs': {'enable_thinking': True}}
        report['thinking_tokens'] = {model: validate_thinking(req(model, thinking), tokenizer) for model in record['models']}
        if len(record['models']) == 2:
            prompt = {'messages': [{'role': 'user', 'content': 'Write a Python binary search.'}],
                'max_tokens': 8, 'temperature': 0, 'logprobs': True, 'top_logprobs': 5, 'chat_template_kwargs': {'enable_thinking': False}}
            outputs = [req(model, prompt) for model in record['models']]
            probabilities = [item['choices'][0].get('logprobs') for item in outputs]
            if None in probabilities or probabilities[0] == probabilities[1]:
                raise RuntimeError('Shared service LoRA activation not demonstrated')
            report['lora_logits_differ'] = True
        prompt = {'messages': [{'role': 'user', 'content': ('def f(x): return x+1\n'*900)+'\nExplain the function.'}],
            'max_tokens': 128, 'temperature': .2, 'top_p': .95, 'thinking_token_budget': 6000, 'chat_template_kwargs': {'enable_thinking': True}}
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(req, record['models'][i % len(record['models'])], prompt) for i in range(4)]
            outputs = [future.result() for future in futures]
        for output in outputs:
            msg = message(output)
            if not (msg.get('content') or msg.get('reasoning') or msg.get('reasoning_content')):
                raise RuntimeError('Concurrent request produced empty output')
        seconds = time.monotonic()-started
        verify_service_identity(record)
        report.update(passed=True, seconds=seconds, tokens_per_second=sum(item['usage']['completion_tokens'] for item in outputs)/seconds)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['finished'] = time.time()
        save_json(state/'health.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    launch_parser = actions.add_parser('launch')
    for flag in ('root', 'models-json', 'model', 'state'):
        launch_parser.add_argument('--'+flag, type=Path, required=True)
    launch_parser.add_argument('--adapter', type=Path)
    launch_parser.add_argument('--arm', required=True)
    launch_parser.add_argument('--gpu', type=int, required=True)
    launch_parser.add_argument('--port', type=int, required=True)
    launch_parser.add_argument('--slots', type=int, default=4)
    launch_parser.add_argument('--memory-utilization', type=float, default=.65)
    launch_parser.add_argument('--vllm-topk', type=positive_top_k)
    health_parser = actions.add_parser('health')
    health_parser.add_argument('--state', type=Path, required=True)
    health_parser.add_argument('--timeout', type=float, default=1200)
    args = parser.parse_args()
    print(json.dumps(launch(args) if args.action == 'launch' else health(args.state, timeout=args.timeout)))


if __name__ == '__main__':
    main()
