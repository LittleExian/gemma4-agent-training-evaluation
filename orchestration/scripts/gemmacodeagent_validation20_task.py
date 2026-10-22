"""Official ADK agent loop, isolated calibrated Podman task, fresh hidden tests."""
import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.parse

DEFAULT_ROOT = Path('/workspace/gemma4-lora246-validation-20261021')
DEFAULT_SUBMISSION = Path('/workspace/gemma4-lora246-eval-20261021/submissions/mm15_6000')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    temporary.replace(path)


def validate_endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != '/v1' or not parsed.port or not 1024 <= parsed.port <= 65535):
        raise ValueError('Expected explicit loopback vLLM HTTP endpoint /v1')
    return value


def classify_agent_error(message):
    if not message:
        return {'kind': None, 'infra_error': False, 'timed_out': False}
    lower = str(message).lower()
    timed_out = 'session timeout' in lower or 'agent timed out' in lower
    model_markers = ('tool calls', 'tool_calls', 'tool budget', 'tool call budget',
                     'turns budget', 'turn budget', 'contextwindowexceedederror',
                     'maximum allowed llm turns', 'context_length', 'context length',
                     'maximum context', 'max tokens', 'token limit', 'session timeout',
                     'without calling submit_patch', 'without explicit submit_patch')
    model_error = timed_out or any(marker in lower for marker in model_markers)
    return {'kind': 'model_budget_or_output' if model_error else 'runtime_error',
            'infra_error': not model_error, 'timed_out': timed_out}


def load_checked_gate(root, task_id):
    gate = json.loads((root/'reports/cpu-gate.json').read_text())
    binding = gate.get('tasks', {}).get(task_id, {})
    if (gate.get('passed') is not True or binding.get('eligible') is not True
            or binding.get('gold_passed') is not True or binding.get('empty_passed') is not False):
        raise ValueError('Task lacks successful immutable CPU controls')
    for path, digest in gate.get('artifact_sha256', {}).items():
        if not Path(path).is_absolute() or sha(path) != digest:
            raise ValueError('Bound evaluation artifact changed: ' + str(path))
    for filename, field in [('selection.json', 'selection_sha256'),
                            ('tasks.jsonl', 'dataset_sha256')]:
        if field in gate and sha(root/'data'/filename) != gate[field]:
            raise ValueError('Public task selection/dataset changed')
    return gate


def final_result(task_id, agent_error, verification, runtime):
    required = ('test_exit_code', 'passed', 'failed', 'errors')
    if any(type(verification.get(key)) is not int for key in required):
        raise ValueError('Fresh verifier must report integer test result fields')
    if type(verification.get('expected_tests_covered')) is not bool:
        raise ValueError('Fresh verifier must report expected-test coverage')
    expected = runtime['results']['gold']['test_outcomes']
    actual = verification.get('test_outcomes', {})
    coverage = verification['expected_tests_covered'] and set(actual) == set(expected)
    states_match = bool(actual) and all(actual[key] == state for key, state in expected.items()
                                       if key in actual)
    classification = classify_agent_error(agent_error)
    infra = bool(verification.get('infra_error', verification.get('infra_failed', False)))
    infra = infra or classification['infra_error']
    resolved = (not infra and verification['test_exit_code'] == 0
                and verification['passed'] > 0 and verification['failed'] == 0
                and verification['errors'] == 0 and coverage and states_match)
    return {'instance_id': task_id,
            'status': 'infra_error' if infra else ('agent_timeout' if classification['timed_out'] else 'completed'),
            'resolved': resolved, 'normal_output': agent_error is None,
            'infra_error': infra, 'agent_error': agent_error,
            'agent_error_kind': classification['kind'],
            'verification': dict(verification, expected_tests_covered=coverage)}


@contextlib.contextmanager
def calibrated_setup(module):
    names = ('setup_container_wheels', 'extract_snapshot', 'install_editable_package',
             'install_test_dependencies', 'setup_workspace_test_config')
    originals = {name: getattr(module, name) for name in names}
    try:
        for name in names:
            setattr(module, name, lambda *args, **kwargs: None)
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def load_verifier(root, task_id):
    path = root/'src/gemmacodeagent_validation20_verify.py'
    spec = importlib.util.spec_from_file_location('validation20_fresh_verifier', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify_patch


async def agent_patch(root, output, task, runtime, endpoint, submission):
    from adk_submission import ModelRegistry, discover_adapters
    from adk_submission.resolvers.generation import install_litellm_reasoning_patch
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    import litellm
    import swegemma.harness.agent_runner as runner
    from swegemma.config import EvalConfig
    from gemmacodeagent_validation20_podman import PodmanManager

    install_litellm_reasoning_patch()
    litellm.drop_params = True
    models = ModelRegistry()
    arm = 'gemma4_w4_lora246'
    backend = LiteLlm(model='openai/' + arm, api_base=endpoint, api_key='EMPTY',
                      num_retries=3, extra_body={'thinking_token_budget': 6000,
                      'chat_template_kwargs': {'enable_thinking': True}})
    models.register('gemma-4-31b-it-qat-w4a16-ct', backend)
    for name in discover_adapters(submission).adapters:
        models.register('adapter:' + name, backend)
    for name in ('graphs', 'embeddings'):
        directory = root/'data'/name
        directory.mkdir(exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError('Held-out graph/embedding directory must remain empty')
    config = EvalConfig(tasks_path=root/'data/tasks.jsonl', snapshots_dir=root/'data',
                        results_dir=output, submission_dir=submission, models=models,
                        image=runtime['image'], sandbox='subprocess',
                        graph_dir=str(root/'data/graphs'), embeddings_dir=str(root/'data/embeddings'),
                        timeout_seconds=240, verification_timeout_seconds=300,
                        max_time_minutes=15, max_tool_calls=56, max_turns=160,
                        verbose=False, display_mode='quiet',
                        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800,
                                                               cache_intervals=10),
                        events_compaction_config=EventsCompactionConfig(compaction_interval=15,
                            overlap_size=2, token_threshold=14336, event_retention_size=5))
    manager = PodmanManager(runtime, output)
    tempfile.tempdir = str(manager.tmp)
    save(output/'agent-protocol.json', {'official_agent_runner_sha256': sha(runner.__file__),
        'minutes': 15, 'tools': 56, 'turns': 160, 'command_timeout': 240,
        'thinking': 6000, 'completion_tokens': 8192, 'compaction_threshold': 14336,
        'graph': 'unavailable; official tool fallback', 'embeddings': 'unavailable',
        'submission_files': {str(p.relative_to(submission)): sha(p)
                             for p in submission.rglob('*') if p.is_file()}})
    try:
        with calibrated_setup(runner):
            patch, error, trace = await runner.run_agent_sandbox(
                manager, config, task, root/'data/managed-image-snapshot',
                task_index=1, total_tasks=1)
        if trace:
            trace.save(output/'trace.json')
        return patch, error
    finally:
        manager.cleanup_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--submission-dir', type=Path,
                        default=Path(os.environ.get('SUBMISSION_DIR', str(DEFAULT_SUBMISSION))))
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', args.task_id) or '..' in args.task_id:
        raise ValueError('Unsafe task ID')
    if not output.is_relative_to(root) or output == root:
        raise ValueError('Output must remain under independent validation root')
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        endpoint = validate_endpoint(args.endpoint)
        gate = load_checked_gate(root, args.task_id)
        binding = gate['tasks'][args.task_id]
        runtime = json.loads(Path(binding['runtime_path']).read_text())
        rows = [json.loads(line) for line in (root/'data/tasks.jsonl').read_text().splitlines()]
        public = [row for row in rows if row['instance_id'] == args.task_id]
        if len(public) != 1 or public[0].get('patch') or public[0].get('test_patch'):
            raise ValueError('Agent task must be unique and contain no reference/test patch')
        from swegemma.models import Task
        task = Task(**public[0])
        temporary = output/'tmp'
        temporary.mkdir()
        tempfile.tempdir = str(temporary)
        os.environ.update(OTEL_SDK_DISABLED='true', LITELLM_LOCAL_MODEL_COST_MAP='True',
                          LITELLM_TELEMETRY='False', OPENAI_API_KEY='EMPTY')
        patch, error = asyncio.run(agent_patch(root, output, task, runtime, endpoint,
                                               args.submission_dir.resolve()))
        from swegemma.harness.verification import strip_protected_files_from_patch
        patch = strip_protected_files_from_patch(patch or '')
        patch_file = output/'agent.patch'
        patch_file.write_text(patch)
        verification = load_verifier(root, args.task_id)(runtime, patch_file, output/'verification')
        result = final_result(args.task_id, error, verification, runtime)
        result['patch_path'] = str(patch_file)
        result['gate_sha256'] = sha(root/'reports/cpu-gate.json')
    except Exception as error:
        import traceback
        (output/'infra-traceback.txt').write_text(traceback.format_exc())
        result = {'instance_id': args.task_id, 'status': 'infra_error', 'resolved': False,
                  'normal_output': False, 'infra_error': True, 'error': str(error),
                  'verification': {'test_exit_code': -1, 'passed': 0, 'failed': 0,
                                   'errors': 0, 'expected_tests_covered': False}}
    result['wall_seconds'] = time.time() - started
    save(output/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 2 if result['infra_error'] else 0


if __name__ == '__main__':
    sys.exit(main())
