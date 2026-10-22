"""CPU-only contract smoke: fake OpenAI -> official ADK -> Podman -> fresh tests.

This is not a model evaluation. The fake backend never repairs repository code.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
from pathlib import Path
import sys
import threading
import time
import uuid


class FakeOpenAI(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        steps = [('get_code_neighbors', {'node': 'coverage.version._make_url'}),
                 ('search_similar_code', {'query': '_make_url'}),
                 ('get_code_subgraph', {'nodes': ['coverage.version._make_url']}),
                 ('submit_patch', {})]
        index = len(self.server.requests) - 1
        if index < len(steps):
            name, arguments = steps[index]
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'cpu-contract-' + str(index), 'type': 'function',
                 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
            finish = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': 'Complete.'}
            finish = 'stop'
        response = {'id': 'cpu-contract-' + str(index), 'object': 'chat.completion',
                    'created': int(time.time()), 'model': body.get('model'),
                    'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}}
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'] = [dict(delta['tool_calls'][0], index=0)]
            response.update(object='chat.completion.chunk')
            response['choices'] = [{'index': 0, 'delta': delta, 'finish_reason': None}]
            last = dict(response, choices=[{'index': 0, 'delta': {}, 'finish_reason': finish}])
            payload = ('data: ' + json.dumps(response) + '\n\ndata: ' + json.dumps(last)
                       + '\n\ndata: [DONE]\n\n').encode()
            content_type = 'text/event-stream'
        else:
            payload = json.dumps(response).encode()
            content_type = 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/workspace/gemma4-lora246-validation-20261021'))
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'src'))
    import gemmacodeagent_validation20_task as task_module
    from swegemma.harness.agent_runner import run_agent_sandbox
    # Bind the actual installed ABI, including keyword-only task indices.
    inspect.signature(run_agent_sandbox).bind(None, None, None, root/'data/sentinel',
                                            task_index=1, total_tasks=1)
    task_id = 'r2e_coveragepy-05fbe9c95352'
    runtime = root/'runtime-probe/r2e-calibration-v3'/task_id/'task-runtime.json'
    assert json.loads(runtime.read_text())['passed'] is True
    mini = root/'cpu-contract-smoke'/('run-' + uuid.uuid4().hex[:12])
    (mini/'data').mkdir(parents=True)
    (mini/'reports').mkdir()
    (mini/'src').symlink_to(root/'src', target_is_directory=True)
    records = [json.loads(line) for line in (root/'data/tasks.jsonl').read_text().splitlines()]
    public = next(row for row in records if row['instance_id'] == task_id)
    (mini/'data/tasks.jsonl').write_text(json.dumps(public) + '\n')
    (mini/'data/selection.json').write_text(json.dumps({'tasks': [task_id]}))
    gate = {'passed': True, 'fake_backend_contract_fixture': True,
            'dataset_sha256': digest(mini/'data/tasks.jsonl'),
            'selection_sha256': digest(mini/'data/selection.json'),
            'tasks': {task_id: {'eligible': True, 'gold_passed': True,
                                'empty_passed': False, 'runtime_path': str(runtime)}},
            'artifact_sha256': {str(runtime): digest(runtime)}}
    (mini/'reports/cpu-gate.json').write_text(json.dumps(gate, indent=2))
    server = ThreadingHTTPServer(('127.0.0.1', 0), FakeOpenAI)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = mini/'task-output'
    previous = sys.argv
    try:
        sys.argv = ['cpu-contract-task', '--root', str(mini), '--task-id', task_id,
                    '--endpoint', f'http://127.0.0.1:{server.server_port}/v1',
                    '--output', str(output)]
        exit_code = task_module.main()
    finally:
        sys.argv = previous
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    (mini/'fake-openai-requests.json').write_text(json.dumps(server.requests, indent=2))
    result = json.loads((output/'result.json').read_text())
    assert exit_code == 0, result
    assert result['status'] == 'completed' and result['infra_error'] is False, result
    assert result['resolved'] is False, 'An empty fake repair must not pass real hidden tests'
    assert result['verification']['expected_tests_covered'] is True, result
    assert result['verification']['passed'] == 2 and result['verification']['failed'] == 1, result
    assert (output/'agent.patch').read_text() == '', 'Fake backend must never repair code'
    assert len(server.requests) >= 5, 'Real official tools/model loop did not execute'
    # ADK LiteLlm currently serializes responses as tool_responses, while native
    # OpenAI clients use tool. Check the actual tool payload, not one wire role.
    tools = [json.loads(message['content'])
             for request in server.requests for message in request.get('messages', [])
             if message.get('role') in {'tool', 'tool_responses'}]
    observed_errors = {item.get('error_type') for item in tools}
    assert {'GraphLookupError', 'SimilaritySearchError', 'SubgraphExtractionError'} <= observed_errors, tools
    assert any(item.get('status') == 'ok' and item.get('patch_size') == 0
               for item in tools), 'Official submit_patch acknowledgement was not observed'
    report = {'passed': True, 'fake_backend': True, 'counts_as_model_evaluation': False,
              'official_runner_keyword_abi': True, 'real_podman_sandbox': True,
              'fresh_hidden_tests': True, 'empty_patch_unresolved': True,
              'graph_embedding_unavailable_fallback_exercised': True,
              'requests': len(server.requests), 'task_result': result}
    (mini/'CPU_CONTRACT_PASS.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'passed': True, 'fake_backend': True, 'report': str(mini/'CPU_CONTRACT_PASS.json')}))


if __name__ == '__main__':
    main()
