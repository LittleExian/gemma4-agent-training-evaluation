"""Validate non-streaming/streaming channels and tools on the repaired service."""
import argparse
import json
from pathlib import Path
import time
import requests


def streamed(url, payload):
    chunks, reasoning, content, tools = [], '', '', {}
    complete = False
    with requests.post(url, json={**payload, 'stream': True,
            'stream_options': {'include_usage': True}}, stream=True, timeout=180) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b'data:'): continue
            value = line[5:].strip()
            if value == b'[DONE]':
                complete = True
                break
            item = json.loads(value)
            chunks.append(item)
            for choice in item.get('choices', []):
                delta = choice.get('delta', {})
                reasoning += delta.get('reasoning') or delta.get('reasoning_content') or ''
                content += delta.get('content') or ''
                for call in delta.get('tool_calls') or []:
                    index = call['index']
                    saved = tools.setdefault(index, {'type': 'function', 'function': {'name': '', 'arguments': ''}})
                    function = call.get('function') or {}
                    saved['function']['name'] += function.get('name') or ''
                    saved['function']['arguments'] += function.get('arguments') or ''
    if not complete: raise RuntimeError('Stream ended without DONE')
    return {'choices': [{'message': {'reasoning': reasoning, 'content': content,
                                    'tool_calls': list(tools.values())}}], 'raw_chunks': chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    url = 'http://127.0.0.1:8140/v1'
    for _ in range(240):
        try:
            response = requests.get(url+'/models', timeout=3)
            if response.status_code == 200:
                assert 'gemma4_capybara' in [m['id'] for m in response.json()['data']]
                break
        except requests.RequestException: pass
        time.sleep(5)
    else: raise RuntimeError('Repaired service readiness timeout')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.root/'research/distributed_model_comparison/models/gemma4_capybara'), local_files_only=True, trust_remote_code=False)
    proof = {'model': 'gemma4_capybara', 'messages': [{'role': 'user', 'content':
        'Prove that there are infinitely many primes, with detailed reasoning.'}],
        'temperature': 0, 'max_tokens': 256, 'thinking_token_budget': 16,
        'chat_template_kwargs': {'enable_thinking': True}}
    tool = {**proof, 'messages': [{'role': 'user', 'content': 'Use run_command to execute pwd.'}],
        'temperature': .2, 'top_p': .95, 'max_tokens': 1024, 'thinking_token_budget': 6000,
        'tools': [{'type': 'function', 'function': {'name': 'run_command', 'description': 'Run a shell command',
            'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}},
                           'required': ['command'], 'additionalProperties': False}}}]}
    cases = [('nonstream16', proof, False), ('stream16', proof, True),
             ('nonstream6000unfinished', {**proof, 'max_tokens': 128, 'thinking_token_budget': 6000}, False),
             ('stream_tool', tool, True)]
    summary = {}
    for name, payload, stream in cases:
        if stream:
            data = streamed(url+'/chat/completions', payload)
        else:
            response = requests.post(url+'/chat/completions', json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
        (args.output/(name+'.json')).write_text(json.dumps({'request': payload,
            'stream': stream, 'response': data}, indent=2))
        message = data['choices'][0]['message']
        reason = message.get('reasoning') or message.get('reasoning_content') or ''
        count = len(tokenizer.encode(reason, add_special_tokens=False))
        if name.endswith('16'):
            assert 0 < count <= 24 and message.get('content'), (name, count, message)
        elif name == 'nonstream6000unfinished':
            assert count > 24 and not message.get('content'), (name, count, message)
        else:
            calls = message.get('tool_calls') or []
            assert calls and all(c['function']['name'] == 'run_command' and
                isinstance(json.loads(c['function']['arguments']).get('command'), str) for c in calls), calls
        summary[name] = {'passed': True, 'thinking_tokens': count,
                         'content_present': bool(message.get('content')), 'tool_calls': bool(message.get('tool_calls'))}
        print(name, summary[name], flush=True)
    (args.output/'summary.json').write_text(json.dumps({'passed': True, 'cases': summary}, indent=2))


if __name__ == '__main__': main()
