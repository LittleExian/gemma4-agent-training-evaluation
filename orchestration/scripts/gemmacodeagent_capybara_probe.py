"""Record discriminating Capybara requests against an owned diagnostic endpoint."""
import argparse
import json
from pathlib import Path
import time
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    url = 'http://127.0.0.1:8150/v1'
    for _ in range(240):
        try:
            if requests.get(url+'/models', timeout=3).status_code == 200: break
        except requests.RequestException: pass
        time.sleep(5)
    else: raise RuntimeError('Diagnostic service never became ready')
    source = (args.root/'research/distributed_model_comparison/models/gemma4_capybara/chat_template.jinja').read_text()
    prefix = source + "\n{% if add_generation_prompt and enable_thinking %}{{ '<|channel>thought\\n' }}{% endif %}\n"
    base = {'model': 'gemma4_capybara', 'messages': [{'role': 'user', 'content':
        'Prove that there are infinitely many primes, with detailed reasoning.'}],
        'max_tokens': 256, 'temperature': 0, 'thinking_token_budget': 16,
        'chat_template_kwargs': {'enable_thinking': True}, 'return_token_ids': True,
        'skip_special_tokens': False}
    cases = [('original16', base), ('original6000', {**base, 'thinking_token_budget': 6000}),
             ('prefix16', {**base, 'chat_template': prefix}),
             ('continued16', {**base, 'messages': base['messages'] + [{'role': 'assistant',
                 'content': '<|channel>thought\n'}], 'add_generation_prompt': False,
                 'continue_final_message': True})]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in cases:
        started = time.time()
        response = requests.post(url+'/chat/completions', json=payload, timeout=180)
        data = response.json()
        (args.output/(name+'.json')).write_text(json.dumps({'request': payload, 'status': response.status_code,
            'response': data, 'seconds': time.time()-started}, indent=2))
        choice = data.get('choices', [{}])[0]
        msg = choice.get('message', {})
        print(json.dumps({'case': name, 'status': response.status_code,
            'reasoning': (msg.get('reasoning') or msg.get('reasoning_content') or '')[:300],
            'content': (msg.get('content') or '')[:300], 'finish_reason': choice.get('finish_reason'),
            'error': data.get('error')}), flush=True)


if __name__ == '__main__': main()
