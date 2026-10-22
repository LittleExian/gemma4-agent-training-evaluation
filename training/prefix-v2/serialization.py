"""Native targets conditioned on independently rendered deployment prompts."""
import copy
import hashlib
import json

SERIALIZATION_VERSION = 'gemma4-deployment-prefix-v2'
THOUGHT_OPEN = '<|channel>thought\n'
THOUGHT_CLOSE = '<channel|>'

def normalize(messages):
    messages = copy.deepcopy(messages)
    for m in messages:
        m['content'] = m.get('content') or ''
        for tc in m.get('tool_calls') or []:
            value = tc['function'].get('arguments')
            if isinstance(value, str):
                tc['function']['arguments'] = json.loads(value)
    return messages

def deployment_prompt(tokenizer, messages, tools):
    return tokenizer.apply_chat_template(
        normalize(messages), tools=tools, tokenize=False,
        enable_thinking=True, preserve_thinking=False, add_generation_prompt=True)

def render_example(tokenizer, row):
    messages = normalize(row['messages'])
    target = normalize([row['target']])[0]
    if target.get('role') != 'assistant':
        raise ValueError('Target must be an assistant message')
    reason = target.get('reasoning') or target.get('reasoning_content') or ''
    if not isinstance(reason, str):
        raise ValueError('Reasoning must be text')
    tools = row['tools']
    opts = dict(tools=tools, tokenize=False, enable_thinking=True,
                preserve_thinking=False, add_generation_prompt=False)
    marker = 'TARGET_BOUNDARY_61fe6086'
    if marker in json.dumps(messages + [target], ensure_ascii=False):
        raise ValueError('Boundary marker collision')
    dummy = dict(role='assistant', reasoning_content=marker, content='target')
    probe = tokenizer.apply_chat_template(messages + [dummy], **opts)
    anchor = THOUGHT_OPEN + marker + '\n' + THOUGHT_CLOSE
    if probe.count(anchor) != 1:
        raise ValueError('Unsupported native template target boundary')
    # Used ONLY to extract native target syntax; never used as training input.
    history = probe.split(anchor)[0]
    native = tokenizer.apply_chat_template(messages + [target], **opts)
    if not native.startswith(history):
        raise ValueError('Target-dependent history cannot be safely extracted')
    body = native[len(history):]
    prompt = deployment_prompt(tokenizer, messages, tools)
    prefills_thought = prompt.endswith(THOUGHT_OPEN)
    inserted_empty_thought = False
    if reason:
        expected = THOUGHT_OPEN + reason + '\n' + THOUGHT_CLOSE
        if not body.startswith(expected):
            raise ValueError('Native reasoning block does not match teacher text')
        completion = body[len(THOUGHT_OPEN):] if prefills_thought else body
        reason_start = 0 if prefills_thought else len(THOUGHT_OPEN)
        reason_end = reason_start + len(reason)
    else:
        if body.startswith(THOUGHT_OPEN):
            raise ValueError('Unrecognized embedded reasoning; supply structured reasoning')
        completion = (THOUGHT_CLOSE if prefills_thought else '') + body
        inserted_empty_thought = prefills_thought
        reason_start = reason_end = -1
    if not completion:
        raise ValueError('Empty assistant continuation')
    return dict(prompt=prompt, completion=completion,
                reason_start=reason_start, reason_end=reason_end,
                inserted_empty_thought=inserted_empty_thought,
                native_prefix_matched=native.startswith(prompt))

def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()

def validate_serialization(row):
    """Reject stale caches before model allocation; no legacy escape hatch."""
    if row.get('serialization_version') != SERIALIZATION_VERSION:
        raise ValueError('Legacy token cache: rerun prepare.py into a NEW directory')
    ids, weights = row['input_ids'], row['weights']
    p = row.get('prompt_tokens')
    if not isinstance(p, int) or not 0 < p < len(ids):
        raise ValueError('Invalid deployment prompt boundary')
    if len(ids) != len(weights) or any(w != 0 for w in weights[:p]):
        raise ValueError('Deployment prompt must be fully masked')
    if ids_hash(ids[:p]) != row.get('prompt_ids_sha256'):
        raise ValueError('Deployment prompt token hash mismatch')
