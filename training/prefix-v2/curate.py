"""Audit decisions against tool feedback without rewriting history.
Failed shell/tests are retained. Compaction is separately controllable.
"""
import argparse, collections, gzip, hashlib, json
from pathlib import Path
SUMMARY_PREFIX = 'The following is a conversation history between a user and an AI agent. Please summarize the conversation'
INVALID_ERRORS = {'FileEditError', 'BudgetExceeded'}

def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def request_kind(row):
    messages = row['messages']
    if (not row['tools'] and len(messages) == 1
            and messages[0].get('role') == 'user'
            and isinstance(messages[0].get('content'), str)
            and messages[0]['content'].startswith(SUMMARY_PREFIX)):
        return 'compaction'
    return 'agent'

def feedback_index(path):
    index = {}; count = 0
    for row in read_rows(path):
        count += 1
        for message in row['messages']:
            if message.get('role') != 'tool': continue
            key = (row['attempt_id'], message['tool_call_id'])
            value = message.get('content', '')
            if key in index and index[key] != value:
                raise ValueError(f'Conflicting tool feedback: {key!r}')
            index[key] = value
    return index, count

def inspect(row, feedback):
    reasons, observations = [], []
    attempt = row['sample_id'].rsplit('/', 1)[0]
    for call in row['target'].get('tool_calls') or []:
        function = call['function']
        try:
            args = function['arguments']
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict): raise ValueError('arguments must be an object')
        except (ValueError, TypeError):
            reasons.append('malformed_arguments'); continue
        if (function['name'] == 'edit_file' and 'old_string' in args
                and 'new_string' in args and args['old_string'] == args['new_string']):
            reasons.append('identical_edit')
        raw = feedback.get((attempt, call.get('id')))
        obs = {'tool': function['name'], 'tool_call_id': call.get('id'), 'matched': raw is not None}
        if raw is not None:
            try: value = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, TypeError): value = None
            if isinstance(value, dict) and value.get('status') == 'error':
                obs['error_type'] = value.get('error_type', 'unknown')
                if obs['error_type'] in INVALID_ERRORS: reasons.append(obs['error_type'])
        observations.append(obs)
    if row.get('finish_reason') == 'length': reasons.append('truncated_target')
    return sorted(set(reasons)), observations

def curate(source, output, compaction='separate'):
    source, output = Path(source), Path(output)
    if output.exists(): raise FileExistsError('Use a new output directory; preserve past audits.')
    names = ['requests.jsonl.gz', 'attempts.jsonl.gz', 'train.jsonl', 'validation.jsonl']
    hashes = {name: digest(source / name) for name in names}
    feedback, requests = feedback_index(source / 'requests.jsonl.gz')
    attempts = {}
    for row in read_rows(source / 'attempts.jsonl.gz'):
        if row['attempt_id'] in attempts: raise ValueError('Duplicate attempt id')
        attempts[row['attempt_id']] = row
    tasks = {s: {r['task_id'] for r in read_rows(source / (s + '.jsonl'))}
             for s in ['train', 'validation']}
    if tasks['train'] & tasks['validation']: raise ValueError('Task leakage between splits')
    output.mkdir(parents=True)
    report = {'schema_version': 1, 'source_sha256': hashes, 'request_records': requests,
              'feedback_keys': len(feedback), 'compaction_policy': compaction, 'splits': {}}
    for split in ['train', 'validation']:
        counts, errors, kinds, task_rows = (collections.Counter() for _ in range(4))
        seen = set()
        with open(output / (split + '.jsonl'), 'w') as kept, \
             open(output / (split + '.compaction.jsonl'), 'w') as summaries, \
             open(output / (split + '.audit.jsonl'), 'w') as audit:
            for row in read_rows(source / (split + '.jsonl')):
                sid = row['sample_id']
                if sid in seen: raise ValueError(f'Duplicate sample id: {sid}')
                seen.add(sid)
                attempt = attempts[sid.rsplit('/', 1)[0]]
                if (attempt['task_id'] != row['task_id'] or attempt['training_split'] != split
                        or not all(attempt[k] is True for k in
                                   ['selected_success', 'test_verified_success', 'environment_valid', 'gold_passed', 'empty_failed'])):
                    raise ValueError(f'Unverified or incorrectly split sample: {sid}')
                kind = request_kind(row)
                reasons, observations = inspect(row, feedback)
                kinds[kind] += 1; counts['input_rows'] += 1
                for obs in observations:
                    counts['tool_calls'] += 1
                    counts['unmatched_tool_calls'] += not obs['matched']
                    if 'error_type' in obs: errors[obs['error_type']] += 1
                destination = 'quarantine' if reasons else ('compaction' if kind == 'compaction' and compaction == 'separate' else 'train')
                audit.write(json.dumps({'sample_id': sid, 'request_kind': kind, 'destination': destination,
                                        'reasons': reasons, 'feedback': observations}) + '\n')
                if reasons:
                    counts['quarantined_rows'] += 1; continue
                # Keep original history, including mistakes preceding successful recovery.
                enriched = dict(row, training_metadata={'request_kind': kind,
                    'curation_version': 1, 'verified_success': True})
                if destination == 'compaction':
                    summaries.write(json.dumps(enriched) + '\n'); counts['separate_compaction_rows'] += 1
                else:
                    kept.write(json.dumps(enriched) + '\n'); counts['kept_rows'] += 1
                    task_rows[row['task_id']] += 1
        report['splits'][split] = {'counts': dict(counts), 'observed_errors': dict(errors),
                                  'request_kinds': dict(kinds), 'kept_rows_per_task': dict(task_rows)}
    report['output_sha256'] = {p.name: digest(p) for p in output.glob('*.jsonl')}
    (output / 'audit_summary.json').write_text(json.dumps(report, indent=2))
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--compaction', choices=['separate', 'keep'], default='separate')
    args = parser.parse_args()
    print(json.dumps(curate(args.data, args.output, args.compaction), indent=2))

if __name__ == '__main__': main()
