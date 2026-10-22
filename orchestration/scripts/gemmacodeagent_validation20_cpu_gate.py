"""Assemble strict CPU readiness evidence; never launches a model or GPU worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def controls_ready(runtime, gold, empty):
    expected = runtime['results']['gold']['test_outcomes']
    good, bad = gold.get('test_outcomes', {}), empty.get('test_outcomes', {})
    return (bool(expected) and good == expected and set(good) == set(bad)
            and gold.get('test_exit_code') == 0 and empty.get('test_exit_code') == 1
            and gold.get('resolved') is True and empty.get('resolved') is False
            and all(result.get('task_id') == runtime['task_id']
                    and result.get('infra_error') is False
                    and result.get('expected_tests_covered') is True
                    and result.get('hidden_payload_unchanged') is True
                    for result in (gold, empty))
            and not any(state in {'failed', 'error'} for state in good.values())
            and any(good[name] == 'passed' and bad[name] in {'failed', 'error'} for name in good))


def required_cases_passed(task_id, source, outcomes):
    if task_id.startswith('r2e_'):
        normalized = {}
        for name, state in outcomes.items():
            prefix = 'r2e_tests.test_1'
            short = name[len(prefix):].lstrip('.:') if name.startswith(prefix) else name
            base, bracket, parameters = short.partition('[')
            key = base.replace('::', '.') + bracket + parameters
            if key in normalized and normalized[key] != state:
                return False
            normalized[key] = state
        required = json.loads(source['expected_output_json'])
        return bool(required) and all(normalized.get(name) == state.lower()
                                      for name, state in required.items())
    required = source['FAIL_TO_PASS'] + source['PASS_TO_PASS']
    for name in required:
        matches = [key for key in outcomes if key == name or
                   ('[' in name and not name.endswith(']') and key.startswith(name))]
        if not matches or any(outcomes[key] != 'passed' for key in matches):
            return False
    return bool(required)


def sanitized_proof_ready(proof):
    return (proof.get('forbidden_objects_absent') is True
            and proof.get('other_git') == [] and proof.get('artifacts_remaining') == []
            and proof.get('history_count') == 1 and bool(proof.get('removed_git'))
            and bool(proof.get('original_objects_checked'))
            and bool(proof.get('source_tree_before'))
            and proof.get('source_tree_before') == proof.get('source_tree_after'))


def bind_file(bindings, path):
    path = Path(path).resolve(strict=True)
    bindings[str(path)] = sha(path)
    return path


def task_evidence(root, task_id, entry, bindings):
    runtime_path = bind_file(bindings, root/'data/runtimes'/(task_id+'.json'))
    runtime = read(runtime_path)
    source_path = bind_file(bindings, root/'source-records'/(task_id+'.source.json'))
    source = read(source_path)
    if runtime['source_sha256'] != sha(source_path) or source['selection']['task_id'] != task_id:
        raise ValueError('Source identity mismatch: '+task_id)
    controls = {kind: read(bind_file(bindings, Path(entry[kind])/'verification.json'))
                for kind in ('gold', 'empty')}
    if not controls_ready(runtime, controls['gold'], controls['empty']):
        raise ValueError('Shared gold/empty controls failed: '+task_id)
    if not required_cases_passed(task_id, source['source'], controls['gold']['test_outcomes']):
        raise ValueError('Official required cases not passed: '+task_id)
    for kind, result in controls.items():
        directory = Path(entry[kind])
        for name in ('junit.xml', 'test.log', 'executed-command.json',
                     'hidden-test-hashes.json', 'injected-hidden.json'):
            bind_file(bindings, directory/name)
        command = read(directory/'executed-command.json')
        if command['command'] != runtime['test_command'] or command['repo_cwd'] != runtime['repo_cwd']:
            raise ValueError('Test command drift: '+task_id)
        if command['trusted_control'] is not (kind == 'gold'):
            raise ValueError('Invalid trusted-control provenance: '+task_id)
        injection = read(directory/'injected-hidden.json')
        if not injection or any(Path(item['payload_relative']).name in
                {'gold.patch', 'old-files.json', 'expected-output.json'} for item in injection):
            raise ValueError('Reference artifact injected into scorer: '+task_id)
        for item in injection:
            actual = bind_file(bindings, Path(runtime['payload_dir'])/item['payload_relative'])
            if item['sha256'] != sha(actual):
                raise ValueError('Injected hidden payload drift: '+task_id)
        if not result.get('manager_proofs'):
            raise ValueError('Missing actual sanitized-sandbox proof: '+task_id)
        for proof in result['manager_proofs']:
            if not sanitized_proof_ready(read(bind_file(bindings, proof))):
                raise ValueError('Sanitization proof failed: '+task_id)
    payload = Path(runtime['payload_dir'])
    pinned = runtime.get('payload_sha256', {})
    if not pinned:
        # The historical SWE calibrator did not write payload_sha256. Bind its
        # actual files now and match hashes against the recorded injection and
        # gold application, rather than inventing an earlier pin.
        pinned = {str(path.relative_to(payload)): sha(path)
                  for path in payload.rglob('*') if path.is_file()}
    for filename, digest in pinned.items():
        path = bind_file(bindings, payload/filename)
        if sha(path) != digest:
            raise ValueError('Pinned payload drift: '+str(path))
    if controls['gold'].get('patch_sha256') != sha(payload/'gold.patch'):
        raise ValueError('Executed gold patch hash mismatch: '+task_id)
    return {'eligible': True, 'gold_passed': True, 'empty_passed': False,
            'runtime_path': str(runtime_path), 'controls': entry,
            'official_required_cases_passed': True, 'sanitized_scoring_controls': True}


def build_gate(root, controls_index, fake_report):
    root = Path(root).resolve()
    selection, dataset = root/'data/selection.json', root/'data/tasks.jsonl'
    tasks = read(selection)['tasks']
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if len(tasks) != 20 or len(set(tasks)) != 20 or {row['instance_id'] for row in rows} != set(tasks) or len(rows) != 20:
        raise ValueError('Expected exact unique held-out20 selection')
    if any(row.get('patch') or row.get('test_patch') for row in rows):
        raise ValueError('Public task data contains a reference/test patch')
    index, smoke = read(controls_index), read(fake_report)
    if smoke.get('passed') is not True or smoke.get('fake_backend') is not True or smoke.get('counts_as_model_evaluation') is not False:
        raise ValueError('CPU fake OpenAI contract smoke is missing')
    required_smoke_flags = ('official_runner_keyword_abi', 'real_podman_sandbox',
                            'fresh_hidden_tests', 'empty_patch_unresolved',
                            'graph_embedding_unavailable_fallback_exercised')
    if smoke.get('requests', 0) < 5 or any(smoke.get(flag) is not True for flag in required_smoke_flags):
        raise ValueError('CPU fake OpenAI smoke did not exercise the full official path')
    bindings = {}
    for path in (selection, dataset, controls_index, fake_report):
        bind_file(bindings, path)
    for path in (root/'src').glob('*.py'):
        bind_file(bindings, path)
    official_manifest = root/'reports/official-runtime-files.json'
    if official_manifest.exists():
        manifest = read(bind_file(bindings, official_manifest))
        for filename, digest in manifest.items():
            if not Path(filename).is_absolute() or sha(bind_file(bindings, filename)) != digest:
                raise ValueError('Official runtime/submission drift: '+filename)
    # Keep the real HTTP/tool trace and sanitized sandbox evidence alongside the
    # explicit fake-backend result; these never count as model performance.
    for path in Path(fake_report).parent.rglob('*'):
        if path.is_file() and not path.is_symlink():
            bind_file(bindings, path)
    evidence = {task_id: task_evidence(root, task_id, index[task_id], bindings) for task_id in tasks}
    return {'passed': True, 'created_unix': time.time(), 'scope': 'CPU environment readiness only',
            'model_evaluation_held_by_user': True, 'no_gpu_started': True,
            'selection_sha256': sha(selection), 'dataset_sha256': sha(dataset),
            'tasks': evidence, 'artifact_sha256': bindings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--controls-index', type=Path, required=True)
    parser.add_argument('--fake-report', type=Path, required=True)
    args = parser.parse_args()
    if not (args.root/'reports/official-runtime-files.json').is_file():
        raise ValueError('Missing official runtime/submission manifest')
    gate = build_gate(args.root, args.controls_index, args.fake_report)
    target = args.root/'reports/cpu-gate.json'
    if target.exists():
        raise FileExistsError('Refusing to overwrite a frozen gate')
    save(target, gate)
    print(json.dumps({'passed': True, 'gate': str(target), 'tasks': len(gate['tasks'])}))


if __name__ == '__main__':
    main()
