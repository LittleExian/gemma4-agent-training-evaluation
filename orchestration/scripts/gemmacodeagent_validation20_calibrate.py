"""Calibrate immutable official SWE images using fresh gold/empty controls."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import uuid
import xml.etree.ElementTree as ET


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def image_reference(source):
    image = source.get('image_name')
    if not image:
        # Published SWE-Gym Docker Hub repositories use '_s_', confirmed using
        # Docker Hub's repository/tag API (the older generator retained '__').
        instance = source['instance_id'].replace('__', '_s_').lower()
        if not re.fullmatch(r'[a-z0-9_.-]+', instance):
            raise ValueError('Invalid instance ID')
        image = 'docker.io/xingyaoww/sweb.eval.x86_64.' + instance + ':latest'
    if not image.startswith('docker.io/'):
        image = 'docker.io/' + image
    return image


def test_files(source):
    cases = source.get('FAIL_TO_PASS', []) + source.get('PASS_TO_PASS', [])
    paths = sorted({case.split('::')[0] for case in cases})
    if not paths or any(not re.fullmatch(r'[A-Za-z0-9_./-]+\.py', p)
                        or p.startswith('/') or '..' in p.split('/') for p in paths):
        raise ValueError('Missing or unsafe official test file paths')
    return paths


def select_command(source, python):
    if not python.startswith('/') or not re.fullmatch(r'[A-Za-z0-9_./-]+', python):
        raise ValueError('Python must be an absolute executable path')
    original = source.get('install_config', {}).get('test_cmd', '')
    if any(token in original for token in (';', '|', '&', '`', '$', '\n', '>', '<')):
        raise ValueError('Unsupported shell operators in official test command')
    args = shlex.split(original) if original else ['pytest', '-ra', '--tb=short']
    if not args or args[0] != 'pytest':
        raise ValueError('Only official pytest commands are supported')
    files = test_files(source)
    if not original:
        args += files
    args = [python, '-m', *args, '-o', 'junit_family=legacy',
            '--junitxml=/tmp/validation-result.xml']
    return shlex.join(args)


def parse_junit(text, files):
    root = ET.fromstring(text)
    outcomes = {}
    priorities = {'passed': 0, 'skipped': 1, 'failed': 2, 'error': 3}
    for case in root.iter('testcase'):
        cls = case.get('classname', '')
        filename = case.get('file')
        if filename:
            prefix = filename.removesuffix('.py').replace('/', '.')
        else:
            matches = [p for p in files if cls == p[:-3].replace('/', '.')
                       or cls.startswith(p[:-3].replace('/', '.') + '.')]
            if len(matches) != 1:
                raise ValueError('Cannot uniquely map JUnit classname: ' + cls)
            filename = matches[0]
            prefix = filename[:-3].replace('/', '.')
        suffix = cls[len(prefix):].lstrip('.') if cls.startswith(prefix) else ''
        identifier = filename + ('::' + suffix.replace('.', '::') if suffix else '')
        identifier += '::' + case.get('name', '')
        status = next((label for tag, label in (('error', 'error'), ('failure', 'failed'),
                      ('skipped', 'skipped')) if case.find(tag) is not None), 'passed')
        if priorities[status] >= priorities.get(outcomes.get(identifier), -1):
            outcomes = {**outcomes, identifier: status}
    return outcomes


def assess_controls(gold, empty, expected_failures, expected_passes=()):
    good, bad = gold['test_outcomes'], empty['test_outcomes']
    matched = {target: [case for case in good if case == target or
               ('[' in target and not target.endswith(']') and case.startswith(target))]
               for target in [*expected_failures, *expected_passes]}
    # Broken metadata sometimes truncates several parameterized cases to one
    # prefix. Require *every* full case under that prefix, never select one.
    target_ok = all(cases and all(good[case] == 'passed' for case in cases)
                    for cases in matched.values())
    changed = [case for case in good if good[case] == 'passed'
               and bad.get(case) == 'failed']
    passed = (gold['exit_code'] == 0 and empty['exit_code'] == 1 and bool(good)
              and set(good) == set(bad) and bool(changed) and target_ok
              and not any(status in ('failed', 'error') for status in good.values())
              and not any(status == 'error' for status in bad.values()))
    return {'passed': passed, 'matched_official_failures': matched,
            'discriminating_cases': changed, 'same_nonzero_collection': bool(good) and set(good) == set(bad),
            'metadata_prefix_warnings': [target for target, cases in matched.items() if cases and target != cases[0]],
            'test_cases': sorted(good),
            'passed_cases': sorted(case for case in good if good[case] == 'passed')}


def assess_patch(runtime, result):
    expected = runtime['expected_output']
    cases = result['test_outcomes']
    same = set(cases) == set(expected['test_cases']) and bool(cases)
    resolved = result['exit_code'] == 0 and same and all(
        cases.get(case) == 'passed' for case in expected['passed_cases'])
    counts = {label: sum(value == label for value in cases.values()) for label in ('passed', 'failed', 'error')}
    return {**result, 'resolved': resolved, 'same_nonzero_collection': same,
            'infra_failed': False, 'failure_kind': None if resolved else result.get('failure_kind', 'candidate_test_failure'),
            'test_count': len(cases), 'test_exit_code': result['exit_code'],
            'expected_tests_covered': same, 'passed': counts['passed'],
            'failed': counts['failed'], 'errors': counts['error']}


def patch_paths(text):
    paths = []
    for line in text.splitlines():
        if line.startswith(('+++ ', '--- ')):
            value = line[4:].split('\t')[0]
            if value == '/dev/null':
                continue
            if not value.startswith(('a/', 'b/')):
                raise ValueError('Unsupported patch path')
            value = value[2:]
            if value.startswith('/') or any(part in ('.', '..', '.git') for part in value.split('/')):
                raise ValueError('Unsafe patch path')
            paths.append(value)
    if not paths and text.strip():
        raise ValueError('Patch contains no unified diff paths')
    return sorted(set(paths))


def validate_candidate(text, hidden_paths):
    paths = patch_paths(text)
    for path in paths:
        parts = path.lower().split('/')
        if path in hidden_paths or any(p in ('test', 'tests', 'testing', '.git') for p in parts) or parts[-1].startswith('test_') or parts[-1] == 'conftest.py':
            raise ValueError('Candidate changes protected tests: ' + path)
    return paths


class Container:
    def __init__(self, image, output):
        if not re.fullmatch(r'docker\.io/[A-Za-z0-9_./:-]+@sha256:[a-f0-9]{64}', image):
            raise ValueError('Image must have immutable fully qualified digest')
        self.image, self.output = image, Path(output)
        self.owner, self.cid = uuid.uuid4().hex, None
        self.output.mkdir(parents=True, exist_ok=False)

    def cli(self, args, timeout=300, input_data=None, check=True):
        return subprocess.run(['podman', *args], input=input_data, capture_output=True,
                              timeout=timeout, check=check)

    def __enter__(self):
        args = ['run', '-d', '--pull=never', '--network=none', '--cpus=2', '--memory=4g',
                '--pids-limit=256', '--cap-drop=all', '--security-opt=no-new-privileges',
                '--label', 'gemma.validation.swe.owner=' + self.owner]
        for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS'):
            args += ['--env', variable + '=1']
        args += ['--entrypoint', '/bin/bash', self.image, '-c', 'while true; do sleep 3600; done']
        self.cid = self.cli(args).stdout.decode().strip()
        write_json(self.output/'owned-container.json', {'id': self.cid, 'owner': self.owner,
                   'image': self.image, 'command': ['podman', *args]})
        return self

    def verify_owned(self):
        data = json.loads(self.cli(['inspect', self.cid]).stdout)[0]
        if data['Id'] != self.cid or data['Config']['Labels'].get('gemma.validation.swe.owner') != self.owner:
            raise PermissionError('Container ownership mismatch')

    def execute(self, command, timeout=300, input_data=None, check=True):
        self.verify_owned()
        return self.cli(['exec', '-i', self.cid, '/bin/bash', '-c', command],
                        timeout=timeout, input_data=input_data, check=check)

    def send(self, path, data):
        command = 'base64 -d > ' + shlex.quote(path)
        self.execute(command, input_data=base64.b64encode(data))

    def __exit__(self, kind, value, traceback):
        self.verify_owned()
        self.cli(['rm', '-f', self.cid])


def execute_test(runtime, patch, output):
    """A new immutable-image container for each verification; no host mounts."""
    with Container(runtime['image'], output) as container:
        cwd = runtime['repo_cwd']
        commit = runtime['resolved_base_commit']
        if not re.fullmatch(r'[a-f0-9]{40}', commit) or not re.fullmatch(r'/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', cwd):
            raise ValueError('Unsupported repository base')
        prefix = 'cd ' + shlex.quote(cwd) + ' && '
        setup = prefix + 'git rev-parse HEAD && git status --porcelain'
        state = container.execute(setup).stdout.decode()
        if state.splitlines()[0] != commit:
            raise ValueError('Image HEAD differs from official base commit: ' + state)
        container.execute(prefix + 'git reset --hard ' + commit)
        hidden = Path(runtime['payload_dir'])/'test.patch'
        container.send('/tmp/hidden-tests.patch', hidden.read_bytes())
        container.execute(prefix + 'git apply --whitespace=nowarn /tmp/hidden-tests.patch')
        hidden_paths = patch_paths(hidden.read_text())
        hash_command = prefix + 'sha256sum -- ' + ' '.join(shlex.quote(p) for p in hidden_paths)
        before = container.execute(hash_command).stdout.decode()
        if patch and Path(patch).read_text().strip():
            if Path(patch).name != 'gold.patch':
                validate_candidate(Path(patch).read_text(), hidden_paths)
            container.send('/tmp/candidate.patch', Path(patch).read_bytes())
            applied = container.execute(prefix + 'git apply --whitespace=nowarn /tmp/candidate.patch', check=False)
            if applied.returncode:
                (Path(output)/'apply.log').write_bytes(applied.stdout + applied.stderr)
                return {'exit_code': applied.returncode, 'test_count': 0, 'test_outcomes': {},
                        'log': str(Path(output)/'apply.log'), 'junit': None,
                        'patch_apply_failed': True, 'failure_kind': 'candidate_invalid_patch'}
        after_apply = container.execute(hash_command).stdout.decode()
        if before != after_apply:
            raise ValueError('Candidate changed hidden test files')
        try:
            result = container.execute(prefix + runtime['test_command'], timeout=900, check=False)
        except subprocess.TimeoutExpired:
            return {'exit_code': 124, 'test_count': 0, 'test_outcomes': {},
                    'log': None, 'junit': None, 'failure_kind': 'candidate_test_timeout'}
        log = Path(output)/'test.log'
        log.write_bytes(result.stdout + result.stderr)
        collected = container.execute('cat /tmp/validation-result.xml', check=False)
        junit = Path(output)/'junit.xml'
        junit.write_bytes(collected.stdout)
        try:
            outcomes = parse_junit(collected.stdout.decode(), runtime['test_files']) if collected.returncode == 0 else {}
        except (ValueError, ET.ParseError):
            outcomes = {}
        after_tests = container.execute(hash_command, check=False).stdout.decode()
        write_json(Path(output)/'hidden-test-hashes.json', {'before': before, 'after_apply': after_apply, 'after_tests': after_tests})
        if before != after_tests:
            return {'exit_code': 125, 'test_count': 0, 'test_outcomes': {}, 'log': str(log),
                    'junit': str(junit), 'failure_kind': 'candidate_changed_hidden_tests'}
        return {'exit_code': result.returncode, 'test_count': len(outcomes),
                'test_outcomes': outcomes, 'log': str(log), 'junit': str(junit),
                'patch_apply_failed': False}


def verify_patch(runtime, patch_file, output):
    if not runtime.get('passed'):
        raise ValueError('Task has no passing gold/empty calibration')
    # Copy into a name that cannot use the trusted gold-control bypass.
    candidate = Path(output).parent / ('.candidate-' + uuid.uuid4().hex + '.patch')
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(Path(patch_file).read_bytes())
    try:
        try:
            validate_candidate(candidate.read_text(), patch_paths((Path(runtime['payload_dir'])/'test.patch').read_text()))
        except ValueError as error:
            result = {'exit_code': 126, 'test_count': 0, 'test_outcomes': {}, 'log': None,
                      'junit': None, 'failure_kind': 'candidate_invalid_patch', 'error': str(error)}
        else:
            result = execute_test(runtime, candidate, Path(output))
    finally:
        candidate.unlink()
    return assess_patch(runtime, result)


def repository_cwd(configured, base_commit, probe):
    candidates = [configured] if configured and configured != '/' else []
    if '/testbed' not in candidates:
        candidates.append('/testbed')
    for path in candidates:
        if not re.fullmatch(r'/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', path):
            continue
        if probe(path) == base_commit:
            return path
    raise ValueError('No baked repository matches the exact official base commit')


def pull_image(source, output):
    image = image_reference(source)
    path = Path(output)/'pull.log'
    with path.open('wb') as log:
        subprocess.run(['podman', 'pull', image], stdout=log, stderr=subprocess.STDOUT,
                       timeout=1800, check=True)
    details = json.loads(subprocess.check_output(['podman', 'image', 'inspect', image]))[0]
    write_json(Path(output)/'image-inspect.json', details)
    digest = next((item for item in details['RepoDigests'] if item.startswith('docker.io/')), None)
    if not digest:
        raise ValueError('No fully qualified image digest found')
    with Container(digest, Path(output)/('repository-probe-'+uuid.uuid4().hex[:8])) as container:
        def probe(path):
            result = container.execute('git -C '+shlex.quote(path)+' rev-parse HEAD', check=False)
            return result.stdout.decode().strip() if result.returncode == 0 else None
        cwd = repository_cwd(details['Config'].get('WorkingDir'), source['base_commit'], probe)
    return digest, cwd


def probe_python(image, output):
    with Container(image, output) as container:
        script = "for p in /opt/miniconda3/envs/testbed/bin/python /opt/conda/envs/testbed/bin/python /opt/venv/bin/python /usr/local/bin/python /usr/bin/python3; do if test -x $p && $p -c 'import pytest' >/dev/null 2>&1; then echo $p; exit 0; fi; done; exit 1"
        return container.execute(script).stdout.decode().strip()


def calibrate(source_file, output):
    document = json.loads(Path(source_file).read_text())
    source = document['source']
    output = Path(output)/source['instance_id']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'task-runtime.json').exists():
        saved = json.loads((output/'task-runtime.json').read_text())
        if saved.get('passed'):
            return saved
    payload = output/'payload'
    payload.mkdir(exist_ok=True)
    (payload/'test.patch').write_text(source['test_patch'])
    (payload/'gold.patch').write_text(source['patch'])
    (output/'source.json').write_text(Path(source_file).read_text())
    image, cwd = pull_image(source, output)
    python = probe_python(image, output/('probe-' + uuid.uuid4().hex[:8]))
    runtime = {'task_id': source['instance_id'], 'passed': False, 'image': image,
               'repo_cwd': cwd, 'resolved_base_commit': source['base_commit'],
               'python_executable': python, 'test_command': select_command(source, python),
               'test_files': test_files(source), 'payload_dir': str(payload),
               'source_sha256': hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
               'provenance': document['provenance']}
    stamp = uuid.uuid4().hex[:8]
    results = {'gold': execute_test(runtime, payload/'gold.patch', output/('gold-' + stamp)),
               'empty': execute_test(runtime, None, output/('empty-' + stamp))}
    expected = assess_controls(results['gold'], results['empty'], source['FAIL_TO_PASS'], source['PASS_TO_PASS'])
    runtime = {**runtime, 'results': results, 'expected_output': expected,
               'passed': expected['passed']}
    write_json(output/'task-runtime.json', runtime)
    print(json.dumps({'task': runtime['task_id'], 'passed': runtime['passed'],
                      'gold': results['gold']['exit_code'], 'empty': results['empty']['exit_code'],
                      'tests': results['gold']['test_count']}), flush=True)
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [p for p in sorted(args.sources.glob('*.source.json'))
               if not p.name.startswith('r2e_') and (not args.ids or p.name[:-12] in args.ids)]
    results = []
    for source in sources:
        try:
            results.append(calibrate(source, args.output))
        except Exception as error:
            task = source.name[:-12]
            failure = {'task_id': task, 'passed': False, 'error': repr(error)}
            write_json(args.output/task/'CALIBRATION_ERROR.json', failure)
            print(json.dumps(failure), flush=True)
            results.append(failure)
        write_json(args.output/'summary.json', {'tasks': results, 'passed': sum(r['passed'] for r in results)})


if __name__ == '__main__':
    main()
