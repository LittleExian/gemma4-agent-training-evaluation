"""Install isolated, snapshot-derived test profiles without changing model runtime."""
import concurrent.futures
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/workspace/gemma4-notebook-run-20261008')


PROBE = '''
import importlib.metadata, json, platform, sys
from pip._vendor.packaging.requirements import Requirement
errors = []
for declaration in json.loads(sys.argv[1]):
    requirement = Requirement(declaration)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    try:
        version = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        errors.append(declaration + ': missing distribution')
        continue
    if requirement.url or not requirement.specifier.contains(version, prereleases=True):
        errors.append(declaration + ': installed ' + version)
print(json.dumps({'python_version': platform.python_version(), 'requirement_errors': errors}))
'''


def freeze_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def inspect_environment(env_dir, profile, environment):
    python = str(env_dir / 'bin/python')
    value = json.loads(subprocess.check_output(
        [python, '-I', '-B', '-c', PROBE, json.dumps(profile['requirements'])],
        env=environment, text=True, timeout=60))
    subprocess.run([python, '-I', '-B', '-m', 'pip', 'check'], env=environment,
                   capture_output=True, text=True, check=True, timeout=60)
    frozen = subprocess.check_output([python, '-I', '-B', '-m', 'pip', 'freeze', '--all'],
                                     env=environment, text=True, timeout=60)
    return {**value, 'actual_freeze': frozen}


def validate_snapshot(snapshot, profile):
    version = '.'.join(snapshot['python_version'].split('.')[:2])
    if version != profile['python']:
        raise ValueError('Profile Python does not match actual interpreter')
    if snapshot['requirement_errors']:
        raise ValueError('Unsatisfied profile requirements: ' + '; '.join(snapshot['requirement_errors']))


def reuse_cached(env_dir, profile, marker, requirement_sha, environment):
    try:
        previous = json.loads(marker.read_text())
        if (not previous.get('success') or previous.get('fingerprint') != profile['fingerprint']
                or previous.get('requirements_sha256') != requirement_sha
                or previous.get('environment') != str(env_dir)
                or not (env_dir / 'bin/python').exists()):
            raise ValueError('Cached installation metadata is stale or incomplete')
        snapshot = inspect_environment(env_dir, profile, environment)
        validate_snapshot(snapshot, profile)
        frozen = snapshot['actual_freeze']
        if frozen != (env_dir / 'requirements.freeze.txt').read_text():
            raise ValueError('Actual dependency freeze differs from saved freeze')
        digest = freeze_sha(frozen)
        if previous.get('actual_freeze_sha256', digest) != digest:
            raise ValueError('Recorded dependency freeze hash differs')
        result = {**previous, 'actual_freeze_sha256': digest,
                  'python_version': snapshot['python_version']}
        marker.write_text(json.dumps(result, indent=2))
        return result
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise RuntimeError('Existing profile environment cannot be reused; select a new environment path. '
                           'The existing environment was not repaired. Reason: ' + str(exc)) from exc


def install(item):
    key, profile = item
    env_dir = Path(profile['environment'])
    if not env_dir.resolve().is_relative_to((ROOT / 'test-envs').resolve()):
        raise ValueError('Environment must be inside this preflight run')
    requirement = ROOT / 'requirements' / (key + '.txt')
    requirement_text = '\n'.join(profile['requirements']) + '\n'
    requirement_sha = freeze_sha(requirement_text)
    marker = ROOT / 'installation' / (key + '.json')
    executable = '/usr/bin/python3.10' if profile['python'] == '3.10' else '/opt/anaconda3/bin/python'
    environment = {**os.environ, 'HOME': str(ROOT / 'setup-home'),
                   'PIP_CONFIG_FILE': '/dev/null', 'PIP_INDEX_URL': 'https://pypi.org/simple',
                   'PIP_CACHE_DIR': str(ROOT / 'pip-cache'), 'PIP_NO_INPUT': '1',
                   'PYTHONNOUSERSITE': '1'}
    if marker.exists() or env_dir.exists():
        return reuse_cached(env_dir, profile, marker, requirement_sha, environment)
    requirement.parent.mkdir(exist_ok=True)
    requirement.write_text(requirement_text)
    log_path = ROOT / 'installation' / (key + '.log')
    success = False
    error = None
    snapshot = None
    with log_path.open('w') as log:
        try:
            if not (env_dir / 'bin/python').exists():
                subprocess.run([executable, '-m', 'venv', str(env_dir)], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            python = str(env_dir / 'bin/python')
            subprocess.run([python, '-m', 'pip', 'install', '--disable-pip-version-check',
                            '--progress-bar', 'off', '--find-links', str(BASE / 'data/wheels'),
                            '--index-url', 'https://pypi.org/simple', '-r', str(requirement)],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            snapshot = inspect_environment(env_dir, profile, environment)
            validate_snapshot(snapshot, profile)
            frozen = snapshot['actual_freeze']
            (env_dir / 'requirements.freeze.txt').write_text(frozen)
            success = True
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    result = {'profile': key, 'success': success, 'error': error,
              'fingerprint': profile['fingerprint'], 'environment': str(env_dir),
              'requirements_sha256': requirement_sha,
              'actual_freeze_sha256': freeze_sha(snapshot['actual_freeze']) if success else None,
              'python_version': snapshot['python_version'] if success else None,
              'log': str(log_path)}
    marker.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-map', default=str(ROOT / 'task-profiles.json'))
    args = parser.parse_args()
    for directory in ['installation', 'requirements', 'setup-home', 'pip-cache', 'test-envs']:
        (ROOT / directory).mkdir(exist_ok=True)
    mapping = json.loads(Path(args.profile_map).read_text())
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(install, item) for item in mapping['profiles'].items()]
        for job in concurrent.futures.as_completed(jobs):
            results.append(job.result())
            (ROOT / 'install-status.json').write_text(json.dumps(results, indent=2))
    failures = [r['profile'] for r in results if not r['success']]
    print(json.dumps({'profiles': len(results), 'failed': failures}), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
