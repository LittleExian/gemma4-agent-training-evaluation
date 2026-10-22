"""Finalize reference-control calibration without altering raw generation results.

Only task metadata, gold/empty controls and recorded environments inform this gate.
Model outcomes and model success rates are never read.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import tempfile
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

RULE_VERSION = 'reference-controls-v4-declared-requirements'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(record):
    summaries = [re.findall(r'\b(\d+) (passed|failed|skipped|errors?)\b', line)
                 for line in str(record.get('test_output') or '').splitlines()]
    summaries = [line for line in summaries if line]
    result = {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}
    if summaries:
        for number, label in summaries[-1]:
            result['errors' if label.startswith('error') else label] = int(number)
    return result


def all_text(record):
    return '\n'.join(str(record.get(key) or '') for key in ('test_output', 'error', 'error_message', 'exception'))


def canonical_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def frozen_versions(freeze):
    return {canonical_name(name): version for name, version in
            re.findall(r'^([A-Za-z0-9_.-]+)==([^\s]+)$', freeze, re.MULTILINE)}


def under(path, root):
    return PurePosixPath(posixpath.normpath(path)).is_relative_to(PurePosixPath(posixpath.normpath(root)))


def declared_profile_issues(profile, probe, versions):
    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import InvalidVersion
    issues = []
    actual_version = re.match(r'^(\d+\.\d+)(?:\.\d+)?', str(probe.get('python_version') or ''))
    if not actual_version:
        return ['missing_actual_python_version']
    full_version = actual_version.group(0)
    for field in ('python', 'python_version'):
        declared = profile.get(field)
        if declared:
            declared_version = re.match(r'^\d+\.\d+', str(declared))
            if not declared_version or declared_version.group(0) != actual_version.group(1):
                issues.append('declared_python_version_mismatch')
    environment = {**default_environment(), 'python_version': actual_version.group(1),
                   'python_full_version': full_version, 'implementation_name': 'cpython',
                   'platform_python_implementation': 'CPython',
                   'sys_platform': 'linux', 'os_name': 'posix', 'platform_system': 'Linux', 'extra': ''}
    requirements = profile.get('requirements')
    if not isinstance(requirements, list):
        return issues + ['missing_declared_requirements']
    for declaration in requirements:
        try:
            requirement = Requirement(declaration)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            name = canonical_name(requirement.name)
            actual = versions.get(name)
            if actual is None:
                issues.append('declared_requirement_missing:' + name)
            elif requirement.url:
                issues.append('declared_requirement_url_not_verified:' + name)
            elif not requirement.specifier.contains(actual, prereleases=True):
                issues.append('declared_requirement_version_mismatch:' + name + ':' + actual + ':' + str(requirement.specifier))
        except (InvalidRequirement, InvalidVersion, TypeError, ValueError) as error:
            issues.append('invalid_declared_requirement:' + str(declaration) + ':' + str(error))
    return sorted(set(issues))


def validate_probe(record, package):
    profile = record.get('dependency_profile') or {}
    if not profile.get('actual_freeze') or not profile.get('environment'):
        return ['missing_profile_freeze_or_environment']
    probes = record.get('environment_probes') or []
    if not probes:
        return ['missing_actual_environment_probe']
    issues = []
    versions = frozen_versions(profile['actual_freeze'])
    for probe in probes:
        issues.extend(declared_profile_issues(profile, probe, versions))
        if not probe.get('site_packages') or not probe.get('sys_path'):
            issues.append('missing_actual_import_paths')
        origin = str(probe.get('origin') or '')
        workspace = origin.split('/workspace/', 1)[0] + '/workspace'
        sandbox = workspace.rsplit('/', 1)[0]
        if ('/swegemma_sandbox_' not in workspace or '/workspace/' not in origin
                or not under(origin, workspace) or probe.get('package') != package):
            issues.append('target_package_origin_not_in_sandbox_workspace')
        if not under(str(probe.get('executable') or ''), sandbox + '/venv/bin'):
            issues.append('interpreter_not_in_sandbox_venv')
        recorded_version = str(profile.get('interpreter_version') or '').removeprefix('Python ')
        if not recorded_version or not str(probe.get('python_version') or '').startswith(recorded_version):
            issues.append('interpreter_version_mismatch')
        for path in probe.get('site_packages') or []:
            if not under(path, sandbox + '/venv'):
                issues.append('site_packages_outside_sandbox_venv')
        for path in probe.get('sys_path') or []:
            if 'site-packages' in path or 'dist-packages' in path:
                if not any(under(path, root) for root in (sandbox + '/venv', profile['environment'])):
                    issues.append('inherited_external_site_packages')
        dependencies = probe.get('dependencies')
        if not isinstance(dependencies, dict) or not dependencies:
            issues.append('missing_actual_dependency_versions')
        else:
            for name, version in dependencies.items():
                if versions.get(canonical_name(name)) != version:
                    issues.append('dependency_version_mismatch:' + name)
    return sorted(set(issues))


def environment_issues(task, gold, empty):
    issues = []
    profile = gold.get('dependency_profile') or {}
    if profile != (empty.get('dependency_profile') or {}):
        issues.append('gold_empty_dependency_profile_mismatch')
    if not gold.get('fingerprint') or gold.get('fingerprint') != empty.get('fingerprint'):
        issues.append('gold_empty_fingerprint_mismatch')
    package = task['repo'].rsplit('/', 1)[-1]
    issues.extend('gold:' + issue for issue in validate_probe(gold, package))
    issues.extend('empty:' + issue for issue in validate_probe(empty, package))
    gold_probes = gold.get('environment_probes') or []
    empty_probes = empty.get('environment_probes') or []
    if gold_probes and empty_probes:
        for key in ('python_version', 'dependencies'):
            if gold_probes[-1].get(key) != empty_probes[-1].get(key):
                issues.append('gold_empty_actual_' + key + '_mismatch')
    return sorted(set(issues))


def patch_paths(task):
    return sorted(set(re.findall(r'^\+\+\+ b/(\S+)', task.get('patch') or '', re.MULTILINE)))


def implementation_evidence(module, task, symbol=None):
    package = task['repo'].rsplit('/', 1)[-1]
    paths = patch_paths(task)
    module_path = module.replace('.', '/')
    corresponding = [path for path in paths if path in (module_path + '.py', module_path + '/__init__.py')
                     or path.startswith(module_path + '/') or path.startswith('src/' + module_path + '/')
                     or path in ('src/' + module_path + '.py', 'src/' + module_path + '/__init__.py')]
    is_repo = module == package or module.startswith(package + '.')
    is_script = module.startswith(('scripts.', 'docs_src.')) and bool(corresponding)
    if not is_repo and not is_script:
        return None
    if symbol and not (corresponding or re.search(r'^\+.*\b' + re.escape(symbol) + r'\b', task.get('patch') or '', re.MULTILINE)):
        return None
    return {'kind': 'expected_missing_implementation', 'module': module, 'symbol': symbol,
            'reference_patch_paths': corresponding or [path for path in paths if path.startswith(package + '/')],
            'basis': 'Target-repository module/API or reference-patch implementation path; accepted only when the gold control passes in the same recorded environment.'}


def import_diagnostics(task, record):
    text = all_text(record)
    evidence, external = [], []
    for module in sorted(set(re.findall(r"No module named [\"']?([\w.]+)", text))):
        entry = implementation_evidence(module, task)
        if entry:
            evidence.append(entry)
        else:
            external.append(module)
    for symbol, module in sorted(set(re.findall(r"cannot import name [\"']([^\"']+)[\"'] from [\"']([^\"']+)[\"']", text))):
        entry = implementation_evidence(module, task, symbol)
        if entry:
            evidence.append(entry)
        else:
            external.append(module + ':' + symbol)
    import_mentioned = re.search(r'\b(?:ModuleNotFoundError|ImportError):', text)
    if import_mentioned and not evidence and not external:
        external.append('unattributed_import_failure')
    return evidence, external


def hard_failures(record):
    lower = all_text(record).lower()
    patterns = {
        'patch_application_failure': ('failed to apply', 'patch does not apply', 'patch file missing'),
        'verification_timeout': ('command timed out', 'timeout expired', 'timeoutexpired', 'command timeout'),
        'no_tests': ('no tests ran', 'no tests collected', 'collected 0 items', 'missing test specification', 'missing or empty junit'),
        'setup_failure': ('containersetuperror', 'failed to set up', 'snapshot file not found', 'no space left on device',
                          'pytest: command not found', 'target package import origin outside workspace', 'landlock guard failed', 'unexpected subprocess sandbox'),
    }
    failures = [name for name, tokens in patterns.items() if any(token in lower for token in tokens)]
    if record.get('test_exit_code') not in (0, 1, 2):
        failures.append('invalid_or_setup_exit_code')
    return failures


def reviewed_permission_evidence(task, record):
    text = all_text(record)
    patch = task.get('patch') or ''
    if (task['instance_id'] == 'rich_4070' and 'test_input_password' in text and '/dev/tty' in text
            and 'getpass.py' in text and '-from getpass import getpass' in patch
            and re.search(r'^\+\s+(?:from getpass import getpass|import getpass(?: as \w+)?)$', patch, re.MULTILINE)
            and not any(path != '/dev/tty' for path in re.findall(r"Permission denied: ['\"]([^'\"]+)", text))):
        return {'kind': 'reviewed_reference_failure', 'task': 'rich_4070',
                'basis': 'Reference patch moves the getpass import into Console.input so the test monkeypatch takes effect; unpatched code calls the captured getpass alias and reaches /dev/tty under pytest capture.',
                'reference_patch_path': 'rich/console.py', 'test': 'test_input_password',
                'permissions_relaxed': False}
    return None


def reviewed_nontermination_evidence(task, empty):
    if empty.get('test_exit_code') != 124:
        return None
    patch, tests = task.get('patch') or '', task.get('test_patch') or ''
    instance_id = task['instance_id']
    proof = None
    if (instance_id == 'rich_4006' and 'rich/cells.py' in patch
            and 'while index < codepoint_count:' in patch and '-        elif spans:' in patch
            and '+                spans.append((index, index := index + 1, 0))' in patch
            and 'test_chop_cells_zero_width' in tests
            and all(code in tests for code in (r'\x1b', r'\ufe0f', r'\u200d'))):
        proof = {'source_path': 'rich/cells.py', 'test_path': 'tests/test_cells.py',
                 'basis': 'Old split_graphemes loop does not increment index for a leading zero-width code point with empty spans; reference patch appends a zero-width span and increments index. Added controls exercise ESC, VS16 and ZWJ.'}
    elif (instance_id == 'rich_3772' and 'rich/traceback.py' in patch
          and 'cause is not None and cause is not exc_value' in patch
          and '_visited_exceptions' in patch and 'grouped_exceptions.add(exception)' in patch
          and 'test_recursive_exception' in tests and 'raise e from e' in tests):
        proof = {'source_path': 'rich/traceback.py', 'test_path': 'tests/test_traceback.py',
                 'basis': 'Regression constructs an exception whose cause is itself; reference patch rejects cause is exc_value and tracks visited grouped exceptions, preventing unbounded traceback traversal.'}
    elif (instance_id == 'rich_3480' and 'rich/text.py' in patch
          and re.search(r'^-.*for start, end, style in text\._spans$', patch, re.MULTILINE)
          and re.search(r'^\+.*for start, end, style in text\._spans\.copy\(\)', patch, re.MULTILINE)
          and 'test_append_loop_regression' in tests and 'a.append(a)' in tests and 'b.append_text(b)' in tests):
        proof = {'source_path': 'rich/text.py', 'test_path': 'tests/test_text.py',
                 'basis': 'Self-append extends the same span list being lazily iterated; the reference patch iterates a copy, preventing unbounded list growth.'}
    if not proof:
        return None
    commands = [entry for entry in empty.get('commands', [])
                if 'python3 -s -m pytest ' in entry.get('command', '')
                and proof['test_path'] in entry['command'] and entry.get('exit_code') == 124]
    if not commands:
        return None
    return {'kind': 'reviewed_reference_nontermination', 'instance_id': instance_id, **proof,
            'raw_exit_code': 124, 'raw_result_preserved': True,
            'reference_patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
            'reference_test_patch_sha256': hashlib.sha256(tests.encode()).hexdigest(),
            'pytest_command': commands[-1]['command'],
            'acceptance': 'Only this reviewed task and exact source/test evidence, with a normally passing gold control; never a general timeout waiver.'}


def classify_pair(task, gold, empty, *, allow_required_skip=False):
    gold_counts, empty_counts = counts(gold), counts(empty)
    evidence = []
    normal_gold_pass = (gold.get('resolved') is True and gold.get('test_exit_code') == 0
                        and gold_counts['passed'] > 0 and not gold_counts['failed'] and not gold_counts['errors'])
    nontermination = reviewed_nontermination_evidence(task, empty) if normal_gold_pass else None
    reasons = environment_issues(task, gold, empty)
    reasons.extend('gold:' + item for item in hard_failures(gold))
    empty_hard = hard_failures(empty)
    if nontermination:
        empty_hard = [item for item in empty_hard if item not in ('invalid_or_setup_exit_code', 'verification_timeout')]
        evidence.append(nontermination)
    reasons.extend('empty:' + item for item in empty_hard)
    if task['instance_id'] == 'requests_6757' and re.search(r'certificate[_ ]expired', all_text(gold), re.IGNORECASE):
        reasons.append('snapshot_tls_fixture_expired')
        evidence.append({'kind': 'snapshot_tls_fixture_expired', 'not_after': '2026-03-13',
                         'verification_date': '2026-10-16', 'tls_verification_disabled': False,
                         'clock_modified': False,
                         'basis': 'Reviewed bundled mutual-TLS certificate is expired on the evaluation date; gold emits certificate-expired SSL failure. Snapshot fixture needs a separately reviewed repair.'})
    _, gold_external = import_diagnostics(task, gold)
    empty_evidence, empty_external = import_diagnostics(task, empty)
    evidence.extend(empty_evidence)
    if gold_external:
        reasons.append('gold_external_import_failure:' + ','.join(gold_external))
    if empty_external:
        reasons.append('empty_external_import_failure:' + ','.join(empty_external))
    required_skip = allow_required_skip and 'Required test node did not pass:' in all_text(gold)
    if not ((gold.get('resolved') is True or required_skip) and gold.get('test_exit_code') == 0
            and gold_counts['passed'] > 0 and not gold_counts['failed'] and not gold_counts['errors']):
        reasons.append('gold_not_verified_passing_tests')
    if re.search(r'permission denied|permissionerror', all_text(gold), re.IGNORECASE):
        reasons.append('gold_permission_failure')
    if re.search(r'permission denied|permissionerror', all_text(empty), re.IGNORECASE):
        reviewed = reviewed_permission_evidence(task, empty)
        if reviewed:
            evidence.append(reviewed)
        else:
            reasons.append('unreviewed_empty_permission_failure')
    gate = 'quarantined'
    if not reasons:
        if (empty.get('resolved') is True and empty.get('test_exit_code') == 0 and empty_counts['passed'] > 0
                and not empty_counts['failed'] and not empty_counts['errors']):
            gate = 'nondiscriminating'
        elif nontermination:
            gate = 'eligible'
        elif (empty.get('resolved') is False and empty.get('test_exit_code') in (1, 2)
              and empty_counts['failed'] + empty_counts['errors'] > 0):
            gate = 'eligible'
        else:
            reasons.append('empty_has_no_verified_negative_control')
    return {'instance_id': task['instance_id'], 'repo': task['repo'], 'gate': gate,
            'reasons': sorted(set(reasons)), 'evidence': evidence,
            'gold_counts': gold_counts, 'empty_counts': empty_counts, 'empty_raw_exit_code': empty.get('test_exit_code'),
            'dependency_profile': gold.get('dependency_profile'),
            'profile_freeze_sha256': hashlib.sha256(str((gold.get('dependency_profile') or {}).get('actual_freeze') or '').encode()).hexdigest(),
            'record_fingerprint': gold.get('fingerprint'), 'rule_version': RULE_VERSION}


MATRIX_TASKS = {'fastapi_14186', 'rich_3486'}


def required_test_names(task):
    explicit = list(task.get('FAIL_TO_PASS') or []) + list(task.get('PASS_TO_PASS') or [])
    if explicit:
        return sorted(set(explicit)), 'dataset FAIL_TO_PASS/PASS_TO_PASS'
    patch = task.get('test_patch') or ''
    added = re.findall(r'^\+\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(', patch, re.MULTILINE)
    context = re.findall(r'^ \s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(', patch, re.MULTILINE)
    headers = re.findall(r'^@@[^@]+@@\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\b', patch, re.MULTILINE)
    removed = set(re.findall(r'^-\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(', patch, re.MULTILINE))
    names = set(added + context + headers) - (removed - set(added + context))
    return sorted(names), 'reference test_patch added/context/hunk-header test definitions (conservative union)'


def junit_outcomes(record):
    reports = [entry.get('stdout') or '' for entry in record.get('commands', [])
               if 'cat ' in entry.get('command', '') and '_swegemma_junit_' in entry.get('command', '')]
    if not reports:
        return None
    try:
        root = ET.fromstring(reports[-1])
    except ET.ParseError:
        return None
    outcomes = []
    for node in root.findall('.//testcase'):
        status = 'passed'
        for tag in ('failure', 'error', 'skipped'):
            if node.find(tag) is not None:
                status = {'failure': 'failed', 'error': 'errors', 'skipped': 'skipped'}[tag]
                break
        outcomes.append({'name': node.get('name', ''), 'classname': node.get('classname', ''), 'status': status})
    return outcomes or None


def node_matches(node, required):
    name = node['name']
    short = required.rsplit('::', 1)[-1].rsplit('.', 1)[-1]
    base = short.split('[', 1)[0]
    return name in (required, short, base) or name.split('[', 1)[0] == base


def classify_matrix(task, pairs):
    required, method = required_test_names(task)
    reasons, arms, coverage = [], [], {name: [] for name in required}
    if task['instance_id'] not in MATRIX_TASKS:
        reasons.append('matrix_not_reviewed_for_task')
    if not required:
        reasons.append('cannot_extract_required_test_nodes')
    # Keep the latest complete pair for each exact dependency profile.
    latest = {}
    for pair in sorted(pairs, key=lambda item: item['generation']):
        profile_id = pair['gold'].get('dependency_profile', {}).get('profile_id')
        latest[profile_id] = pair
    for profile_id, pair in latest.items():
        gold, empty = pair['gold'], pair['empty']
        verdict = classify_pair(task, gold, empty, allow_required_skip=True)
        outcomes = junit_outcomes(gold)
        arm_reasons = [reason for reason in verdict['reasons'] if reason != 'empty_has_no_verified_negative_control']
        if not outcomes:
            arm_reasons.append('missing_gold_junit_nodes')
        elif not any(node['status'] == 'passed' for node in outcomes) or any(node['status'] in ('failed', 'errors') for node in outcomes):
            arm_reasons.append('gold_junit_not_passing')
        elif not arm_reasons:
            for required_name in required:
                matches = [node for node in outcomes if node_matches(node, required_name)]
                if matches and any(node['status'] == 'passed' for node in matches):
                    coverage[required_name].append(profile_id)
        arm = {'profile_id': profile_id, 'generation': pair['generation'],
               'original_gold_resolved': gold.get('resolved'), 'original_gold_error': gold.get('error_message') or gold.get('error'),
               'gold_counts': counts(gold), 'empty_counts': counts(empty), 'negative_control_gate': verdict['gate'],
               'reasons': arm_reasons, 'evidence': verdict['evidence'], 'gold_junit_nodes': outcomes,
               'dependency_profile': gold.get('dependency_profile'),
               'profile_freeze_sha256': hashlib.sha256(str((gold.get('dependency_profile') or {}).get('actual_freeze') or '').encode()).hexdigest(),
               'record_fingerprint': gold.get('fingerprint'),
               'files': pair.get('files')}
        arms.append(arm)
    missing = [name for name, profiles in coverage.items() if not profiles]
    valid_arms = [arm for arm in arms if not arm['reasons']]
    if missing:
        reasons.append('required_nodes_not_covered:' + ','.join(missing))
    if len(valid_arms) < 2:
        reasons.append('fewer_than_two_verified_matrix_profiles')
    if not any(arm['negative_control_gate'] == 'eligible' for arm in valid_arms):
        reasons.append('no_verified_matrix_negative_control')
    # Only profiles contributing passing nodes or the negative control are required.
    required_profiles = sorted({profile for profiles in coverage.values() for profile in profiles}
                               | {arm['profile_id'] for arm in valid_arms if arm['negative_control_gate'] == 'eligible'})
    return {'instance_id': task['instance_id'], 'repo': task['repo'],
            'gate': 'quarantined' if reasons else 'eligible',
            'calibration_status': 'uncalibrated_matrix' if reasons else 'calibrated_matrix',
            'reasons': reasons, 'evidence': [{'kind': 'complementary_environment_matrix', 'basis': method}],
            'required_profile_ids': required_profiles, 'rule_version': RULE_VERSION,
            'gold_counts': {'passed': sum(arm['gold_counts']['passed'] for arm in valid_arms)},
            'empty_counts': {'failed': sum(arm['empty_counts']['failed'] for arm in valid_arms),
                             'errors': sum(arm['empty_counts']['errors'] for arm in valid_arms)},
            'matrix': {'required_nodes': required, 'coverage': coverage, 'missing_nodes': missing,
                       'arms': arms, 'scoring_requirement': 'Future scoring must run the same required profiles and union their required-test coverage; no raw single-profile resolved value is changed.'}}


def select_pairs(root, generations):
    selected, provenance = {}, {}
    for generation in sorted(set(generations)):
        seen = set()
        for directory in sorted((Path(root) / f'generation-{generation}').glob('shard-*/results/*')):
            if not directory.is_dir():
                continue
            paths = {variant: directory / f'{variant}.json' for variant in ('gold', 'empty')}
            available = {variant: json.loads(path.read_text()) for variant, path in paths.items() if path.exists()}
            if not available:
                continue
            ids = {record.get('instance_id') for record in available.values()}
            if len(ids) != 1 or None in ids:
                raise ValueError(f'Mismatched result identifiers: {directory}')
            instance_id = next(iter(ids))
            if instance_id in seen:
                raise ValueError(f'Duplicate result for generation {generation}: {instance_id}')
            seen.add(instance_id)
            item = {'generation': generation, 'complete': len(available) == 2,
                    'files': {variant: {'path': str(paths[variant]), 'sha256': digest_file(paths[variant])}
                              for variant in available}}
            provenance.setdefault(instance_id, []).append(item)
            if item['complete']:
                selected[instance_id] = {**available, **item}
    return selected, provenance


def finalize(base, root, generations):
    source = Path(base) / 'data/tasks.jsonl'
    tasks = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    identifiers = [task['instance_id'] for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('Duplicate dataset task identifiers')
    selected, provenance = select_pairs(root, generations)
    unknown = set(selected) - set(identifiers)
    if unknown:
        raise ValueError(f'Results outside dataset: {sorted(unknown)}')
    records = []
    for task in tasks:
        pair = selected.get(task['instance_id'])
        if pair:
            verdict = classify_pair(task, pair['gold'], pair['empty'])
            if task['instance_id'] in MATRIX_TASKS:
                history = [{**item, **{variant: json.loads(Path(item['files'][variant]['path']).read_text())
                                       for variant in ('gold', 'empty')}}
                           for item in provenance[task['instance_id']] if item['complete']]
                verdict = classify_matrix(task, history)
            record = {**verdict, 'selected_generation': pair['generation'], 'selected_files': pair['files']}
        else:
            record = {'instance_id': task['instance_id'], 'repo': task['repo'], 'gate': 'quarantined',
                      'reasons': ['missing_complete_reference_pair'], 'evidence': [], 'selected_generation': None}
        records.append({**record, 'provenance': provenance.get(task['instance_id'], [])})
    metadata = {'rule_version': RULE_VERSION, 'finalizer_sha256': digest_file(__file__),
                'dataset_sha256': digest_file(source), 'dataset_count': len(tasks),
                'generations': sorted(set(generations)),
                'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    lists = {gate: [record['instance_id'] for record in records if record['gate'] == gate]
             for gate in ('eligible', 'quarantined', 'nondiscriminating')}
    summary = {**metadata, **{name: len(items) for name, items in lists.items()},
               'complete_pairs': len(selected), 'missing_pairs': len(tasks) - len(selected),
               'all_environments_verified': not lists['quarantined'],
               'all_tasks_discriminating': len(lists['eligible']) == len(tasks)}
    atomic_json(Path(root) / 'final-calibration.json', {**metadata, 'tasks': records})
    atomic_json(Path(root) / 'final-summary.json', summary)
    atomic_json(Path(root) / 'final-gate.json', {**summary, **{name + '_ids': items for name, items in lists.items()},
                                               'task_bindings': {r['instance_id']: {'profile': r.get('dependency_profile'),
                                                                                'fingerprint': r.get('record_fingerprint'),
                                                                                'freeze_sha256': r.get('profile_freeze_sha256'),
                                                                                'files': r.get('selected_files'),
                                                                                'required_profile_ids': r.get('required_profile_ids'),
                                                                                'matrix': r.get('matrix')}
                                                                 for r in records if r['gate'] == 'eligible'}})
    output = io.StringIO()
    fields = ['instance_id', 'repo', 'gate', 'selected_generation', 'gold_passed', 'empty_passed',
              'empty_failed', 'empty_errors', 'reasons', 'evidence']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow({**{key: record.get(key) for key in fields[:4]},
                         'gold_passed': record.get('gold_counts', {}).get('passed'),
                         'empty_passed': record.get('empty_counts', {}).get('passed'),
                         'empty_failed': record.get('empty_counts', {}).get('failed'),
                         'empty_errors': record.get('empty_counts', {}).get('errors'),
                         'reasons': ';'.join(record['reasons']), 'evidence': json.dumps(record['evidence'], ensure_ascii=False)})
    (Path(root) / 'calibration.csv').write_text(output.getvalue())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--generations', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if min(args.generations) < 1:
        parser.error('generations must be positive')
    print(json.dumps(finalize(Path(args.base).resolve(), Path(args.run_dir).resolve(), args.generations), indent=2))


if __name__ == '__main__':
    main()
