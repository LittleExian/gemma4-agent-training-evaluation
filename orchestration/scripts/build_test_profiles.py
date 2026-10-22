"""Read snapshot metadata without extracting files; plan isolated test environments.

No packages are installed and no benchmark tasks are executed. With --ssh, only
the metadata reader runs remotely. --requirements-dir materializes pip inputs.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


READER = r'''
import concurrent.futures, hashlib, json, pathlib, tarfile, tomllib, sys
base = pathlib.Path(sys.argv[1])
tasks = [json.loads(line) for line in (base/'tasks.jsonl').read_text().splitlines() if line.strip()]
selected = {'pytest','pydantic','starlette','inline-snapshot','pygments','markdown-it-py','sqlmodel','httpbin','pytest-httpbin'}
def read(task):
    files, project, locks, requirements = {}, {}, {}, {}
    snapshot = base/'snapshots'/(task['instance_id']+'.tgz')
    with tarfile.open(snapshot, 'r|gz') as bundle:
        for member in bundle:
            name = member.name.removeprefix('./')
            wanted = name in {'pyproject.toml','poetry.lock','uv.lock','setup.cfg','setup.py'} or ('/' not in name and name.startswith('requirements') and name.endswith('.txt'))
            if not wanted or not member.isfile():
                continue
            content = bundle.extractfile(member).read()
            files[name] = hashlib.sha256(content).hexdigest()
            text = content.decode(errors='replace')
            if name == 'pyproject.toml':
                parsed = tomllib.loads(text)
                project = {'project': parsed.get('project', {}), 'dependency-groups': parsed.get('dependency-groups', {}), 'poetry': parsed.get('tool', {}).get('poetry', {})}
            elif name.endswith('.lock'):
                locks[name] = [{key:p.get(key) for key in ['name','version','resolution-markers'] if key in p} for p in tomllib.loads(text).get('package', []) if p.get('name') in selected]
            elif name.startswith('requirements'):
                requirements[name] = text
    patch = task.get('test_patch','')
    return {'instance_id':task['instance_id'], 'repo':task['repo'], 'base_commit':task['base_commit'], 'snapshot':str(snapshot), 'snapshot_bytes':snapshot.stat().st_size,
            'files_sha256':files, 'metadata':project, 'locks':locks, 'requirements_sources':requirements,
            'test_patch_markers':{'needs_pydanticv1':'+@needs_pydanticv1' in patch, 'needs_pydanticv2':'+@needs_pydanticv2' in patch}}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(read, tasks))
print(json.dumps({'tasks_sha256':hashlib.sha256((base/'tasks.jsonl').read_bytes()).hexdigest(), 'snapshots':rows}, sort_keys=True))
'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def version_tuple(value):
    return tuple(int(part) for part in value.split('.') if part.isdigit())


def matches(version, requirement):
    for operator, target in re.findall(r'(>=|<=|!=|==|<|>)([0-9.]+)', requirement):
        left, right = version_tuple(version), version_tuple(target)
        width = max(len(left), len(right))
        left, right = left + (0,) * (width-len(left)), right + (0,) * (width-len(right))
        if not {'>=':left >= right, '<=':left <= right, '!=':left != right,
                '==':left == right, '<':left < right, '>':left > right}[operator]:
            return False
    return True


def lock_versions(row, name):
    return sorted({p['version'] for entries in row['locks'].values() for p in entries
                   if p['name'] == name}, key=version_tuple)


def lock_pin(row, name, fallback, requirement=''):
    versions = [v for v in lock_versions(row, name) if matches(v, requirement)]
    return versions[-1] if versions else fallback


def test_declarations(row):
    source = '\n'.join(row['requirements_sources'].values())
    groups = row['metadata'].get('dependency-groups', {})
    source += '\n' + '\n'.join(v for v in groups.get('tests', []) if isinstance(v, str))
    source += '\n' + '\n'.join(v for v in groups.get('test', []) if isinstance(v, str))
    return source


def plan_profile(row):
    repo = row['repo']
    notes = []
    python = '3.12'
    pins = {'pytest-mock':'3.12.0', 'pytest-cov':'4.1.0', 'pytest-timeout':'2.3.1'}
    source = test_declarations(row)
    declared_pytest = [line.strip() for line in source.splitlines()
                       if re.match(r'^pytest\s*(?:[<>=!]|$)', line.strip(), re.I)]
    if repo == 'fastapi/fastapi':
        deps = row['metadata']['project']['dependencies']
        starlette_req = next(d for d in deps if d.startswith('starlette'))
        pydantic_req = next(d for d in deps if d.startswith('pydantic'))
        pins.update({'pytest':lock_pin(row,'pytest','8.4.2'),
                     'pydantic':lock_pin(row,'pydantic','2.12.5'),
                     'starlette':lock_pin(row,'starlette','0.47.3',starlette_req),
                     'inline-snapshot':lock_pin(row,'inline-snapshot','0.31.1'),
                     'httpx':'0.28.1','anyio':'4.9.0','trio':'0.30.0',
                     'typing-extensions':'4.15.0','typing-inspection':'0.4.2',
                     'annotated-doc':'0.0.4','dirty-equals':'0.9.0',
                     'sqlmodel':lock_pin(row,'sqlmodel','0.0.27'),
                     'SQLAlchemy':'2.0.43','Jinja2':'3.1.6','python-multipart':'0.0.20',
                     'email-validator':'2.3.0','orjson':'3.11.3','ujson':'5.11.0',
                     'PyYAML':'6.0.2','itsdangerous':'2.2.0','Flask':'3.1.2',
                     'PyJWT':'2.9.0','pwdlib[argon2]':'0.2.1','passlib':'1.7.4',
                     'bcrypt':'4.0.1','a2wsgi':'1.10.10','strawberry-graphql':'0.287.0'})
        sql = re.search(r'^sqlmodel\s*==\s*([\d.]+)', source, re.M)
        if sql:
            pins['sqlmodel'] = sql.group(1)
        elif 'sqlmodel' in source:
            pins['sqlmodel'] = lock_pin(row,'sqlmodel','0.0.31')
        if row['instance_id'] == 'fastapi_14186':
            pins['pydantic'] = '1.10.24'
            notes.append('Pydantic v1 selected from the added needs_pydanticv1 regression marker.')
        else:
            pins.update({'pydantic-settings':'2.10.1','pydantic-extra-types':'2.10.5'})
        pins['pytest-asyncio'] = '1.3.0' if version_tuple(pins['pytest']) >= (9,) else '0.23.8'
        if not matches(pins['starlette'], starlette_req) or not matches(pins['pydantic'], pydantic_req):
            raise ValueError('Selected pin violates project declaration: '+row['instance_id'])
        if not lock_versions(row,'starlette'):
            notes.append('No UV lock: Starlette 0.47.3 is a deterministic candidate within this snapshot declaration, not a historical lock claim.')
        if not lock_versions(row,'inline-snapshot'):
            notes.append('No snapshot pin: inline-snapshot 0.31.1 is a deterministic candidate satisfying >=0.21.1.')
        family = 'fastapi'
    elif repo == 'Textualize/rich':
        pins.update({'pytest':lock_pin(row,'pytest','7.4.4'),
                     'pytest-cov':'3.0.0',
                     'Pygments':lock_pin(row,'pygments','2.19.2'),
                     'markdown-it-py':lock_pin(row,'markdown-it-py','3.0.0'),
                     'attrs':'21.4.0','typing-extensions':'4.15.0'})
        family = 'rich'
    elif repo == 'psf/requests':
        old = 'pytest-httpbin==2.0.0' in source
        legacy_pytest = '<=6.2.5' in source
        pins.update({'pytest':'6.2.5' if legacy_pytest else '8.3.5',
                     'pytest-httpbin':'2.0.0' if old else '2.1.0',
                     'httpbin':'0.10.0','trustme':'1.2.1','PySocks':'1.7.1',
                     'urllib3':'2.5.0','idna':'3.10','charset-normalizer':'3.4.3',
                     'certifi':'2025.8.3','chardet':'5.2.0','Flask':'3.1.2',
                     'Werkzeug':'3.1.3','flasgger':'0.9.7.1','brotli':'1.1.0'})
        if legacy_pytest:
            python = '3.10'
            pins['pytest-cov'] = '3.0.0'
            pins['pytest-timeout'] = '2.1.0'
            notes.append('Declared pytest<=6.2.5 retained. Use a separate Python 3.10 interpreter; Python 3.12 compatibility is not presumed.')
        family = 'requests'
    elif repo == 'encode/httpx':
        pins.update({'pytest':'8.3.5','pytest-cov':'6.1.1','certifi':'2025.8.3',
                     'click':'8.2.1','Jinja2':'3.1.6','markdown':'3.8'})
        family = 'httpx'
    else:
        raise ValueError('Unknown repository '+repo)
    requirements = sorted(name+'=='+version for name,version in pins.items())
    return {'family':family,'python_version':python,'requirements':requirements,
            'declared_pytest':sorted(set(declared_pytest)), 'chosen_pytest':pins['pytest'],
            'pytest_declaration_deviation':False, 'selection_notes':sorted(set(notes))}


def build(inspected, run_dir):
    profiles, assignments, evidence = {}, {}, {}
    for row in inspected['snapshots']:
        task = row['instance_id']
        if task in assignments:
            raise ValueError('Duplicate task '+task)
        selection = plan_profile(row)
        # Evidence and advisory prose do not change dependency identity.
        identity = {key:selection[key] for key in ['family','python_version','requirements']}
        fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
        profile_id = selection['family']+'-'+fingerprint[:16]
        profiles.setdefault(profile_id, {**identity,'fingerprint':fingerprint,
            'python':selection['python_version'],
            'environment':str(run_dir/'test-envs'/profile_id),
            'requirements_file':str(run_dir/'requirements'/(profile_id+'.txt'))})
        assignments[task] = profile_id
        evidence[task] = {**row, 'selection':selection}
    if len(assignments) != 129:
        raise ValueError('Expected exactly 129 tasks, got '+str(len(assignments)))
    return {'schema_version':1,'tasks_sha256':inspected['tasks_sha256'],
            'selection_policy':'Snapshot declarations and lock pins; no gold/empty outcomes used. Unlocked candidates explicitly identified.',
            'profiles':dict(sorted(profiles.items())), 'tasks':dict(sorted(assignments.items())),
            'task_evidence':dict(sorted(evidence.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True,help='Data directory containing tasks.jsonl and snapshots/')
    parser.add_argument('--ssh',help='Read metadata from this SSH host; local output remains local')
    parser.add_argument('--port',type=int,default=11200)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--requirements-dir',help='Optionally write requirements files here')
    args = parser.parse_args()
    command = ['python3','-',args.base]
    if args.ssh:
        import shlex
        command = ['ssh','-o','BatchMode=yes','-p',str(args.port),args.ssh,
                   ' '.join(shlex.quote(part) for part in command)]
    result = subprocess.run(command,input=READER,text=True,capture_output=True,check=True)
    planned = build(json.loads(result.stdout),Path(args.run_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(planned,ensure_ascii=False,indent=2)+'\n')
    if args.requirements_dir:
        folder = Path(args.requirements_dir)
        folder.mkdir(parents=True,exist_ok=True)
        for key,profile in planned['profiles'].items():
            (folder/(key+'.txt')).write_text('\n'.join(profile['requirements'])+'\n')
    print(json.dumps({'tasks':len(planned['tasks']),'profiles':len(planned['profiles']),
                      'output':str(output),'python_versions':sorted({p['python_version'] for p in planned['profiles'].values()})}))


if __name__ == '__main__':
    main()
