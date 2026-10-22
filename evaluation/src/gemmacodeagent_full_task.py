"""Run one upstream v5_6000 task with calibrated dependencies and explicit routing.

Transforms only audited configuration anchors; upstream evaluator, budget, request
recorder, cache/compaction and submission behavior remain intact. Fails closed if
that source changes. This is the documented non-root Mario adaptation.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

UPSTREAM_SHA = '0d380fb5247846dee7e5376055144fa6aa05b9fee4695f95776acf71bfa47e77'


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Upstream configuration anchor changed: ' + before[:90])
    return source.replace(before, after, 1)


def validate_endpoint(endpoint):
    value = urlsplit(endpoint)
    if (value.scheme != 'http' or value.hostname not in ('127.0.0.1','localhost')
            or not value.port or value.username or value.password or value.query
            or value.fragment or value.path != '/v1'):
        raise ValueError('An explicit loopback HTTP /v1 endpoint without credentials is required')
    return endpoint


def adapt_source(source, context):
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA:
        raise ValueError('Audited upstream runner source changed')
    for before,after in (
        ('out=ROOT/"runs"/args.run', 'out=Path('+repr(context['output_dir'])+')'),
        ('choices=["gold","empty","agent"]', 'choices=["gold","empty","agent","verify-patch"]'),
        ('api_base="http://127.0.0.1:8106/v1"', 'api_base='+repr(context['endpoint'])),
    ):
        # The API base occurs for the base model and optional adapter registry.
        if before.startswith('api_base='):
            if source.count(before) != 2:
                raise ValueError('Upstream model registry anchors changed')
            source=source.replace(before,after)
        else:
            source=replace_once(source,before,after)
    source=replace_once(source,'from gemmacodeagent_mario_adapter import ChrootManager',
                         'from gemmacodeagent_full_adapter import ChrootManager')
    source=source.replace('ROOT/"data/tasks.jsonl"','Path('+repr(context['base']+'/data/tasks.jsonl')+')')
    source=replace_once(source,'ROOT/"data/matched56_snapshots"','Path('+repr(context['base']+'/data/snapshots')+')')
    source=replace_once(source,'ROOT/"data"/args.wheels','Path('+repr(context['base']+'/data/wheels')+')')
    source=replace_once(source,'agent_patch=t.patch if args.mode=="gold" else ""',
        'agent_patch=t.patch if args.mode=="gold" else (Path('+repr(context.get('patch_file') or '')+').read_text() if args.mode=="verify-patch" else "")')
    return source


def normalize_result(row, output, profile, commands, fingerprint, mode):
    data={key:value for key,value in row.items() if key!='agent_patch'}
    patch=row.get('agent_patch') or ''
    patch_path=output/'agent.patch'
    if mode=='agent':
        patch_path.write_text(patch)
    error=str(row.get('error_message') or '').lower()
    timed_out=mode=='agent' and (str(row.get('status')).upper() in ('TIMEOUT','BUDGET_EXCEEDED')
        or 'session timed out' in error or 'session timeout' in error or 'time budget' in error)
    data.update(instance_id=row.get('task_id'),dependency_profile=profile,commands=commands,
        environment_probes=[entry['environment'] for entry in commands if 'environment' in entry],
        fingerprint=fingerprint,patch_file=str(patch_path) if mode=='agent' else None,
        patch_chars=len(patch),agent_timed_out=timed_out,
        error_code='agent_budget_timeout' if timed_out else None,
        verification={key:row.get(key) for key in ('resolved','test_exit_code','test_output','error_message')})
    # Classify infrastructure only on explicit harness error, never plain test failure.
    data['infra_error']=bool(row.get('traceback') or (not timed_out and (
        str(row.get('status')).upper() == 'ERROR' or (row.get('error_message')
        and row.get('test_exit_code') is None))))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','repo','run-dir','output-dir','task-id','endpoint'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gate',default='/workspace/gemma4-env-preflight-20261016/final-gate.json')
    parser.add_argument('--mode',choices=['gold','empty','agent','verify-patch'],required=True)
    parser.add_argument('--profile-id')
    parser.add_argument('--patch-file')
    args=parser.parse_args()
    repo,base,run,output=[Path(getattr(args,key)).resolve() for key in ('repo','base','run_dir','output_dir')]
    if not output.is_relative_to(run) or not run.is_relative_to(repo/'runs'):
        raise ValueError('Run and output must remain inside repository/runs')
    if args.mode=='verify-patch' and (not args.patch_file or not Path(args.patch_file).is_file()):
        raise ValueError('verify-patch requires an existing agent patch file')
    if (output/'result.json').exists() or (output/'task_results.jsonl').exists():
        raise ValueError('Refusing to overwrite an existing task invocation')
    output.mkdir(parents=True,exist_ok=True)
    import calibrate_all_tasks as calibration
    from gemmacodeagent_full_adapter import profile_candidate,select_profile
    gate=json.loads(Path(args.gate).read_text())
    candidate=profile_candidate(gate,args.task_id,args.profile_id)
    profile=select_profile(gate,args.task_id,args.profile_id,
        calibration.freeze(Path(candidate['environment'])/'bin/python'))
    context={'base':str(base),'repo':str(repo),'runtime':str(repo/'envs/eval'),
        'output_dir':str(output),'task_id':args.task_id,'profile_id':args.profile_id,
        'gate':str(Path(args.gate).resolve()),'endpoint':validate_endpoint(args.endpoint),
        'patch_file':str(Path(args.patch_file).resolve()) if args.patch_file else None}
    context_path=output/'invocation.json'
    calibration.atomic_json(context_path,context)
    os.environ.update(GEMMA_FULL_CONTEXT=str(context_path),PYTHONNOUSERSITE='1',
        OTEL_SDK_DISABLED='true',LITELLM_LOCAL_MODEL_COST_MAP='True',LITELLM_TELEMETRY='False')
    source=(repo/'src/run_eval_autodl_matched56.py').read_text()
    adapted=adapt_source(source,context)
    (output/'adapted_runner.py').write_text(adapted)
    inputs={'upstream_sha256':UPSTREAM_SHA,'task_source_sha256':calibration.sha(__file__),
        'adapter_sha256':calibration.sha(Path(__file__).with_name('gemmacodeagent_full_adapter.py')),
        'native_guard_sha256':calibration.sha(Path(__file__).with_name('gemmacodeagent_native_guard.py')),
        'gate_sha256':calibration.sha(args.gate),'dataset_sha256':calibration.sha(base/'data/tasks.jsonl'),
        'snapshot_sha256':calibration.sha(base/'data/snapshots'/f'{args.task_id}.tgz'),
        'profile':profile,'task_id':args.task_id,
        'submission':{str(f.relative_to(repo/'submissions/v5_6000')):calibration.sha(f)
            for f in sorted((repo/'submissions/v5_6000').rglob('*')) if f.is_file()}}
    fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    calibration.atomic_json(output/'fingerprint.json',{'fingerprint':fingerprint,'inputs':inputs})
    sys.argv=['adapted_runner','--mode',args.mode,'--ids',args.task_id,'--submission','v5_6000',
        '--run',str(output.relative_to(repo/'runs'))]
    namespace={'__name__':'gemmacodeagent_audited_runner','__file__':str(repo/'src/run_eval_autodl_matched56.py')}
    exec(compile(adapted,str(output/'adapted_runner.py'),'exec'),namespace)
    asyncio.run(namespace['main']())
    rows=[json.loads(line) for line in (output/'task_results.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=1 or rows[0].get('task_id')!=args.task_id:
        raise ValueError('Expected exactly one matching upstream result')
    commands=json.loads((output/'adapter/commands.json').read_text())
    result=normalize_result(rows[0],output,profile,commands,fingerprint,args.mode)
    calibration.atomic_json(output/'result.json',result)
    print(json.dumps({key:result.get(key) for key in ('instance_id','resolved','agent_timed_out','infra_error')}),flush=True)

if __name__=='__main__':
    main()
