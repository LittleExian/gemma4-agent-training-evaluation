"""Pinned distributed experiment task runner with calibrated Landlock confinement.

Keeps the official evaluator and request recorder. The only source transforms
bind local paths, the manager, explicit backend routing, and patch verification.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from gemmacodeagent_full_task import normalize_result, replace_once, validate_endpoint

UPSTREAM_SHA = '82151d5202d4e10370c498a9217d463e9cca3dc47f98f1796494a4b41571429b'
SUBMISSION = 'mm15_6000'
SOURCE_RELATIVE = 'experiments/distributed_model_comparison/run_task.py'


def validate_arm(arm):
    if not isinstance(arm,str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}',arm) or '..' in arm:
        raise ValueError('An explicit safe served model name is required')
    return arm


def validate_budget(raw):
    required={'timeout_seconds':240,'max_time_minutes':15.,'max_tool_calls':56,'max_turns':160}
    if any(raw.get(key)!=value for key,value in required.items()):
        raise ValueError('Submission must specify 15 minutes, 56 tools, 160 turns, and 240 second tool timeout')


def validate_output(repo,output):
    if not output.is_relative_to(repo/'runs') or output == repo/'runs':
        raise ValueError('Task output must remain inside repository/runs')
    if any((output/name).exists() for name in ('result.json','task_results.jsonl')):
        raise ValueError('Refusing to overwrite an existing task invocation')


def adapt_source(source,context):
    if hashlib.sha256(source.encode()).hexdigest()!=UPSTREAM_SHA:
        raise ValueError('Audited upstream runner source changed')
    arm=validate_arm(context['arm'])
    endpoint=validate_endpoint(context['endpoint'])
    body={'thinking_token_budget':6000,'chat_template_kwargs':{'enable_thinking':True}}
    for before,after in (
        ('choices=["gold","empty","agent"]','choices=["gold","empty","agent","verify-patch"]'),
        ('out=ROOT/"runs"/args.run','out=Path('+repr(context['output_dir'])+')'),
        ('from autodl_chroot import ChrootManager','from gemmacodeagent_full_adapter import ChrootManager'),
        ('snapshots_dir=Path(os.environ.get("MM_SNAPSHOTS",str(ROOT/"data/matched56_snapshots")))',
         'snapshots_dir=Path('+repr(context['base']+'/data/snapshots')+')'),
        ('wheels_dir=ROOT/"data"/args.wheels','wheels_dir=Path('+repr(context['base']+'/data/wheels')+')'),
        ('agent_patch=t.patch if args.mode=="gold" else ""',
         'agent_patch=t.patch if args.mode=="gold" else (Path('+repr(context.get('patch_file') or '')+').read_text() if args.mode=="verify-patch" else "")'),
        ('"hardware":"AutoDL RTX PRO 6000 Blackwell 96GB TP1; AutoDL local chroot from calibrated Docker rootfs; separate calibration required; scorer TP4 L4 timing not identical"',
         '"hardware":"Mario RTX PRO 6000 TP1; calibrated Landlock adaptation; timing differs from official scorer", "arm":'+repr(arm)),
    ):
        source=replace_once(source,before,after)
    if source.count('ROOT/"data/tasks.jsonl"')!=2:
        raise ValueError('Upstream dataset anchors changed')
    source=source.replace('ROOT/"data/tasks.jsonl"','Path('+repr(context['base']+'/data/tasks.jsonl')+')')
    for original in ('model','adapter_name'):
        before='LiteLlm(model="openai/"+'+original+',api_base=os.environ["MM_PROXY_BASE"],api_key="EMPTY",num_retries=3)'
        after='LiteLlm(model="openai/"+'+repr(arm)+',api_base='+repr(endpoint)+',api_key="EMPTY",num_retries=3,extra_body='+repr(body)+')'
        source=replace_once(source,before,after)
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','repo','output-dir','task-id','endpoint','gate'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--arm',default=os.environ.get('G4_MM_ARM'))
    parser.add_argument('--mode',choices=['gold','empty','agent','verify-patch'],required=True)
    parser.add_argument('--profile-id')
    parser.add_argument('--patch-file')
    args=parser.parse_args()
    repo,base,output=[Path(getattr(args,key)).resolve() for key in ('repo','base','output_dir')]
    validate_output(repo,output)
    if args.run_dir and not output.is_relative_to(Path(args.run_dir).resolve()):
        raise ValueError('Task output must remain inside selected run')
    if args.mode=='verify-patch' and (not args.patch_file or not Path(args.patch_file).is_file()):
        raise ValueError('verify-patch requires an existing agent patch file')
    import yaml
    import calibrate_all_tasks as calibration
    from gemmacodeagent_full_adapter import profile_candidate,select_profile
    validate_budget(yaml.safe_load((repo/'submissions'/SUBMISSION/'eval_config.yaml').read_text()).get('evaluation',{}))
    gate=json.loads(Path(args.gate).read_text())
    candidate=profile_candidate(gate,args.task_id,args.profile_id)
    profile=select_profile(gate,args.task_id,args.profile_id,
        calibration.freeze(Path(candidate['environment'])/'bin/python'))
    context={'base':str(base),'repo':str(repo),'runtime':str(repo/'envs/eval'),
        'output_dir':str(output),'task_id':args.task_id,'profile_id':args.profile_id,
        'gate':str(Path(args.gate).resolve()),'endpoint':validate_endpoint(args.endpoint),
        'arm':validate_arm(args.arm),
        'patch_file':str(Path(args.patch_file).resolve()) if args.patch_file else None}
    source=(repo/SOURCE_RELATIVE).read_text()
    adapted=adapt_source(source,context)
    output.mkdir(parents=True,exist_ok=True)
    context_path=output/'invocation.json'
    calibration.atomic_json(context_path,context)
    os.environ.update(GEMMA_FULL_CONTEXT=str(context_path),CODEAGENT_ROOT=str(repo),
        PYTHONNOUSERSITE='1',OTEL_SDK_DISABLED='true',LITELLM_LOCAL_MODEL_COST_MAP='True',
        LITELLM_TELEMETRY='False')
    (output/'adapted_runner.py').write_text(adapted)
    inputs={'upstream_sha256':UPSTREAM_SHA,'task_source_sha256':calibration.sha(__file__),
        'adapter_sha256':calibration.sha(Path(__file__).with_name('gemmacodeagent_full_adapter.py')),
        'normalization_sha256':calibration.sha(Path(__file__).with_name('gemmacodeagent_full_task.py')),
        'native_guard_sha256':calibration.sha(Path(__file__).with_name('gemmacodeagent_native_guard.py')),
        'gate_sha256':calibration.sha(args.gate),'dataset_sha256':calibration.sha(base/'data/tasks.jsonl'),
        'snapshot_sha256':calibration.sha(base/'data/snapshots'/f'{args.task_id}.tgz'),
        'profile':profile,'task_id':args.task_id,'arm':context['arm'],'endpoint':context['endpoint'],
        'protocol':{'minutes':15,'tools':56,'turns':160,'thinking':6000,'output_tokens':8192,
                    'temperature':.2,'top_p':.95,'context':32768,'compaction_threshold':14336},
        'submission':{str(f.relative_to(repo/'submissions'/SUBMISSION)):calibration.sha(f)
            for f in sorted((repo/'submissions'/SUBMISSION).rglob('*')) if f.is_file()}}
    fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    calibration.atomic_json(output/'fingerprint.json',{'fingerprint':fingerprint,'inputs':inputs})
    sys.argv=['adapted_runner','--mode',args.mode,'--ids',args.task_id,'--submission',SUBMISSION,
              '--run',str(output.relative_to(repo/'runs'))]
    namespace={'__name__':'gemmacodeagent_audited_mm_runner','__file__':str(repo/SOURCE_RELATIVE)}
    exec(compile(adapted,str(output/'adapted_runner.py'),'exec'),namespace)
    asyncio.run(namespace['main']())
    rows=[json.loads(line) for line in (output/'task_results.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=1 or rows[0].get('task_id')!=args.task_id:
        raise ValueError('Expected exactly one matching upstream result')
    commands=json.loads((output/'adapter/commands.json').read_text())
    result=normalize_result(rows[0],output,profile,commands,fingerprint,args.mode)
    result['arm']=context['arm']
    calibration.atomic_json(output/'result.json',result)
    print(json.dumps({key:result.get(key) for key in ('instance_id','arm','resolved','agent_timed_out','infra_error')}),flush=True)

if __name__=='__main__':main()
