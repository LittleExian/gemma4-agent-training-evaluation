"""Per-task calibrated environment for the upstream runner, without root."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

from gemmacodeagent_mario_adapter import install_confinement, require_supported_verification
from gemmacodeagent_native_guard import install_native_guards


def profile_candidate(gate, task_id, profile_id):
    if task_id not in gate.get('eligible_ids', []):
        raise ValueError('Task not eligible in reference gate: ' + task_id)
    binding = gate.get('task_bindings', {}).get(task_id, {})
    if binding.get('required_profile_ids'):
        if profile_id not in binding['required_profile_ids']:
            raise ValueError('Explicit required matrix profile must be selected')
        matches = [arm['dependency_profile'] for arm in binding.get('matrix', {}).get('arms', [])
                   if arm.get('profile_id') == profile_id and arm.get('dependency_profile')]
        if len(matches) != 1:
            raise ValueError('Missing or duplicate matrix profile binding')
        return dict(matches[0])
    profile = binding.get('profile', {})
    if profile_id and profile_id != profile.get('profile_id'):
        raise ValueError('Profile override is not bound by calibration gate')
    return dict(profile)


def select_profile(gate, task_id, profile_id, actual_freeze):
    profile = profile_candidate(gate, task_id, profile_id)
    if not profile.get('environment') or not actual_freeze or profile.get('actual_freeze') != actual_freeze:
        raise ValueError('Dependency profile missing or installed versions changed')
    return profile


def ChrootManager(timeout_seconds=300, **kwargs):
    import calibrate_all_tasks as calibration
    import calibrate_all_tasks_v2 as calibration_v2
    import swegemma.sandbox.subprocess as backend
    context = json.loads(Path(os.environ['GEMMA_FULL_CONTEXT']).read_text())
    base, repo, runtime, output = [Path(context[key]).resolve() for key in ('base','repo','runtime','output_dir')]
    if Path(sys.prefix).resolve() != runtime:
        raise ValueError('Independent eval interpreter required')
    gate_path = Path(context['gate'])
    gate = json.loads(gate_path.read_text())
    if calibration.sha(base/'data/tasks.jsonl') != gate['dataset_sha256']:
        raise ValueError('Task dataset changed since reference gate')
    task_id = context['task_id']
    candidate = profile_candidate(gate, task_id, context.get('profile_id'))
    profile = select_profile(gate, task_id, context.get('profile_id'),
                             calibration.freeze(Path(candidate['environment'])/'bin/python'))
    tasks = [json.loads(line) for line in (base/'data/tasks.jsonl').read_text().splitlines() if line.strip()]
    matches = [task for task in tasks if task['instance_id'] == task_id]
    if len(matches) != 1:
        raise ValueError('Expected one task record')
    shard = output/'adapter'
    (shard/'tmp').mkdir(parents=True,exist_ok=True)
    transcript = []
    install_confinement(backend,base,shard,runtime,Path(profile['environment']))
    manager = calibration_v2.with_declared_async_backend(backend.SubprocessManager,shard,
                Path(profile['environment']),transcript,matches[0]['repo'].rsplit('/',1)[-1])
    manager.timeout_seconds = timeout_seconds
    manifest = {'environment':'mario-landlock-profile-adaptation-not-upstream-chroot',
        'task_id':task_id,'profile':profile,'eval_runtime':str(runtime),
        'gate_sha256':calibration.sha(gate_path),'adapter_sha256':calibration.sha(__file__),
        'native_guard_sha256':calibration.sha(Path(__file__).with_name('gemmacodeagent_native_guard.py')),
        'upstream_constructor_kwargs':kwargs}
    calibration.atomic_json(shard/'manifest.json',manifest)
    execute = manager.exec
    def logged(identifier,command,**options):
        try:
            require_supported_verification(command)
            return execute(identifier,command,**options)
        finally:
            calibration.atomic_json(shard/'commands.json',transcript)
    manager.exec = logged
    install_native_guards(manager,shard)
    return manager
