"""Verify golden patches through the same confined Phase 2 as the full run."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

TASK_IDS = ('fastapi_15661', 'rich_4070', 'requests_7505', 'httpx_3672')


def download(base, run_dir):
    from kaggle.api.kaggle_api_extended import KaggleApi

    manifest = json.loads((base / 'data/manifest.json').read_text())
    selected = {f'snapshots/{task}.tgz' for task in TASK_IDS}
    api = KaggleApi()
    api.authenticate()
    staging = run_dir / 'preflight-download'
    staging.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        if entry['name'] not in selected:
            continue
        target = base / 'data' / entry['name']
        if target.exists() and target.stat().st_size == entry['size']:
            print(f'CACHED {entry["name"]}', flush=True)
            continue
        api.competition_download_file('gemma-4-developer-agent', entry['name'],
                                     path=str(staging), quiet=True)
        source = staging / Path(entry['name']).name
        if source.stat().st_size != entry['size']:
            raise ValueError(f'Size mismatch: {entry["name"]}')
        temporary = target.with_suffix('.preflight.tmp')
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
        print(f'DOWNLOADED {entry["name"]}', flush=True)


async def verify(base, run_dir, task_ids=TASK_IDS):
    from budget_full_run import atomic_json, configure_environment, install_confinement

    shard = run_dir / 'preflight'
    configure_environment(shard, base)
    from swegemma.config import EvalConfig
    from swegemma.harness.verification import verify_task
    from swegemma.models.task import load_tasks
    from swegemma.sandbox.subprocess import SubprocessManager

    config = EvalConfig(tasks_path=base / 'data/tasks.jsonl',
                        snapshots_dir=base / 'data/snapshots', results_dir=shard / 'results',
                        submission_dir=shard / 'agent', models=None, sandbox='subprocess',
                        wheels_dir=base / 'data/wheels', timeout_seconds=300)
    install_confinement(base, shard)
    by_id = {task.instance_id: task for task in load_tasks(config.tasks_path)}
    records = [json.loads(path.read_text())
               for path in sorted((shard / 'results').glob('*.json'))
               if path.stem in TASK_IDS and path.stem not in task_ids]
    for instance_id in task_ids:
        task = by_id[instance_id]
        print(f'PREFLIGHT START {instance_id}', flush=True)
        manager = SubprocessManager(timeout_seconds=300, base_dir=shard / 'tmp')
        started = time.perf_counter()
        try:
            result = await verify_task(manager, config, task,
                                       config.snapshots_dir / f'{instance_id}.tgz',
                                       agent_patch=task.patch, start_time=started)
            record = result.model_dump(mode='json', exclude={'agent_patch', 'trace'})
        except Exception as exc:
            record = {'resolved': False, 'test_exit_code': None, 'test_output': '',
                      'error_message': f'{type(exc).__name__}: {exc}',
                      'traceback': traceback.format_exc()}
        output = record.get('test_output', '')
        record = {**record, 'instance_id': instance_id, 'repo': task.repo,
                  'golden_patch': True, 'duration_seconds': time.perf_counter() - started,
                  'pytest_summary': re.findall(r'[^\n]*(?:passed|failed|error|collected|skipped)[^\n]*', output),
                  'missing_modules': sorted(set(re.findall(r"No module named ['\"]([^'\"]+)", output)))}
        atomic_json(shard / 'results' / f'{instance_id}.json', record)
        records.append(record)
        atomic_json(shard / 'summary.json', records)
        print(f'PREFLIGHT DONE {instance_id} resolved={record.get("resolved")} '
              f'exit={record.get("test_exit_code")} missing={record["missing_modules"]}', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--task-ids', nargs='+', choices=TASK_IDS, default=list(TASK_IDS))
    args = parser.parse_args()
    if args.download_only:
        download(args.base.resolve(), args.run_dir.resolve())
    else:
        asyncio.run(verify(args.base.resolve(), args.run_dir.resolve(), args.task_ids))


if __name__ == '__main__':
    main()
