"""Resume the complete official competition input download on the server."""
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFileRequest

BASE = Path('/workspace/gemma4-notebook-run-20261008')
DATA = BASE / 'data'
MANIFEST = json.loads((DATA / 'manifest.json').read_text())
PARTS = BASE / 'full-download-parts'


def download_parallel(api, entry, destination):
    request = ApiDownloadDataFileRequest()
    request.competition_name = 'gemma-4-developer-agent'
    request.file_name = entry['name']
    with api.build_kaggle_client() as client:
        response = client.competitions.competition_api_client.download_data_file(request)
    url = response.url
    response.close()
    # Signed storage URLs remain in process memory and the downloader's stdin.
    # Never write them into logs, command arguments, or persisted input files.
    if not (urlsplit(url).hostname or '').endswith('.googleapis.com'):
        api.competition_download_file('gemma-4-developer-agent', entry['name'],
                                     path=str(destination.parent), quiet=True)
        return
    partial = PARTS / entry['name']
    partial.parent.mkdir(parents=True, exist_ok=True)
    args = ['/usr/bin/aria2c', '--input-file=-', '--split=16',
            '--max-connection-per-server=16', '--min-split-size=1M',
            '--file-allocation=none', '--continue=true', '--auto-file-renaming=false',
            '--allow-overwrite=true', '--max-tries=5', '--retry-wait=3',
            '--timeout=120', '--quiet=true', '--console-log-level=error',
            '--download-result=hide', '--dir=' + str(partial.parent),
            '--out=' + partial.name]
    result = subprocess.run(args, input=url + '\n', text=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode or not partial.is_file() or partial.stat().st_size != entry['size']:
        raise RuntimeError(f'Multipart transfer incomplete: {entry["name"]}')
    os.replace(partial, destination)


def fetch(entry):
    relative = Path(entry['name'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Invalid manifest path: {relative}')
    destination = DATA / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == entry['size']:
        return entry['name'], 'cached'
    for attempt in range(5):
        try:
            api = KaggleApi()
            api.authenticate()
            download_parallel(api, entry, destination)
            if destination.stat().st_size != entry['size']:
                raise RuntimeError(f"Size mismatch: {entry['name']}")
            return entry['name'], 'downloaded'
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))


def main():
    print(f'Full manifest: {len(MANIFEST)} files / '
          f"{sum(row['size'] for row in MANIFEST)} bytes", flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(fetch, entry): entry for entry in MANIFEST}
        for index, future in enumerate(concurrent.futures.as_completed(pending), 1):
            entry = pending[future]
            try:
                name, state = future.result()
                print(f'[{index}/{len(MANIFEST)}] {state}: {name}', flush=True)
            except Exception as exc:
                failures.append({'name': entry['name'], 'error_type': type(exc).__name__})
                print(f'[{index}/{len(MANIFEST)}] FAILED {entry["name"]}: '
                      f'{type(exc).__name__}', flush=True)
    missing = [entry['name'] for entry in MANIFEST
               if not (DATA / entry['name']).is_file()
               or (DATA / entry['name']).stat().st_size != entry['size']]
    report = {'total_files': len(MANIFEST), 'total_bytes': sum(x['size'] for x in MANIFEST),
              'missing_or_wrong_size': missing, 'failures': failures,
              'complete': not missing}
    (DATA / 'full-download-status.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
