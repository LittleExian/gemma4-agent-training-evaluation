"""Transport the exact HF-pinned BF16 model through a hash-checked mirror."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from gemmacodeagent_mm_download import (
    atomic_json, download_file, file_urls, identity, safe_directory,
    safe_name, verified,
)

MODEL = 'google/gemma-4-31B-it-qat-q4_0-unquantized'
REVISION = '1e4d8beecacb8b7590c1d8bedd7335f687bf311f'


def locked_download(destination, info, urls, workers):
    """Serialize independent metadata/weight transports for each destination."""
    lock_path = destination.with_name(destination.name + '.download.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        download_file(destination, info, urls, workers=workers)


def trusted_files(tree):
    result = []
    for item in tree:
        if item['type'] != 'file':
            continue
        safe_name(item['path'])
        info = {'rfilename': item['path'], 'size': item['size'],
                'blobId': item['oid']}
        if item.get('lfs'):
            if item['lfs']['size'] != item['size']:
                raise ValueError('HF LFS size mismatch')
            info['lfs'] = {'sha256': item['lfs']['oid'], 'size': item['size']}
        identity(info)
        result.append(info)
    if not result or len({x['rfilename'] for x in result}) != len(result):
        raise ValueError('Empty or duplicate manifest')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('Invalid worker count')
    destination = safe_directory(args.destination)
    raw = args.manifest.read_bytes()
    files = trusted_files(json.loads(raw))
    source = {'repo': MODEL, 'revision': REVISION}
    with (destination.parent / 'model-download.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        errors = []
        # Start the large immutable weights first. Small metadata can be retried
        # independently if a mirror edits its model card.
        ordered = sorted(files, key=lambda item: item['size'], reverse=True)
        for info in ordered:
            name = info['rfilename']
            print(json.dumps({'event': 'start', 'file': name,
                              'bytes': info['size'], 'time': time.time()}), flush=True)
            try:
                locked_download(destination / name, info,
                                file_urls(source, {}, info, ['modelscope']),
                                workers=args.workers)
                if not verified(destination / name, info):
                    raise ValueError('Post-download identity mismatch')
                completed.append({'file': name, **identity(info)})
                print(json.dumps({'event': 'verified', 'file': name,
                                  'time': time.time()}), flush=True)
            except Exception as exc:
                errors.append({'file': name, 'error_type': type(exc).__name__})
                print(json.dumps({'event': 'failed', **errors[-1]}), flush=True)
            atomic_json(destination.parent / 'model-download-progress.json',
                        {'completed': completed, 'errors': errors,
                         'total_files': len(files), 'time': time.time()})
        if errors:
            raise RuntimeError('Some files failed; see model-download-progress.json')
        atomic_json(destination / 'DOWNLOAD_COMPLETE.json',
                    {'repo': MODEL, 'revision': REVISION,
                     'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                     'transport': 'modelscope master; identity from pinned HF tree',
                     'files': completed, 'completed_unix': time.time()})


if __name__ == '__main__':
    main()
