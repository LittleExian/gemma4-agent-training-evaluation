"""Pinned, resumable, hash-checked model transport; never imports model code."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import time
import urllib.parse
import urllib.request

COMMIT = '29984e6638ef6825ad2636164d59df972c19d0f5'
SOURCE_BLOBS = {
    'qwen35_9b': 'f53832c261e8a6a28081717fdb579314874c1d7d',
    'qwen35_adapter': 'f144e379e9c4e131bce1cee15a0c1a670c90d09c',
    'qwen38_base': '0f121003120815901cf2c386939fde98ccf74fbb',
    'qwen38_coder390': 'b78b4df0b2a570795c6f94e111851dad430ae543',
    'gemma4_teich_agent': 'abbc797e940d9d2242cc481c3b7d2de6c774f4dd',
    'gemma4_capybara': '030b256f0231b14065ad7584676a1e4038d1ab10',
}
MODELS_BLOB = '0cbf913f1182adb3fb4598dc17a1985197050a5d'


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def regular(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError('Refusing non-regular, linked, or shared file: ' + path.name)
    return s


def safe_directory(path):
    path = Path(os.path.abspath(path))
    for parent in reversed((path, *path.parents)):
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError('Refusing linked or non-directory cache component')
        else:
            parent.mkdir()
    return path


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '\\' in name or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('Invalid repository filename')
    return p


def selected_files(source):
    selected = []
    sub = source['subfolder']
    if sub:
        safe_name(sub.rstrip('/'))
    for item in source['files']:
        name = item['rfilename']
        safe_name(name)
        if not name.startswith(sub) or '/' in name[len(sub):]:
            continue
        base = PurePosixPath(name).name
        if base.endswith(('.json', '.safetensors', '.jinja', '.txt', '.model')) or base in ('LICENSE', 'SHA256SUMS'):
            identity(item)
            selected.append(item)
    if not selected or not any(x['rfilename'].endswith('.safetensors') for x in selected):
        raise ValueError('Manifest has no safe model weights')
    return selected


def identity(info):
    size = info.get('size', info.get('lfs', {}).get('size'))
    digest = info.get('lfs', {}).get('sha256')
    algorithm = 'sha256' if digest else 'git-blob-sha1'
    digest = digest or info.get('blobId')
    expected_len = 64 if algorithm == 'sha256' else 40
    if not isinstance(size, int) or size < 0 or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{' + str(expected_len) + '}', digest):
        raise ValueError('Manifest requires trusted size and hash')
    return {'size': size, 'algorithm': algorithm, 'digest': digest}


def verified(path, info):
    expected = identity(info)
    if not path.exists() and not path.is_symlink():
        return False
    if regular(path).st_size != expected['size']:
        return False
    h = hashlib.new('sha256' if expected['algorithm'] == 'sha256' else 'sha1')
    if expected['algorithm'] == 'git-blob-sha1':
        h.update(b'blob ' + str(expected['size']).encode() + b'\0')
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest() == expected['digest']


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    if tmp.exists() or tmp.is_symlink():
        regular(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() or path.is_symlink():
        regular(path)
    tmp.replace(path)


def file_urls(source, model, info, hosts):
    if not re.fullmatch('[0-9a-f]{40}', source['revision']):
        raise ValueError('Unpinned revision refused')
    repo = source['repo']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Invalid repository ID')
    filename = urllib.parse.quote(str(safe_name(info['rfilename'])), safe='/')
    result = []
    for host in hosts:
        if host == 'modelscope':
            msrepo = model.get('modelscope_repo', repo)
            if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', msrepo):
                raise ValueError('Invalid mirror repository ID')
            # Mirror master is transport only: the pinned HF hash remains mandatory.
            result.append(f'https://modelscope.cn/models/{msrepo}/resolve/master/{filename}')
        elif host in ('hf', 'hf-mirror'):
            origin = 'huggingface.co' if host == 'hf' else 'hf-mirror.com'
            result.append(f'https://{origin}/{repo}/resolve/{source["revision"]}/{filename}')
        else:
            raise ValueError('Unknown transport')
    return result


def fetch(urls, start, end, size, retries, timeout):
    for url in urls:
        for attempt in range(retries):
            try:
                headers = {'User-Agent': 'pinned-model-verifier/1', 'Accept-Encoding': 'identity'}
                if start is not None:
                    headers['Range'] = f'bytes={start}-{end}'
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if start is not None:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Unexpected byte range')
                        limit = end - start + 1
                    else:
                        if response.status != 200:
                            raise ValueError('Unexpected HTTP status')
                        limit = size
                    data = response.read(limit + 1)
                    if len(data) != limit:
                        raise ValueError('Unexpected body size')
                    return data
            except Exception:
                if attempt + 1 < retries:
                    time.sleep(min(8, attempt + 1))
    # Deliberately omit exception text, signed redirect URLs and headers.
    raise RuntimeError('All bounded transport retries failed')


def download_file(dest, info, urls, chunk_size=8 * 1024 ** 2, workers=16, retries=4, timeout=60):
    expected = identity(info)
    safe_directory(dest.parent)
    if dest.exists() or dest.is_symlink():
        if verified(dest, info):
            return
        raise ValueError('Existing file fails hash check; preserved: ' + dest.name)
    partial = dest.with_name(dest.name + '.partial')
    journal = dest.with_name(dest.name + '.chunks.json')
    for path in (partial, journal):
        if path.exists() or path.is_symlink():
            regular(path)
    fd = os.open(partial, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        size = expected['size']
        done = set()
        if journal.exists():
            state = json.loads(journal.read_text())
            if state.get('identity') != expected or state.get('chunk_size') != chunk_size or os.fstat(fd).st_size != size:
                raise ValueError('Incompatible resume state; preserved')
            done = set(state['done'])
            if any(type(i) is not int or i < 0 or i >= math.ceil(size / chunk_size) for i in done):
                raise ValueError('Invalid resume chunk index')
        else:
            os.ftruncate(fd, size)
        lock = threading.Lock()

        def block(index):
            if index in done:
                return
            start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
            ranged = size > chunk_size
            data = fetch(urls, start if ranged else None, end, size, retries, timeout)
            offset = 0
            while offset < len(data):
                count = os.pwrite(fd, data[offset:], start + offset)
                if count <= 0:
                    raise OSError('Short cache write')
                offset += count
            with lock:
                os.fsync(fd)
                updated = done | {index}
                atomic_json(journal, {'identity': expected, 'chunk_size': chunk_size, 'done': sorted(updated)})
                done.add(index)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(block, range(math.ceil(size / chunk_size))))
        os.fsync(fd)
    finally:
        os.close(fd)
    if not verified(partial, info):
        raise ValueError('Downloaded hash mismatch; partial preserved: ' + dest.name)
    # Never replace an existing checkpoint even if created during this transfer.
    os.link(partial, dest)
    partial.unlink()
    journal.unlink(missing_ok=True)


def load_pinned(experiment):
    model_bytes = (experiment / 'models.json').read_bytes()
    if git_blob(model_bytes) != MODELS_BLOB:
        raise ValueError('models.json differs from pinned upstream commit')
    models = json.loads(model_bytes)
    output = {}
    for arm, model in models.items():
        directory = model['directory']
        if directory not in SOURCE_BLOBS:
            continue
        raw = (experiment / 'sources' / (directory + '_source.json')).read_bytes()
        if git_blob(raw) != SOURCE_BLOBS[directory]:
            raise ValueError('Source manifest differs from pinned upstream commit')
        source = json.loads(raw)
        if source['repo'] != model['repo'] or source['subfolder'] != model['subfolder']:
            raise ValueError('Source repository mismatch')
        files = selected_files(source)
        output[arm] = {'model': model, 'source': source, 'files': files,
                       'bytes': sum(identity(x)['size'] for x in files)}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'plan', 'download'])
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--arm', action='append')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--hosts', default='modelscope,hf-mirror,hf')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32:
        parser.error('workers must be between 1 and 32')
    arms = load_pinned(args.experiment_dir)
    chosen = args.arm or list(arms)
    if any(x not in arms for x in chosen):
        parser.error('Unknown or existing-local arm; choose one of ' + ', '.join(arms))
    if args.command in ('list', 'plan'):
        print(json.dumps({x: {'repo': arms[x]['source']['repo'], 'revision': arms[x]['source']['revision'],
                             'directory': arms[x]['model']['directory'], 'bytes': arms[x]['bytes'],
                             'files': len(arms[x]['files'])} for x in chosen}, indent=2))
        return
    if args.cache_dir is None:
        parser.error('--cache-dir is required for download')
    cache = safe_directory(args.cache_dir)
    for arm in chosen:
        entry = arms[arm]
        target = safe_directory(cache / entry['model']['directory'])
        lockpath = cache / (entry['model']['directory'] + '.download.lock')
        if lockpath.exists() or lockpath.is_symlink():
            regular(lockpath)
        with os.fdopen(os.open(lockpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for info in entry['files']:
                filename = PurePosixPath(info['rfilename']).name
                status = {'arm': arm, 'file': filename, 'revision': entry['source']['revision'], 'status': 'downloading'}
                atomic_json(target / 'download-status.json', status)
                download_file(target / filename, info, file_urls(entry['source'], entry['model'], info, args.hosts.split(',')), workers=args.workers)
                print(json.dumps({**status, 'status': 'verified', 'bytes': identity(info)['size']}), flush=True)
            atomic_json(target / 'DOWNLOAD_COMPLETE.json', {'arm': arm, 'upstream_commit': COMMIT,
                        'source': entry['source'], 'verified_files': entry['files'], 'bytes': entry['bytes']})
            atomic_json(target / 'download-status.json', {'arm': arm, 'status': 'complete', 'bytes': entry['bytes']})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # No transport exception messages, credentials or signed URLs in logs.
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'reason': str(error) if isinstance(error, ValueError) else 'See resumable cache state'}), flush=True)
        raise SystemExit(1)
