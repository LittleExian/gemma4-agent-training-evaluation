"""Independent CPU-only uploader; no training control and no credential logging."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import time

REPO = 'eexian/gemma4-31b-w4-weighted-reasoning-lora'
PREFIX = 'runs/remediation-20261022'
FILES = ('adapter_config.json', 'adapter_model.safetensors')
RECIPE_KEYS = {'epochs', 'world_size', 'microbatch', 'gradient_accumulation',
               'learning_rate', 'loss_normalization', 'lora_rank', 'lora_alpha',
               'lora_dropout', 'seed', 'total_steps', 'save_every'}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def singleton(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate_recipe(recipe):
    if set(recipe) - {'code_commit', 'data_sha256', 'source_sha256', 'training_recipe'}:
        raise ValueError('Unknown recipe field')
    if not re.fullmatch('[0-9a-f]{40}', recipe.get('code_commit', '')):
        raise ValueError('Missing pinned code commit')
    if not re.fullmatch('[0-9a-f]{64}', recipe.get('data_sha256', '')):
        raise ValueError('Missing data SHA256')
    sources = recipe.get('source_sha256', {})
    if not isinstance(sources, dict) or any(not re.fullmatch('[0-9a-f]{64}', v) for v in sources.values()):
        raise ValueError('Invalid source fingerprints')
    if any(not re.fullmatch(r'[A-Za-z0-9_. /-]+', k) or k.startswith('/') or '..' in k for k in sources):
        raise ValueError('Source names must be safe relative paths')
    settings = recipe.get('training_recipe', {})
    if not isinstance(settings, dict) or set(settings) - RECIPE_KEYS:
        raise ValueError('Unknown training field')
    for key, value in settings.items():
        if key == 'loss_normalization':
            if value not in ('task_mean', 'token_mean'): raise ValueError('Unknown loss')
        elif isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError('Training settings must be numeric')
    json.dumps(recipe, allow_nan=False)
    return recipe


def auth_ready(home):
    token = home/'token'
    if not token.exists() and not token.is_symlink(): return False
    info = token.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError('HF token must be an owned regular file with mode 0600')
    return info.st_size > 0


def discover(output):
    complete = []
    for checkpoint in output.glob('checkpoint-*'):
        match = re.fullmatch(r'checkpoint-([1-9][0-9]*)', checkpoint.name)
        if not match or checkpoint.is_symlink(): continue
        try:
            step = int(match[1])
            if json.loads((checkpoint/'COMPLETE.json').read_text()) != {'step': step, 'world_size': 2}: continue
            if json.loads((checkpoint/'adapter_audit.json').read_text()).get('passed') is not True: continue
            if (checkpoint/'adapter').is_symlink(): continue
            if any((checkpoint/'adapter'/name).is_symlink() or not (checkpoint/'adapter'/name).is_file()
                   or not (checkpoint/'adapter'/name).stat().st_size for name in FILES): continue
            complete.append(checkpoint)
        except (OSError, ValueError, TypeError):
            continue  # A partially written checkpoint is not eligible yet.
    return sorted(complete, key=lambda p: int(p.name.split('-')[1]))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def payload(checkpoint, recipe):
    paths = {name: checkpoint/'adapter'/name for name in FILES}
    hashes = {name: digest(path) for name, path in paths.items()}
    info = {'step': int(checkpoint.name.split('-')[1]), 'world_size': 2,
            'recipe': recipe, 'files': {name: {'sha256': hashes[name], 'bytes': path.stat().st_size}
                                     for name, path in paths.items()}}
    encoded = json.dumps(info, sort_keys=True, indent=2, allow_nan=False).encode()
    return {**paths, 'checkpoint-info.json': encoded}, {**hashes, 'checkpoint-info.json': hashlib.sha256(encoded).hexdigest()}


def check_account(api):
    if api.whoami().get('name') != 'eexian': raise ValueError('Authenticated account must be eexian')
    api.repo_info(REPO, repo_type='model')  # Existing repository only, preserving its visibility.


def verify_remote(api, prefix, commit, hashes, downloader):
    rows = {row.path: row for row in api.list_repo_tree(REPO, path_in_repo=prefix.rstrip('/'),
            revision=commit, recursive=True, expand=True, repo_type='model')}
    for name, expected in hashes.items():
        row = rows.get(prefix+name)
        if row is None: raise ValueError('Remote checkpoint file missing')
        lfs = getattr(row, 'lfs', None)
        if name.endswith('.safetensors'):
            actual = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        else:
            actual = hashlib.sha256(downloader(name)).hexdigest()
        if actual != expected: raise ValueError('Remote checkpoint SHA256 mismatch')


def upload(api, checkpoint, recipe, operation, download_file, verifier=verify_remote):
    files, hashes = payload(checkpoint, recipe)
    prefix = f'{PREFIX}/{checkpoint.name}/'
    operations = [operation(path_in_repo=prefix+name,
                  path_or_fileobj=io.BytesIO(value) if isinstance(value, bytes) else str(value))
                  for name, value in files.items()]
    commit = api.create_commit(repo_id=REPO, repo_type='model', operations=operations,
                commit_message=f'Upload remediation {checkpoint.name}').oid
    def downloader(name):
        path = download_file(repo_id=REPO, filename=prefix+name, revision=commit, repo_type='model')
        return Path(path).read_bytes()
    verifier(api, prefix, commit, hashes, downloader)
    # Recheck local immutable checkpoint data after upload, detecting accidental overwrites.
    if payload(checkpoint, recipe)[1] != hashes: raise ValueError('Local checkpoint changed during upload')
    return {'status': 'uploaded', 'commit': commit, 'sha256': hashes, 'verified_unix': time.time()}


def training_finished(output):
    try: return json.loads((output/'progress.json').read_text()).get('state') == 'completed'
    except (OSError, ValueError, TypeError): return False


def expected_names(recipe):
    settings = recipe['training_recipe']
    total, interval = settings.get('total_steps'), settings.get('save_every')
    if not isinstance(total, int) or not isinstance(interval, int) or min(total, interval) < 1:
        raise ValueError('Positive total_steps and save_every required')
    return {f'checkpoint-{step}' for step in (*range(interval, total+1, interval), total)}


def run(args):
    root = args.root.resolve()
    output, reports = root/'work/full-epoch1', root/'reports'
    recipe = validate_recipe(json.loads(args.recipe_json.read_text()))
    expected_names(recipe)
    home = root/'hf_upload_auth'
    # Force SDK token resolution to the designated private file, never inherited credentials.
    os.environ['HF_HOME'] = str(home)
    os.environ['HF_TOKEN_PATH'] = str(home/'token')
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_HUB_OFFLINE'):
        os.environ.pop(key, None)
    state_path = reports/'checkpoint-upload-state.json'
    with singleton(reports/'checkpoint-upload.lock'):
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        deadline = time.monotonic() + args.max_hours*3600
        while time.monotonic() < deadline:
            checkpoints = discover(output)
            phase, error_type = 'waiting_auth', None
            try:
                if auth_ready(home):
                    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
                    api = HfApi()
                    check_account(api)
                    phase = 'watching'
                    for checkpoint in checkpoints:
                        previous = state.get(checkpoint.name, {})
                        if previous.get('status') == 'uploaded':
                            # Immutable commit and local hashes recorded only after remote verification.
                            if previous.get('sha256') == payload(checkpoint, recipe)[1]: continue
                            raise ValueError('Previously uploaded checkpoint changed')
                        state = {**state, checkpoint.name: {'status': 'uploading'}}
                        atomic_json(state_path, state)
                        result = upload(api, checkpoint, recipe, CommitOperationAdd, hf_hub_download)
                        state = {**state, checkpoint.name: result}
                        atomic_json(state_path, state)
                    expected = expected_names(recipe)
                    if training_finished(output) and expected and expected <= {p.name for p in checkpoints} and all(
                            state.get(name, {}).get('status') == 'uploaded' for name in expected):
                        phase = 'completed'
            except Exception as error:
                # Only exception class is persisted: HTTP messages may contain credentials/URLs.
                phase, error_type = 'retry_pending', type(error).__name__
            progress = {'phase': phase, 'repo': REPO, 'prefix': PREFIX, 'updated_unix': time.time(),
                        'eligible': [p.name for p in checkpoints],
                        'uploaded': [name for name, row in state.items() if row.get('status') == 'uploaded'],
                        'error_type': error_type}
            atomic_json(reports/'hf-upload-progress.json', progress)
            if phase == 'completed': return
            time.sleep(args.poll_seconds)
        atomic_json(reports/'hf-upload-progress.json', {**progress, 'phase': 'deadline_reached'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recipe-json', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-hours', type=float, default=72)
    args = parser.parse_args()
    if args.poll_seconds < 1 or not 0 < args.max_hours <= 72: parser.error('Invalid bounded polling interval')
    run(args)


if __name__ == '__main__': main()
