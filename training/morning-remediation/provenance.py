"""Record the exact code/data used, and reject incompatible resume settings."""
import hashlib,json
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def build_manifest(args):
    root=Path(__file__).parent
    return {'schema_version':1,'data_sha256':sha256(args.data),
            'objective':args.objective,'model':args.model,'revision':args.revision,
            'world_size':int(__import__('os').environ.get('WORLD_SIZE','1')),
            'accum':args.accum,'lr':args.lr,'seed':args.seed,'smoke':args.smoke,'attention':args.attention,
            'source_sha256':{p.name:sha256(p) for p in sorted(root.glob('*.py'))}}

def check_resume_manifest(resume, manifest):
    old_path=Path(resume).parent/'training_manifest.json'
    if not old_path.exists():
        raise ValueError('Resume lacks training_manifest.json. Start a new controlled run; do not guess its objective/data.')
    old=json.loads(old_path.read_text())
    for key in ['data_sha256','objective','model','revision','world_size','accum','lr','seed','smoke','attention','source_sha256']:
        if old.get(key)!=manifest.get(key):raise ValueError(f'Resume mismatch: {key}')

def save_manifest(output,manifest):
    output=Path(output);path=output/'training_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Output directory contains a different training manifest')
    source=output/'source';source.mkdir(parents=True,exist_ok=True)
    for name in manifest['source_sha256']:
        (source/name).write_bytes((Path(__file__).parent/name).read_bytes())
    path.write_text(json.dumps(manifest,indent=2))
