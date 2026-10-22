"""Fetch one trusted model file, sharing the main transport's per-file lock."""
import argparse
import json
from pathlib import Path
import time

from weighted_lora_model_download import MODEL, REVISION, locked_download, trusted_files
from gemmacodeagent_mm_download import atomic_json, file_urls, identity, safe_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--filename', required=True)
    args = parser.parse_args()
    root = safe_directory(args.root)
    tree = json.loads((root / 'download-tools/models-tree.json').read_text())
    info = next(item for item in trusted_files(tree)
                if item['rfilename'] == args.filename)
    print('start single-file transport: ' + args.filename, flush=True)
    locked_download(root / 'model' / args.filename, info,
                    file_urls({'repo': MODEL, 'revision': REVISION}, {}, info,
                              ['modelscope']), workers=8)
    atomic_json(root / 'model-second-verified.json',
                {'file': args.filename, **identity(info),
                 'verified_unix': time.time()})
    print('verified single-file transport: ' + args.filename, flush=True)


if __name__ == '__main__':
    main()
