#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT=${TRAIN_ROOT:-/workspace/weighted_reasoning_lora_v2}
: "${DATASET_REPO:?Set DATASET_REPO to your authorized HF dataset}"
: "${DATASET_REVISION:?Set DATASET_REVISION to a full dataset commit SHA}"
mkdir -p "$ROOT"
export HF_HOME=${HF_HOME:-$ROOT/hf_cache}
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
python - "$ROOT" <<'PY'
import sys,json,pathlib,os,re
from huggingface_hub import snapshot_download
root=pathlib.Path(sys.argv[1]);lock=json.loads(pathlib.Path('revisions.json').read_text())
assert re.fullmatch(r'[0-9a-f]{40}',os.environ['DATASET_REVISION']), 'Use a full immutable dataset commit SHA'
lock['dataset']={'repo_id':os.environ['DATASET_REPO'],'revision':os.environ['DATASET_REVISION']}
(root/'run_revisions.json').write_text(json.dumps(lock,indent=2))
for kind in ['model','dataset']:
 spec=lock[kind];snapshot_download(spec['repo_id'],revision=spec['revision'],repo_type=None if kind=='model' else 'dataset',local_dir=None if kind=='model' else root/'dataset',max_workers=8)
PY
REV=$(python -c 'import json;print(json.load(open("revisions.json"))["model"]["revision"])')
python -c 'from base_loading import verify_source_hash; print(verify_source_hash())'
python curate.py --data "$ROOT/dataset" --output "$ROOT/curated"
python prepare.py --data "$ROOT/curated" --output "$ROOT/tokenized" --revision "$REV"
python test_curation.py
python test_objectives.py
python test_quality_gate.py
python test_provenance.py
python test_weighting.py
python test_template.py
python test_serialization.py
python test_streamed_ce.py
python verify_prefix_alignment.py --data "$ROOT/curated" --tokens "$ROOT/tokenized" --tokenizer "$ROOT/tokenized/tokenizer" --output "$ROOT/prefix_alignment.json"
python test_w4_contract.py
printf 'Ready. Run the single-card smoke test in README.md before a short checkpoint-gated experiment.\n'
