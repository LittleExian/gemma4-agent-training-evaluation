#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT=${TRAIN_ROOT:-/workspace/weighted_reasoning_lora}
ENV=${DEPLOY_ENV:-$ROOT/envs/deployment}
mkdir -p "$ROOT/deployment_wheels"
python3 - "$ROOT/deployment_wheels" <<'PY'
import hashlib,pathlib,sys,urllib.request,urllib.parse
root=pathlib.Path(sys.argv[1])
files={
 'transformers-5.13.1-py3-none-any.whl':'53f0ea8aa397e29244c2377ba981bcaf0c87adcf44fbdd447ef6306522afcacd',
 'vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl':'73c69bafca24fbdad44690c44d6c6446514351e6f3cc9a6057cba05e66051310',
 'compressed_tensors-0.15.0.1-py3-none-any.whl':'e1b1f322e82e475715e242bad46925a304ea8e5c98b5055a15b8eb22fb6bfea9'}
for name,digest in files.items():
 p=root/name
 if not p.exists():
  url='https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse/'+urllib.parse.quote(name)+'?datasetVersionNumber=31'
  urllib.request.urlretrieve(url,p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=digest:raise RuntimeError('Wheel SHA256 mismatch: '+name)
PY
python3 -m venv "$ENV"
"$ENV/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu128 \
  --find-links "$ROOT/deployment_wheels" \
  "$ROOT/deployment_wheels/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl" \
  "$ROOT/deployment_wheels/transformers-5.13.1-py3-none-any.whl" \
  "$ROOT/deployment_wheels/compressed_tensors-0.15.0.1-py3-none-any.whl" \
  'torch==2.10.0' 'huggingface_hub==1.16.1' \
  'flashinfer-python==0.6.6' 'flashinfer-cubin==0.6.6' 'xgrammar==0.2.5.post1' 'numpy==2.2.6' 'safetensors==0.8.0'
# Explicit wheels with identical version strings must replace any PyPI install.
"$ENV/bin/pip" install --force-reinstall --no-deps \
  "$ROOT/deployment_wheels/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl" \
  "$ROOT/deployment_wheels/transformers-5.13.1-py3-none-any.whl" \
  "$ROOT/deployment_wheels/compressed_tensors-0.15.0.1-py3-none-any.whl"
"$ENV/bin/pip" freeze > "$ROOT/deployment_environment.lock"
printf 'Deployment environment ready: %s\n' "$ENV"
