"""Install the supplied core wheels and compatible CUDA dependencies in a venv."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
core_packages = [
    'vllm', 'swegemma', 'adk_submission', 'adk_eval_core', 'transformers',
    'google_adk', 'google_genai', 'compressed_tensors', 'flashinfer_python',
    'flashinfer_cubin', 'tokenizers',
]
local_wheels = [str(next((ROOT / 'wheelhouse').glob(name + '-*.whl')))
                for name in core_packages]
local_wheels += [str(next((ROOT / 'downloads').glob(name + '-*.whl')))
                 for name in ['torch', 'triton']]
cuda_wheels = [
    'https://download.pytorch.org/whl/cu128/torchaudio-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl',
    'https://download.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl',
]
command = [str(ROOT / 'runtime/bin/python'), '-m', 'pip', 'install',
           '--progress-bar', 'off', '--find-links', str(ROOT / 'downloads'),
           *local_wheels, *cuda_wheels,
           'fastapi==0.136.0',
           'nbconvert', 'nbclient', 'ipykernel', 'accelerate', 'litellm']
with (ROOT / 'install.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(ROOT / 'install-status.txt').write_text(str(result.returncode))
print('INSTALL_EXIT', result.returncode, flush=True)
raise SystemExit(result.returncode)
