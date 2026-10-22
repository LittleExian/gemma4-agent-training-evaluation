# Historical dependency audit

The exported evaluation environment was audited with `pip-audit 2.10.1` on 2026-10-22 using pinned versions, `--no-deps --disable-pip`, and the default PyPI advisory service. This checks the recorded evaluation lockfile without installing it or altering the server. Test-profile and training environments were not independently audited by this report.

The tool returned exit status 1: **155 advisory entries across 11 packages** (111 distinct package/advisory-ID pairs; aliases and duplicate records are not independent proven exploit paths). This is not a severity assessment or a demonstration that these paths were exercised in the benchmark.

| Package | Recorded version | Advisory entries |
|---|---|---:|
| flask | 2.2.5 | 2 |
| werkzeug | 2.3.8 | 13 |
| aiohttp | 3.13.4 | 36 |
| certifi | 2026.7.22 | 1 |
| diskcache | 5.6.3 | 2 |
| litellm | 1.83.14 | 6 |
| pillow | 12.3.0 | 24 |
| pytest | 8.3.4 | 2 |
| setuptools | 80.10.2 | 2 |
| starlette | 0.52.1 | 10 |
| vllm | 0.19.1 | 57 |

6 distributions could not be matched/audited by this service:

- `adk-eval-core`: Dependency not found on PyPI and could not be audited: adk-eval-core (0.1.0)
- `adk-submission`: Dependency not found on PyPI and could not be audited: adk-submission (0.2.13)
- `swegemma`: Dependency not found on PyPI and could not be audited: swegemma (0.2.11)
- `torch`: Dependency not found on PyPI and could not be audited: torch (2.10.0+cu128)
- `torchaudio`: Dependency not found on PyPI and could not be audited: torchaudio (2.10.0+cu128)
- `torchvision`: Dependency not found on PyPI and could not be audited: torchvision (0.25.0+cu128)

The full machine-readable report is [dependency-audit.json](dependency-audit.json). Historical pins remain unchanged to preserve run provenance. They are **not a vulnerability-free deployment recommendation**. No public inference endpoint or new cloud workload is created by publishing this repository.

A future deployment should use a separate environment, evaluate the listed fixes against the required GPU/model stack, rerun the environment/serialization/LoRA/sandbox controls, and recalculate benchmark results when dependencies change. Do not silently upgrade the running training environment or label an altered environment as this historical run.

The additional historical runtime freezes under `history-sources/` were not independently audited by this report.
