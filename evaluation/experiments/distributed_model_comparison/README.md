# Seven models / all 129 tasks / 3–4 PRO 6000 servers

Default **7 × 129 = 903 evaluations**. Each server uses its own 96 GiB PRO 6000, runs all models sequentially on a disjoint task shard, with up to 16 rolling concurrent tasks. No cross-server inference.

## Models and validation status (2026-10-17)

| Arm | Model | Fixed16 resolved |
|---|---|---:|
| qwen35_base | Qwen3.5-9B runtime FP8 | 7/16 |
| qwen35_swegym | SWE-Gym LoRA on the same FP8 base | 4/16 |
| qwen38_base | Qwen3.8-27B-FP8 | 5/16 |
| qwen38_coder390 | nerkyor Coder390 published FP8 | 7/16 |
| gemma4_kaggle | original Kaggle Gemma 4 W4A16 | 9/16 |
| gemma4_teich_agent | TeichAI/Gemma-4-31B-Fable-5-Agent-Distill | not run |
| gemma4_capybara | TendieLabs/Capybara-31B | not run |

EfficientThink was cancelled and is excluded. Interrupted attempts are not scored. Both new Gemma models require BF16 download and runtime FP8; compatibility is NOT yet validated. The package has static/partition checks, but a full multi-server run has NOT been performed. See reports/RESULTS.md and reports/current_results.json.

## 1. Prepare EACH server

Keep everything on persistent storage. Recommend **250 GiB or larger** disk; 150 GiB may need manual space management. Each new Gemma downloads about 62.55 GB before FP8 conversion. Completed weight caches are removed by default; logs/results remain. Prefetch is disabled by default. Use MM_KEEP_WEIGHTS=1 only with space for all weights.

```bash
mkdir -p /root/autodl-tmp/codeagent
cd /root/autodl-tmp/codeagent
git clone https://github.com/TheWayLost/gemmacodeagent.git
cd gemmacodeagent
# IMPORTANT: setup.sh installs relative to the repository root.
export CODEAGENT_ROOT="$PWD"
export PACKAGE="$PWD/experiments/distributed_model_comparison"
# Export your own KAGGLE_API_TOKEN and HF_TOKEN if required for gated assets.
bash scripts/setup.sh
envs/eval/bin/python scripts/download_model.py
mkdir -p "$CODEAGENT_ROOT/tmp/mm"
envs/eval/bin/python "$PACKAGE/prepare_all.py"
envs/eval/bin/python "$PACKAGE/check_partition.py"
```

A generic PyPI evaluator is not equivalent. prepare_all.py prepares all 129-task artifacts. Model downloads start automatically and resume range chunks. Bundled source manifests pin available revisions. Large files try ModelScope then HF Mirror. Teich has a ModelScope repository; no usable Capybara mirror was found. Normal file integrity checks remain; no separate cross-site verification is required before downloading.

## 2. Start all 129 tasks

Use ONE partition count and distinct worker indices. The launcher runs a model health gate before each model's tasks; no separate 16-task pilot is necessary.

```bash
# On 3 machines: use index 0, 1, 2 respectively
bash "$PACKAGE/launch_worker.sh" 3 0
# OR on 4 machines: use index 0, 1, 2, 3 respectively
bash "$PACKAGE/launch_worker.sh" 4 0
```

3 machines: 43 tasks × 7 = **301 evaluations each**.
4 machines: 33/32/32/32 tasks = **231/224/224/224 evaluations**.
Do not mix partitions or run multiple drivers per GPU. Use the same Git commit and source manifests everywhere. Launcher prints PID and log path. Default 16 slots; set MM_CONCURRENCY=8 to lower it. The health gate may reduce concurrency. Record contention changes under a fixed wall-clock limit.

The new Gemma arms must pass loading/tool/budget checks before scoring. On errors, preserve server and health logs and diagnose; do not silently substitute quantization or selectively retry results. Resume with the same command/config: completed result files are reused. Changed protocols require a separate result root.

Optional pilot: `python "$PACKAGE/run.py" --scope pilot16 --concurrency 16`.

## 3. Protocol and merge

Budget-Fit v5: **900 seconds/task, thinking budget 6000**, 56 tool calls, 160 turns, output 8192, temperature 0.2, top-p 0.95, context 32768, compaction 14336. Rolling concurrency, no speculative decoding. Qwen/new Gemma use vLLM generation defaults; original Gemma retains checkpoint defaults including top-k. FP8 protected layers vary; original Gemma is W4A16. This is a deployment comparison, not an isolated causal estimate of training quality.

Gold/empty calibration runs for each shard. All 129 tasks are attempted, including historically invalid environments. Report raw 129-task scores AND valid-environment scores, infrastructure/API errors and timeouts separately. calibration_audit.json records fresh checks; environment_previously_valid is historical and is not a substitute. Raw requests/responses, reasoning, patches and tests remain on each worker.

State: `$CODEAGENT_ROOT/runs/distributed_model_comparison/worker_INDEX_of_COUNT/all_state.json`.
Server/health/download logs: `$CODEAGENT_ROOT/research/distributed_model_comparison/`.
Collect state files with distinct filenames and also retain calibration audits and protocol files:

```bash
python "$PACKAGE/merge_results.py" results/server*.json --output results/merged.json
```

Merger rejects incomplete/mixed partitions, duplicate workers and missing task-model pairs. Never publish credentials/private teacher logs. Small pilot differences do not establish general training-method effects.
