> **2026-10-22 required update:** read [PREFIX_ALIGNMENT.md](PREFIX_ALIGNMENT.md). Rebuild token caches and start a new run; old caches/resume are rejected. CPU validation is recorded separately from earlier GPU smoke results.

> **2026-10-21 regression audit:** the epoch-1 adapter solved 5/20 versus the base 11/20 on our paired development evaluation. Read [REMEDIATION.md](REMEDIATION.md) before retraining: it includes audited data counts, conservative curation, task-balanced objectives, short-run checkpoint gates, and 2-4 GPU commands. These changes have not yet demonstrated better repair accuracy.

# Version B: weighted-reasoning LoRA for repository-repair agents

This package trains a text-only LoRA adapter on the frozen deployment Gemma 4 31B W4A16 checkpoint, decoded to BF16 without recovering its pre-quantization values. It supports one 96GB GPU for validation and **2, 3, or 4 RTX PRO 6000 96GB GPUs** with DistributedDataParallel (DDP). It does not train the vision tower.

**Run the single-card smoke test first.** Training correctness does not establish improved bug-fixing accuracy. Real hardware validation results and limitations are in `VALIDATION.md`.

## Objective and data

For each model decision, the prompt is the recorded model-visible history (including runtime compaction and tool responses), and the only supervised completion is the current assistant response. The pinned native Gemma template serializes tool calls and thinking channels. Tool arguments are parsed into objects before rendering.

Token weights are **0 for prompt/history/tool outputs, 0.25 for thinking text, and 1 for the current assistant's action/content and protocol delimiters**. The causal shift is applied explicitly. The legacy `--objective token_mean` objective is `sum(weight * cross_entropy) / sum(weight)` over all ranks and all microbatches in the optimizer step. DDP's gradient averaging is compensated by the world-size factor; averaging separately normalized per-rank losses would be incorrect for unequal completion lengths. The new `--objective task_mean` instead averages within each decision and task; see REMEDIATION.md for the formula and controlled comparisons. Standalone compaction targets are excluded by default; summaries already in repair history remain visible.

The dataset is supplied through `DATASET_REPO` and `DATASET_REVISION` environment variables; pin the full dataset commit before processing. The raw release contains 332 attempts / 305 tasks. Only 200 test-verified successful trajectories feed the SFT views: **180 training tasks (2,951 decisions)** and **20 validation tasks (216 decisions)**. The current default curation retains **2,841 training decisions and 214 validation decisions**, excluding 109/2 standalone compaction targets and one invalid training edit. The original release remains intact. Test success is not exhaustive semantic review. Failed attempts and truncated responses remain available in the raw release but are not positive SFT targets. Never split individual decisions from the same task across train/validation.

`revisions.json` pins the model commit. Supply the full dataset commit SHA through `DATASET_REVISION`; bootstrap records both revisions in `run_revisions.json`. `prepare.py` refuses silently truncated examples: overlength or malformed examples are recorded in `preparation.json`. With the pinned release, all 3,167 eligible decisions fit the 24,576-token preparation ceiling; the longest training example is 19,825 tokens. This ceiling is **not a claim that arbitrary 24K examples fit in memory**.

## Environment

Validated image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Python 3.12, PyTorch `2.8.0+cu128`, 96GB RTX PRO 6000 Blackwell Server Edition. Use a compatible NVIDIA driver and the CUDA 12.8 build of PyTorch; do not install an older CUDA build on Blackwell.

Store downloads, prepared data, logs, and checkpoints on persistent storage. Allow roughly 25GB for the original W4 checkpoint cache plus space for data and checkpoints; decoded BF16 weights occupy GPU memory, not a second saved base checkpoint. Each retained checkpoint includes approximately 0.5GB of adapter weights plus optimizer state. A frozen full model is replicated on every GPU: this package is BF16 LoRA, not NF4 QLoRA and not tensor parallelism.

```bash
cd training/weighted_reasoning_lora
export DATASET_REPO="YOUR_NAMESPACE/YOUR_DATASET"
export DATASET_REVISION="FULL_DATASET_COMMIT_SHA"
export TRAIN_ROOT=/workspace/weighted_reasoning_lora_v2
export HF_HOME=$TRAIN_ROOT/hf_cache
# If model access needs authentication, set HF_TOKEN through your secret manager.
# Do not put a token in a command committed to Git.
bash bootstrap.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
REV=$(python -c 'import json; print(json.load(open("revisions.json"))["model"]["revision"])')
```

`bootstrap.sh` downloads the pinned model and data, audits and curates decisions, excludes standalone compaction targets, prepares native-template weighted samples, and runs numerical checks. Use a fresh TRAIN_ROOT; existing curation outputs are not overwritten. It pins PyTorch 2.8.0 (CUDA 12.8 index), Transformers 5.20.2 and compressed-tensors 0.15.0. Do not upgrade compressed-tensors independently: newer releases require a different PyTorch stack. It does not create cloud resources or delete data.

## First: single-card training gate

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data "$TRAIN_ROOT/tokenized/train.jsonl" \
  --output "$TRAIN_ROOT/smoke" --revision "$REV" \
  --objective task_mean --smoke --save-every 1 2>&1 | tee "$TRAIN_ROOT/smoke.log"
```

The smoke test selects the longest real training sequence, performs one optimizer step, checks finite positive gradient norms and nonzero adapter updates, saves checkpoints, reloads the final adapter, and requires exactly matching evaluation logits. Success is an existing `SMOKE_PASS.json` with `passed: true`, not simply a running process or a falling loss on different samples. A failure must be fixed before the full job.

Memory measures: microbatch size 1, non-reentrant gradient checkpointing, nested 2,048-token feed-forward checkpoint chunks, BF16 autocast, and selective completion logits. Prompt tokens still participate in attention but do not receive direct loss. Cross-entropy FP32 conversion is chunked. The first unchunked implementation exhausted memory on the longest sample; do not remove the memory measures casually.

## 2-4 GPUs on one host

Run the distributed numerical gate first. This compares the distributed weighted gradient against a centralized reference on CPUs; it is not a GPU capacity test.

```bash
for N in 2 3 4; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$N test_weighting.py
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$N test_task_objective_distributed.py
done
```

Then run **25 steps and evaluate the early checkpoints against the base** before continuing. `--steps 0` remains available for explicit full-epoch reproduction; it is not the recommended first run. The last distributed batch may repeat a few examples to keep ranks synchronized.

```bash
N=3  # set to 2, 3, or 4
ACCUM=4
set -o pipefail
torchrun --standalone --nproc_per_node=$N train.py \
  --data "$TRAIN_ROOT/tokenized/train.jsonl" \
  --output "$TRAIN_ROOT/lora-b-${N}gpu" --revision "$REV" \
  --objective task_mean --steps 25 --accum "$ACCUM" --lr 2e-5 --save-every 5 \
  2>&1 | tee "$TRAIN_ROOT/lora-b-${N}gpu.log"
```

| GPUs | Microbatch/GPU | Accumulation | Decisions/optimizer step |
|---|---:|---:|---:|
| 2 | 1 | 4 | 8 |
| 3 | 1 | 4 | 12 |
| 4 | 1 | 4 | 16 |

These settings change the global batch. For a controlled comparison with global batch 24, use accumulation 12 / 8 / 6 for 2 / 3 / 4 GPUs. All GPUs need 96GB; a 48GB RTX 6000 Ada is a different card and cannot use this BF16 recipe as written. Actual multi-GPU throughput depends on PCIe/NCCL and has not been measured by the single-card smoke test.

## One GPU on each of 2-4 hosts

Run `bootstrap.sh` on every host with the same pinned revisions. A shared filesystem is required for the output directory in the current checkpoint implementation, because each rank writes its own RNG state. The dataset/cache may be local copies. Set a private, mutually reachable rendezvous address; do not expose the rendezvous service publicly.

On every host, set `NNODES=2`, `3`, or `4`, and a distinct `NODE_RANK` from zero through `NNODES-1`:

```bash
export NNODES=3
export NODE_RANK=0   # use 1 and 2 on the other hosts
export MASTER_ADDR=10.0.0.10  # replace with rank-0 private address
export MASTER_PORT=29500
export SHARED_OUTPUT=/shared/lora-b-run

torchrun --nnodes="$NNODES" --nproc_per_node=1 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" train.py \
  --data "$TRAIN_ROOT/tokenized/train.jsonl" \
  --output "$SHARED_OUTPUT" --revision "$REV" --accum 4 --objective task_mean --steps 25 --save-every 5
```

The same shared output path must be visible on every host. Without shared output storage, use a single multi-GPU host for this version. Multi-host operation and NCCL networking are implemented through standard torchrun/DDP but were not physically tested in this validation. DDP synchronizes only trainable adapter gradients, yet slow networking can still limit throughput.

## Checkpoints and restart

Each checkpoint contains `adapter/`, `training_state.pt`, and `rng.rankN.pt` for each rank. A `COMPLETE.json` marker is written only after all ranks finish writing. The output also contains `config.json` and per-rank `metrics.rankN.jsonl`. Keep the model/dataset revisions, GPU count, seed, accumulation, and data ordering unchanged when resuming.

```bash
torchrun --standalone --nproc_per_node=3 train.py \
  --data "$TRAIN_ROOT/tokenized/train.jsonl" \
  --output "$TRAIN_ROOT/lora-b-3gpu" --revision "$REV" --accum 4 \
  --objective task_mean --steps 50 --resume "$TRAIN_ROOT/lora-b-3gpu/checkpoint-25"
```

Resume requires matching training_manifest.json (data, objective, code and optimizer settings); legacy runs without it must use their original code. This reloads optimizer and per-rank RNG state and reconstructs the deterministic sampler position. It rejects a different world size. Do not resume from an incomplete checkpoint or an adapter-only download. Reproducibility across different CUDA/library/hardware versions is not guaranteed. The smoke checkpoints from the initial memory debugging run are evidence artifacts, not recommended starting points for a full training run.

## W&B monitoring: configure your own account

W&B is opt-in. Each trainer logs in on their own machine and chooses their own account/team and project. **No API key, entity, project, or online run is embedded in this repository.** `requirements.txt` pins the tested SDK (`wandb==0.31.0`).

First install dependencies using `bootstrap.sh` as above, or `python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt`. Then use the normal W&B login workflow:

```bash
wandb login  # enter your own API key at the prompt
# If this machine is already logged into another account: wandb login --relogin
export WANDB_ENTITY="your-user-or-team"  # optional: omit for your login's default entity
export WANDB_PROJECT="repo-agent-lora"
export WANDB_NAME="lora-b-2gpu-seed42"
export WANDB_MODE=online
```

For a team project, your W&B account needs permission to write to that team. The entity is the user/team slug, not an email address. In containers or CI, inject your own `WANDB_API_KEY` through the platform secret manager instead of interactive login. A shell alternative that does not put the value in command history is:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY; echo
export WANDB_API_KEY
```

Do not put keys in a training script, README, Git commit, or shared `.env` file. The training CLI deliberately has no `--wandb-api-key` argument.

Start training with the existing commands; the environment variables above enable monitoring automatically:

```bash
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
REV=$(python -c 'import json; print(json.load(open("revisions.json"))["model"]["revision"])')
torchrun --standalone --nproc_per_node=2 train.py \
  --data "$TRAIN_ROOT/tokenized/train.jsonl" \
  --output "$TRAIN_ROOT/lora-b-2gpu" --revision "$REV" \
  --objective task_mean --steps 25 --accum 4 --lr 2e-5 --save-every 5 \
  2>&1 | tee "$TRAIN_ROOT/lora-b-2gpu.log"
```

The terminal prints the W&B run URL. It is also saved in `<output>/wandb_run.json`. You can instead pass `--wandb-mode online --wandb-project PROJECT --wandb-entity ENTITY --wandb-name NAME`. CLI options override environment variables. If neither `WANDB_MODE` nor `WANDB_PROJECT` is configured, monitoring defaults to disabled. A project supplied only by CLI should also specify `--wandb-mode online`.

### What appears in the dashboard

- Global weighted training loss, pre-clipping gradient norm, learning rate, and checked adapter update magnitude.
- Optimizer step, total steps, progress percentage, approximate epoch fraction, elapsed training time, and ETA.
- Global sequence tokens/second and decisions/second, completion/reasoning/action token counts, and maximum sequence length per optimizer step.
- Every rank's peak allocated/reserved GPU memory and the slowest rank's compute duration.
- Last saved checkpoint step, checkpoint-save duration, and training exit status.

Use `train/optimizer_step` as the chart X axis. Metrics are logged once per optimizer update, not once per accumulation microbatch. Sequence throughput includes repeated prompt history and is **not** teacher-output throughput. ETA is estimated from completed steps in the current session and becomes more reliable after multiple steps; it excludes initial model loading and future evaluations. Epoch fraction can slightly exceed 1 due to distributed sampler padding. W&B's automatic system monitoring covers the rank-0 host; explicit per-rank peak-memory metrics cover all participating hosts.

Only rank 0 creates a W&B run. Other ranks contribute numerical statistics through a collective; they do not need separate logins. On multiple hosts, all ranks must run the same updated code. Logging configuration is coordinated from rank 0, including initialization failures. Existing `metrics.rankN.jsonl` files remain available. Monitoring does not alter the selected loss/optimizer recipe. Loss values from token_mean and task_mean are different objectives and should not be directly compared. No dataset text, thinking text, sample IDs, model weights, checkpoints, Git metadata, or console transcript is explicitly uploaded by this integration; scalar metrics, selected hyperparameters and SDK system telemetry are sent. Model artifact upload is disabled by omission.

These are training metrics. Validation loss and bug-fixing accuracy are **not** computed automatically.

### Resume, switch accounts, or work offline

For an online run, use the same `--output`, project/entity and the existing `--resume .../checkpoint-N` option. The saved W&B ID is reused with `resume="must"`; checkpoint folders also contain `wandb_run.json`. If you resume a checkpoint created before W&B was enabled, a new run starts at the restored optimizer step. `WANDB_RUN_ID` / `--wandb-id` can explicitly identify an existing online run when local metadata is unavailable.

If a collaborator uses another account/project, or you want a separate comparison run while resuming model weights, add **`--wandb-new-run`** and choose their own entity/project. Do not reuse someone else's `WANDB_RUN_ID`. Resuming an older checkpoint into the same run preserves previously logged history; use a new run when you want a clean branch of the experiment.

For offline tracking:

```bash
export WANDB_MODE=offline
export WANDB_PROJECT="repo-agent-lora"
# Run the same training command. Logs are kept under <output>/wandb/.
# Later, log in with the intended account and sync each offline directory:
wandb login
wandb sync --entity "your-user-or-team" --project "repo-agent-lora" \
  "$TRAIN_ROOT/lora-b-2gpu/wandb/offline-run-YYYYMMDD_HHMMSS-RUNID"
```

The SDK does not support resuming an offline run in place. Each offline training restart creates a separate monitoring segment with its own ID; syncing these segments does not automatically join their histories. To move from offline to an online training continuation, add `--wandb-new-run` and sync the old logs separately.

Set `WANDB_MODE=disabled` or pass `--wandb-mode disabled` to turn monitoring off. Do not set dummy API keys.

Run `python test_tracking.py` before a long job. It uses a fake backend for online resume/account-switch cases and a real offline SDK for persisted metric checks; it requires no GPU or online account. Cloud authentication and real multi-GPU logging must be checked by the trainer with their own account.

The production statistics-gathering helper can also be checked on CPUs:

```bash
for N in 2 3 4; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$N test_tracking_distributed.py
done
```

References: [W&B Python run API](https://docs.wandb.ai/ref/python/experiments/run/) and [W&B SDK](https://github.com/wandb/wandb).

## Original W4 deployment gate (after the one-step smoke)

Training and inference use separate Python environments: training uses Torch 2.8.0 + compressed-tensors 0.15.0; the pinned competition wheelhouse v31 inference stack uses patched vLLM 0.19.1, Torch 2.10.0 and compressed-tensors 0.15.0.1. Do not install vLLM into the training environment. The setup script downloads the three patched wheels and verifies their SHA256; generic PyPI vLLM/Transformers wheels with the same version numbers are not substitutes for this gate.

```bash
bash setup_deployment_env.sh
export HF_HOME=$TRAIN_ROOT/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"$TRAIN_ROOT/envs/deployment/bin/python" validate_deployment.py \
  --checkpoint "$TRAIN_ROOT/smoke/checkpoint-1" \
  --output "$TRAIN_ROOT/deployment_gate"
```

Wait until training has exited and freed GPU memory. This gate serves the **original packed W4 weights**, loads the exported adapter through vLLM, compares no adapter against a zero-B control, and checks that the trained adapter changes raw next-token logits on two fixed prompts. It performs no optimizer updates. `deployment_gate.json` must report `passed: true`. It is an adapter loading/effect gate, not a repair benchmark, and does not guarantee identical BF16/W4 kernel logits or competition scores. Keep the fixed source revision and wheel hashes; revalidate if the deployment environment changes.

## Configuration and evaluation

LoRA: rank 16, alpha 16, dropout 0.05; language-model q/k/v/o and gate/up/down projections; frozen base/embedding/output/vision weights. Optimizer: AdamW, learning rate 2e-5, weight decay 0.01, gradient clipping 1.0, constant learning rate, seed 42. This is a deliberately small, inspectable starting recipe, not an optimized hyperparameter search. `--steps` can cap a development run.

The base is **the pinned deployment checkpoint** `google/gemma-4-31B-it-qat-w4a16-ct` at `52f3f65bc7a02d555763bc923bd1d9094898219d`. Bootstrap checks the complete weight file SHA256. `load_base` explicitly requests `CompressedTensorsConfig(dequantize=True)` and replaces decoded quantization wrappers with ordinary frozen BF16 Linear layers. This is INT4/group32 decoded into BF16 storage; it is **not** Google's pre-quantization QAT-unquantized model and is not NF4 QLoRA. Only LoRA A/B matrices receive gradients.

Every rank loads the same W4 base. Rank 0 independently unpacks the first, middle and last rows of **every packed matrix**, compares them exactly to loaded BF16 values, and writes `base_audit.json`. The smoke test repeats this after training to check the sampled base values stayed frozen. This is a broad sampled weight check, not an exhaustive comparison of all elements. BF16 dense GEMMs and the W4 inference kernels may still differ numerically.

Each checkpoint records `base_identity.json`. **Old QAT-unquantized checkpoints cannot be resumed by this version**, and checkpoints missing identity are rejected. Use a new output directory and run bootstrap again (or re-run preparation with the new revision). Old checkpoints and validation reports remain historical evidence; they do not validate this loader.

The adapter export gate requires the original W4 model ID/revision, nonempty LoRA-only tensors and size below 3 GiB. Deploy only `checkpoint-N/adapter/`; optimizer and RNG files are for training recovery. Do not merge the adapter into a dense base and re-quantize it for an adapter-only deployment. Do not substitute TeichAI, an unquantized ancestor, or another quantizer as the frozen base.

Run `python test_w4_contract.py` before training. The original W4-plus-adapter inference gate described below is also required before a full training job or downstream evaluation. See `VALIDATION.md` for what has actually been tested.

For model selection, compare frozen base and adapter under identical tools, timeout, thinking configuration, compaction, task order and generation sampling. Use held-out tasks and multiple seeds for close comparisons; never train on those evaluation tasks. Report task resolution rate, runtime, output/reasoning tokens, tool-call failures and timeout rate. Teacher-forcing loss alone is insufficient. This package deliberately does not launch an expensive downstream benchmark automatically.

## Files

- `base_loading.py`, `validate_w4.py`, `test_w4_contract.py`: exact-source guards, W4 decoding audit, adapter export checks.
- `prepare.py`: native-template conversion and per-token weights; rejects truncation.
- `train.py`: single-card/DDP training, memory controls, checkpoints, smoke gate.
- `tracking.py`: optional W&B setup, metrics, and run-resume metadata.
- `test_tracking.py`, `test_tracking_distributed.py`: offline SDK, resume policy, and CPU distributed monitoring tests.
- `test_weighting.py`: causal-loss weighting and distributed normalization tests.
- `test_template.py`: native-template target-mask tests.
- `bootstrap.sh`, `requirements.txt`, `revisions.json`: reproducible setup.
- `VALIDATION.md` and `validation/`: measured evidence and limits.

The example dataset variables must be replaced with the authorized dataset name and the 40-character commit SHA from its Files/History page. No credentials are included. The data retains mixed upstream licensing; see its dataset card before redistribution. All cloud lifecycle management remains outside these scripts.
