# Deployment-prefix aligned training (2026-10-22)

This revision fixes a confirmed serialization defect, not a demonstrated repair-score gain. Start a NEW run from the same pinned W4 checkpoint. Do not resume a pre-fix adapter/optimizer or reuse old tokenized JSONL files.

## What changed

`serialization.py` renders the deployment prompt from history alone with `add_generation_prompt=True`, `enable_thinking=True`, and `preserve_thinking=False`. The native template still formats target tool calls, arguments, reasoning and end markers. Its lookahead-dependent history is used only to extract target syntax, never as the training input.

When deployment already prefills `<|channel>thought\n`, that opener belongs to the masked prompt. A target with reasoning continues inside it; a target without reasoning emits `<channel|>` then the original action. No synthetic reasoning is invented. Histories whose native full-chat rendering changes an earlier boundary are reconstructed using the actual standalone prompt. Prompt tokens and continuation tokens are encoded separately so BPE cannot retokenize the already-consumed generation prefix.

The full prompt receives zero loss. Reasoning text retains weight0.25, actions and generated protocol delimiters retain weight1. Standalone compaction targets remain excluded; summaries in input history remain. Base weights, LR, objective choices, batch and LoRA rank are unchanged by this training fix.

Every row records `gemma4-deployment-prefix-v2`, prompt length and prompt-token SHA256. `train.py` validates this contract before allocating a model. Old caches and old resume manifests are rejected. `prepare.py` refuses an existing output directory.

The archived collaborator `weighted_lora_streamed_ce.py` is retained byte-for-byte, with its first-order CE recomputation, so using this release does not silently remove that memory optimization. CPU equivalence tests cover its loss and gradients; this is not a new GPU memory benchmark.

## Rebuild and verify before training

Run inside `training/weighted_reasoning_lora`. `$CURATED` is the already audited main dataset, with train.jsonl and validation.jsonl. Use persistent storage for `$WORK`.

```bash
export WORK=/workspace/weighted_reasoning_prefix_v2
export CURATED=/path/to/curated-main
REV=52f3f65bc7a02d555763bc923bd1d9094898219d
python prepare.py --data "$CURATED" --output "$WORK/tokenized" \
  --revision "$REV" --reasoning-weight 0.25
export TOKENIZER="$WORK/tokenized/tokenizer"
python test_serialization.py
python verify_prefix_alignment.py --data "$CURATED" \
  --tokens "$WORK/tokenized" --tokenizer "$TOKENIZER" \
  --output "$WORK/prefix_alignment.json"
```

Inspect `preparation.json`: an unexpected exclusion is a failed data review. The audited main dataset should retain all2841 train and214 validation decisions. `bootstrap.sh` now runs these checks automatically when preparing a fresh training root. Existing raw dataset and older token caches remain unchanged.

Run the single-GPU smoke procedure from README, then a short controlled run:

```bash
N=2 # 2,3,4 PRO6000 96GB cards on the same host
# Keep the old LR/objective for an isolated serialization comparison.
torchrun --standalone --nproc_per_node="$N" train.py \
  --data "$WORK/tokenized/train.jsonl" --output "$WORK/taskmean-prefix-v2" \
  --revision "$REV" --objective task_mean --steps 25 \
  --save-every 5 --accum 4 --lr 2e-5
```

W&B setup remains user-supplied through the README environment/CLI flags; no account or secret is embedded. Evaluate early checkpoints and the base with the same corrected prompt. Do not assume a full epoch helps. Changing GPU count changes global batch unless accumulation is adjusted. Historical step25/step65 results are diagnostics, not validation of this new code.

## Evaluation budget contract

The active `v5_6000` configuration now has56 calls,15 minutes,160 turns and thinking budget6000. The distributed129-task submission has the same actual budget and corrected text. Both evaluation entry points render budget statements from the effective eval_config.yaml, save a per-task `effective_submission`, and record source/effective prompt hashes and limits in manifest.json. They do not edit a shared submission while jobs run. Resume with changed prompt/config is rejected; use a new run name.

`original_baseline` and `v5_4096` remain legacy28/8 configurations. Comparing them directly to current `v5_6000` changes more than the thinking budget. Use matched limits for a thinking-budget-only experiment. `artifacts/submission.zip` is the current15min submission; the old package is archived as `submission_legacy_8min.zip`. The local requested budget does not override competition-enforced caps.

Run from repository root:

```bash
python src/test_budget_contract.py
```

Use the distributed experiment from the complete repository: its runner imports `src/budget_contract.py`. Existing copied submissions are rendered to effective limits at task startup. Fresh run names are required after upgrading.

## Scope of validation

CPU checks cover real native-tokenizer rendering, no-reasoning actions/submissions, reasoning precedence, lookahead boundaries, loss masks at weights0/.25/1, overlength rejection, malformed inputs, cache/resume rejection, fractional time budgets, shared-source immutability and per-run prompt conflicts. Full-dataset results are in `validation/prefix_alignment_20261022/`.

No new optimizer step, GPU serving test, or repair benchmark was run for this revision. The prior server probe found a separate one-space tool-declaration separator difference between HF rendering and vLLM; these CPU tests do not claim end-to-end server token-ID parity. Before a costly training campaign, run the README smoke gate and compare actual server tokenization on representative requests. The300-second proxy timeout/retry behavior is unchanged here; address it in a separately tested evaluation change.
