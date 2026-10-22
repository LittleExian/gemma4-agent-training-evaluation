> **Superseded encoding:** the legacy reproduction statements below describe the2026-10-21 revision. The current encoder intentionally changes those tokens to match the generation prefix. Follow [PREFIX_ALIGNMENT.md](PREFIX_ALIGNMENT.md) before training.

# Regression audit and controlled retraining

The published epoch-1 adapter is **not recommended for deployment**: on the
locked paired 20-task development set it solved 5/20 versus 11/20 for the base,
with mean elapsed time 715 versus 406 seconds. This is a development result,
not a leaderboard score. The changes below are testable hypotheses and engineering
safeguards, **not evidence that a new adapter already improves repair accuracy**.

## What the data audit actually found

All 6,781 request records were scanned. Tool feedback is joined by
`(attempt_id, tool_call_id)`, not by tool text or task alone. There were no
conflicting feedback values. Among 2,951 training targets / 3,097 tool calls:

- One observed `FileEditError`; no `BudgetExceeded` or identical-text edit targets.
- 86 `CommandError` responses. A failing reproduction test is not a bad action;
  these are retained rather than blanket-filtered.
- Three tool calls had no subsequent recorded feedback; unknown is not success
  and is not automatically grounds for deletion.
- 109 targets were **compaction requests**, not repair decisions. They have an
  explicit summarization prompt and no tools. Their weighted token mass was
  49,851.75 / 442,992 = 11.25%. Compaction exists at deployment too, so this is
  not automatically contamination. Per the requested training policy, exclude standalone summary targets by default. Retain summaries already present in repair histories.
- The 18 highest-supervision tasks contributed 36.78% of total weighted tokens.
  These are token-weight statistics, not measured gradient contributions.

The hypothesis that widespread failed teacher edits caused the student's edit
loops is **not supported** by this audit. Do not claim that removing one row
will recover the six lost solves. Early checkpoints are still needed to distinguish
an initially harmful update from later degradation.

## 1. Preserve raw data; generate a versioned audit

Run from this directory, with the original pinned HF dataset downloaded to
`$DATA`. It must include `requests.jsonl.gz`, `attempts.jsonl.gz`, `train.jsonl`,
and `validation.jsonl`. Dataset revision:
`d912c71174a4522199f07470dfb1b82eb17050b3`.

```bash
python curate.py --data "$DATA" --output "$WORK/curated-main-v1"
# Optional legacy comparison only:
python curate.py --data "$DATA" --output "$WORK/curated-legacy-v1" --compaction keep
```

An existing output directory is rejected. Source/output hashes and per-row audit
records are written. Only demonstrably invalid edit/budget actions, malformed
arguments, and truncated targets are quarantined. If a multi-tool target includes
one invalid action the entire target is quarantined; its original history remains
unchanged in later recovery examples. No synthetic success labels are introduced.
Gold/empty acceptance, selected success, task identity and task-disjoint splits
are checked. Reference patches are not used as training answers.

The default main dataset retains 2,841 train / 214 validation rows; 109 / 2
compaction targets are archived separately and never fed into main training.
Explicit legacy keep mode retains 2,950 / 216.
The original HF release is not overwritten by this script.

## 2. Tokenize without changing the deployed base or input history

Use the same pinned competition W4 base and native template as the main README.

```bash
python prepare.py --data "$WORK/curated-main-v1" --output "$WORK/tokens-r025" \
  --revision 52f3f65bc7a02d555763bc923bd1d9094898219d --reasoning-weight 0.25
```

`prepare.py` also excludes standalone compaction targets by default, even when
passed the original dataset directly. `--include-compaction` is only for explicit
legacy reproduction. Use curated input to also remove invalid edit targets.

`--reasoning-weight 0` masks reasoning supervision but preserves the reasoning
in the token sequence. It is **not** a no-thinking inference mode. Action and
boundary tokens remain supervised. Segment counts remain correct at weights
0, 0.25, and 1. Default 0.25 reproduces the old encoding exactly: all 3,167
archived examples were checked token-for-token and weight-for-weight.

## 3. Train short controlled runs before an entire epoch

Existing `--objective token_mean` is retained as the default for reproducibility.
The new `--objective task_mean` averages weighted token loss within each decision,
then decisions within each task, then tasks. Uniform row sampling uses importance
factor `N / (T * rows_in_task)`. The global denominator is sampled row count,
not the sum of sampled importance weights. DDP's gradient averaging is accounted
for. DistributedSampler may pad up to world_size-1 examples as before; record this
small epoch-level deviation from exact task balance.

Example for **2, 3, or 4 GPUs**, each a 96 GB PRO 6000:

```bash
NGPU=3
torchrun --standalone --nproc_per_node="$NGPU" train.py \
  --data "$WORK/tokens-r025/train.jsonl" --output "$WORK/taskmean-step25" \
  --revision 52f3f65bc7a02d555763bc923bd1d9094898219d \
  --objective task_mean --steps 25 --save-every 5 --accum 4 --lr 2e-5
```

Use a different output directory for every experiment. Keep W&B setup from the
main README: the trainer supplies their own account, project and API credentials.
Never embed credentials in these scripts or commits.

Suggested controlled comparisons, **one change at a time**:

1. Same curated main data (no standalone compaction targets), 0.25 reasoning, seed and update count:
   `token_mean` versus `task_mean`.
2. On the selected objective, 0.25 versus 0 reasoning supervision.
3. Standalone compaction remains excluded in the main recipe; a legacy comparison requires explicit opt-in.
4. If updates already hurt early, test lower learning rate or adapter strength
   separately. Do not label a combined change as evidence for one component.

Each run saves exact Python source and hashes, training-data hash, objective,
base revision and optimizer/batch settings. Resume requires a matching manifest;
legacy runs without one cannot silently resume under a changed objective.
This is deliberate. Keep old code to reproduce/resume an old experiment.

## 4. Gate continuation on real paired repair evaluations

A lower teacher-forced training loss is not a repair-quality gate. Evaluate early
checkpoints with the same 6000 budget, 900-second agent limit, tool limits,
compaction policy, hardware/concurrency and calibrated repositories as the base.
Keep every failed attempt. Use 8 rolling slots when the measured hardware supports
it; do not change the model's per-task limits to improve throughput metrics.

The result ledger must contain `arm`, `task_id`, boolean `resolved`,
`elapsed_seconds`, and `process_exit_code`. Use a separately locked JSON list of
task IDs, created **before** running, not inferred from the completed results.

```bash
python assess_results.py --results "$EVAL/results.jsonl" --tasks locked_tasks.json \
  --output "$EVAL/quality_gate.json" --baseline base --candidate lora
```

The default development policy rejects missing/duplicate/infra-failed results,
a decrease in solve count, or >25% mean elapsed-time increase. Timeout/budget
failures remain failures, not dropped samples. A passing gate is permission to
investigate further, not proof of generalization or statistical significance.
Changing the 25% engineering tolerance should be declared before evaluation.
Use the separate external validation split as well as the development benchmark;
retain untouched tasks for final selection. Repeat close results with matched
settings. Stop expanding a training run that fails the gate.

## Validation performed for this revision

- CPU unit checks: curation, objectives, paired result gate, resume provenance.
- Two-, three- and four-process CPU/Gloo gradient comparisons against a direct
  task-averaged reference: maximum absolute error below 1.2e-7. These are not
  multi-GPU memory or throughput benchmarks.
- All 3,167 original examples retain identical native tokens and default weights.
- A keep-mode control was tokenized: 2,950 / 216 examples. The main dataset excludes standalone compaction (2,841 / 214).
- One real PRO 6000 step on the longest 19,825-token sample: loss 0.973883,
  gradient norm 2.008306, adapter delta 1.99999e-5, 79.20 seconds, 82.49 GiB
  peak allocated memory. Frozen base audit unchanged; checkpoint reload logits
  exactly equal. W&B offline logging succeeded. This one-row smoke cannot prove
  task balancing improves repair quality. This smoke used the keep-mode control before the final compaction-exclusion policy; training mechanics are unchanged.
- The complete prior 40-result paired ledger correctly fails the new quality gate.

Machine-readable reports are in `validation/remediation_20261021/`.

To run the lightweight checks before training:

```bash
python test_curation.py
python test_objectives.py
python test_quality_gate.py
python test_provenance.py
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 test_task_objective_distributed.py
```

For encoding reproducibility against a previously tokenized release:

```bash
python verify_encoding.py --data "$DATA" --baseline-tokens "$OLD_TOKENS" \
  --tokenizer "$BASE_TOKENIZER" --output encoding_check.json
```

## Runtime compaction is distinct from training targets

The competition harness supports context compaction (see the official
[Getting Started notebook](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent/input)).
Our evaluated protocol enables `EventsCompactionConfig` with token threshold
14,336, interval 15, overlap 2 and event retention 5, independently of the 32K
model context limit. This update does not disable runtime compaction. It removes
standalone summary-generation targets from LoRA training and validation while
preserving summaries already visible inside actual repair histories.

The final main dataset has 2,841 examples from 180 training tasks and 214 from
20 validation tasks, with zero standalone compaction targets. Its retained fields
were checked against the raw release. The exact 19,825-token sample used in the
GPU smoke remains present with identical input IDs and weights. The direct
`prepare.py` default-exclusion and explicit legacy opt-in paths were also tested.
