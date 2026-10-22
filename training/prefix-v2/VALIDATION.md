# Current W4-aligned validation — 2026-10-20 UTC

The default training base is now the **original deployment W4A16 checkpoint decoded to BF16**, not its QAT-unquantized ancestor. The records below this section describing the old base are historical only.

On one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB; driver 595.91.07), with Torch 2.8.0+cu128, Transformers 5.20.2, compressed-tensors 0.15.0 and PEFT 0.21.2:

- The full original `model.safetensors` SHA256 matched the pinned deployment file.
- Independently unpacked 10,268,160 sampled values (first/middle/last rows of all 410 packed matrices). Every value exactly matched the decoded BF16 model, both before and after training.
- **Exactly one optimizer step** on the longest real training sequence: 19,825 tokens; weighted loss 0.9738829; gradient norm 1.9802240; checked LoRA update 1.999987e-5; step time 79.224 s; peak allocated memory 82.387 GiB.
- Only LoRA A/B parameters were trainable. The exported adapter has 820 nonempty tensors, rank 16, and weighs 489,840,816 bytes. Its base ID/revision refer to the original W4 checkpoint. No full base weights are exported.
- Saved-adapter reload on the decoded base produced maximum absolute logit difference **0.0**. Training/save/frozen-base-audit/reload took 188.584 s, excluding loading and the initial base audit.
- Actual GPU training also wrote W&B offline metrics successfully; no online account/key was used.
- Three CPU source/resume/INT4 contract tests passed; two-rank CPU checkpoint identity/optimizer/RNG recovery passed; four W&B tests passed. The native-template test passed against the pinned W4 tokenizer.

Evidence: `validation/w4_smoke.json` and `validation/w4_preparation.json`. This single-step test establishes pipeline correctness, not repair accuracy. Actual 2–4 GPU training remains untested; the unchanged DDP training path and CPU tests do not establish multi-GPU throughput.

## Original W4 plus LoRA inference gate

Passed on the same PRO 6000, with original packed W4 weights and the competition wheelhouse v31 patched vLLM 0.19.1 / Transformers 5.13.1 / compressed-tensors 0.15.0.1, Torch 2.10.0+cu128, huggingface_hub 1.16.1, flashinfer 0.6.6. The original W4 used the MarlinLinearKernel; it was not replaced by dense BF16 inference.

On two fixed prompts, one next token each, with `logprobs_mode="raw_logits"`:

- No adapter versus a zero-B LoRA control: maximum raw-logit difference **0.0** over shared returned top-20 token IDs.
- Zero-B control versus the actual one-step LoRA: maximum raw-logit difference **0.25** over shared returned top-20 token IDs.
- Both adapter requests completed, and the process exited 0. No additional optimizer updates or repository-repair evaluation were performed.

Evidence: `validation/w4_deployment_gate.json`. This proves adapter loading and a nonzero numerical effect on the original W4 runtime. It does not establish task-quality improvement, full-vocabulary logit equivalence, or a 4xL4 competition run. The gate used a 32,768 context configuration with short prompts.

An initial environment attempt failed because the same-numbered PyPI Transformers wheel lacks the competition patches; a subsequent attempt exposed a huggingface_hub version mismatch. `setup_deployment_env.sh` now pins the compatible Hub version and verifies/explicitly installs all three patched wheels by SHA256. The training environment is kept separate.

---

# Historical QAT-unquantized validation — 2026-10-20 UTC

One NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB reported by nvidia-smi), CUDA driver 595.91.07, PyTorch 2.8.0+cu128. BF16 base, LoRA rank/alpha 16, dropout 0.05, 122,429,440 trainable parameters (0.3900% of the loaded 31,395,515,952 parameters).

## Actual single-GPU training

Three optimizer updates on real examples from the pinned training split:

| Sequence tokens | Weighted loss | Gradient norm before clipping | Step seconds | Peak allocated GiB |
|---:|---:|---:|---:|---:|
| 1,139 | 1.59310 | 1.22241 | 1.717 | 60.570 |
| 15,048 | 1.90043 | 2.73339 | 51.203 | 77.542 |
| 19,825 | 0.96171 | 1.11915 | 78.763 | 83.300 |

Each update changed the checked LoRA B parameter tensor by approximately 2e-5 at its largest element. Losses are from different examples and must not be interpreted as a learning curve. The run saved checkpoints after each update. Reloading the final adapter and evaluating the same fixed probe gave a **maximum absolute logit difference of 0.0**. The training/checkpoint/reload phase took 154.06 seconds, excluding model loading (about 3m37s), setup and downloads. The process exited normally and released GPU memory.

An earlier unchunked run passed the short and P95 cases but ran out of memory on 19,825 tokens. The final training core uses nested 2,048-token FFN checkpointing, BF16 autocast, gradient checkpointing, selective logits, and expandable CUDA allocator segments. It preserves the full real sequence. The observed long-sequence result does not prove that all 24K contexts fit.

## Numerical and format gates

- Native-template test: prompt tokens masked; thinking weight 0.25; action weight 1; overlength examples rejected; malformed tool JSON rejected.
- Weighted-gradient test on 1/2/3/4 CPU processes: maximum gradient differences against a centralized reference were 0, 0, 2.38e-7, and 1.19e-7, respectively. Zero-weight prompt positions had zero direct gradient.
- Checkpoint helpers passed 1/2/3/4-process CPU tests for optimizer state, per-rank RNG restoration, completion markers, and world-size mismatch rejection. This does not replace a real multi-GPU restart test.
- No examples excluded for length or formatting in preparation: 2,951 training decisions (26,915,299 full-sequence tokens), 216 validation decisions (1,791,555 tokens). Tokens include repeated history; they are not unique teacher-output tokens.

See machine-readable evidence in `validation/`.

## Limits

Physical 2-4-GPU and multi-host runs have **not** been performed. CPU distributed tests establish normalization correctness, not NCCL behavior or throughput. Multi-rank checkpoint/resume code is provided; it has not been validated by a real multi-GPU training restart. The GPU smoke run preceded the addition of per-rank RNG checkpoint files; that addition leaves the tested forward/loss/backward core unchanged.

No held-out repair benchmark or submission-score comparison has been run with this three-step adapter. It is a pipeline verification artifact, not a trained production model. Compressed-tensors inference compatibility and model quality require separate validation. The base is the QAT-unquantized checkpoint, not numerically identical to a quantized deployment checkpoint.


## W&B integration validation — 2026-10-20 UTC

The monitoring update was checked separately on a CPU pod, using the real pinned W&B SDK 0.31.0. `test_tracking.py` passed all four tests: disabled/default configuration, allowlisted config and online resume/account-switch policy with a fake backend, metric aggregation/ETA, and a real offline run whose persisted journal contains the logged loss and exit status. No online account or API key was used.

`test_tracking_distributed.py` passed with 2, 3, and 4 real CPU/Gloo ranks, exercising the production `gather_rank_stats` helper. Synthetic global token counts were 300, 600, and 1,000 respectively; every rank's memory statistics were retained. This verifies aggregation, not GPU performance or NCCL. Python compilation and `git diff --check` passed.

The earlier single-GPU training smoke predates this monitoring integration. Online authentication/upload and end-to-end multi-GPU training with W&B have not been tested. Each trainer should configure their own account and verify their run URL during the smoke test.
