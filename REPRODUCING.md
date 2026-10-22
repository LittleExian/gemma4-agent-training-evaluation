# Running the archived pipeline

This is an execution snapshot, not a self-contained model or benchmark dataset. Scripts retain the original directory layout under the public `/workspace/` placeholder. Choose your workspace and update the path constants or recreate that layout before running.

1. Obtain the Gemma base checkpoint `google/gemma-4-31B-it-qat-w4a16-ct` at revision `52f3f65bc7a02d555763bc923bd1d9094898219d` through your authorized model access. Obtain the competition tasks, repository snapshots and wheels separately. Their hashes are recorded in the result manifests; reference/test patches and original task packages are not redistributed here.
2. For inference, use the package versions in `environments/`. The evaluation environment is separate from training and from the per-project test environments. Do not install the inference PyTorch stack into a running training environment.
3. Read `training/prefix-v2/PREFIX_ALIGNMENT.md` and `README.md`. Start from the original W4 base, curate the task split, create a NEW tokenized cache, validate its deployment-prefix contract, and run the CPU and GPU smoke gates before full training. The archived configuration is 2 GPUs, microbatch1, accumulation4, task_mean, LR2e-5, rank16, seed42, one epoch/356 steps, save every5.
4. Download the intended adapter from the Hugging Face links in README. Morning checkpoint25 and prefix-v2 checkpoint25 are distinct experiments. Verify the adapter SHA against its manifest before serving.
5. For the inference server, explicitly pin `--tokenizer`, `--chat-template` and `--chat-template-content-format string`. The serving templates are from the same pinned base revision. Use the GPU admission checks, native LoRA route and health/route-effect checks in `evaluation/src/gemmacodeagent_morning25_server.py`.
6. Rebuild and calibrate task dependency environments on the new host. A copied historical gate is evidence, not proof that a new machine has the same dependencies. Run fresh gold/empty controls and compare the actual dependency freeze before agent execution.
7. `evaluation/src/gemmacodeagent_morning25_controller.py` is the archived final controller for the successful `full129-single-gpu0-r3` run. It starts 4 workers on one GPU, verifies source/data/adapter/calibration identities, uses the official evaluator with the local Landlock adaptation, and cleans up only its own recorded model process tree.

The effective submission is materialized as real files: the installed official submission compiler rejects symbolic links. Every task gets an independent prompt rendered to the actual 15-minute/56-tool budget. Submission compilation errors fail as infrastructure errors rather than model scores.

## Verification commands

From `evaluation/src` in a compatible evaluation environment:

```bash
python -m unittest test_gemmacodeagent_morning25_task test_gemmacodeagent_morning25_controller test_gemmacodeagent_morning25_server
```

For prefix training, use the test and verification commands in the pinned upstream README/PREFIX_ALIGNMENT, followed by the single-card and two-card stress smoke gates. Historical smoke results demonstrate the recorded machine/run only.

## Credentials

Provide your own credentials through environment variables or owner-only local files. No API token, SSH key, Hugging Face auth directory, W&B raw log or credential store is included. Never commit tokens. Hardware admission and file-permission checks should be kept enabled.

## Limits of the environment

The task runner uses `swegemma` subprocess sandboxes, task-specific Python environments, Landlock filesystem permissions and native-helper path guards. Landlock does not provide container-level network, CPU or memory quotas. The model service and test processes share host resources. Timings from this local setup are not a claim of identical official competition hardware/sandbox performance.
