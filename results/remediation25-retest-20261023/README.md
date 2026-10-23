# Remediation checkpoint-25 retest, 2026-10-23

Completed all 129 tasks: 54/126 eligible resolved, 3 agent session timeouts, 3 reference-control exclusions, zero infrastructure error results. Two PRO 6000 GPUs (physical IDs 0 and 2), two workers each.

The adapter is the remediation checkpoint-25, not a checkpoint from the later prefix-v2 training. Sampling uses top_k=-1, seed=42, thinking budget 6000, 15 minutes, 56 tools, 160 turns. CUDA graph capture and auto→openai content-format resolution are verified for both servers. Initial request proof contains parameters and hashes only. GPU0 launch source snapshot and GPU2/current source are tracked separately.

All agent patches come from agent-mode invocations without supplied patches. Control patch contents, full task prompts, dataset rows, raw test output, requests/responses/traces, credentials and model weights are excluded. Original raw artifact hashes and sizes are inventories only. Paths are generalized; original and public-copy hashes are distinguished. Source-index references identical files already published in this repository. Runtime dependency evidence records whether the existing pinned runtime has identical bytes.
