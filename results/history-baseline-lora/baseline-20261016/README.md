# baseline-20261016

This completed full129 evaluation has 51/126 eligible tasks resolved, 9 agent session timeouts, and 3 reference-control exclusions. All 129 unique IDs have rows in `task-results.json` and CSV.

Runtime: 8 minutes, 28 tools, 80 turns; system prompt budget agrees. GPU IDs [0, 2, 3], 12 total slots. These configurations differ in budget, concurrency, adapter and source versions; scores are descriptive and are not an adapter-only controlled comparison.

`generated-patches/` contains only patches from agent-mode invocations with no supplied patch. Empty generated files are retained. Gold, empty-control and reference patch contents, dataset text, full LLM requests/responses/traces, credentials and weights are omitted. Original traces have hashes and sizes only. Error strings are classified and hashed; raw process/test output is omitted. Environment profiles contain dependency versions and reference counts only.

Code provenance is in `source-index.json`; identical sanitized bytes reuse existing repository files. Sources with differing execution hashes are labeled and must not be treated as exact historical execution copies. Unknown or unrecorded values remain null. Server paths are generalized to `/workspace` and the server address to `SERVER_HOST`.
