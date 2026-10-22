#!/usr/bin/env bash
set -euo pipefail
: "${CODEAGENT_ROOT:?Set CODEAGENT_ROOT to persistent storage}"
workers="${1:?Usage: launch_worker.sh WORKERS INDEX}"
index="${2:?Usage: launch_worker.sh WORKERS INDEX}"
[[ "$workers" =~ ^[34]$ && "$index" =~ ^[0-3]$ ]] && (( index < workers )) || { echo 'Invalid worker partition'; exit 2; }
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$CODEAGENT_ROOT/runs/distributed_model_comparison"
log="$CODEAGENT_ROOT/runs/distributed_model_comparison/worker_${index}_of_${workers}.log"
nohup "$CODEAGENT_ROOT/envs/eval/bin/python" -u "$package/run.py" --scope all --workers "$workers" --worker-index "$index" --concurrency "${MM_CONCURRENCY:-16}" >> "$log" 2>&1 < /dev/null &
echo "Driver PID: $!; log: $log"
echo 'Check this log for model-health failures before assuming scoring has started.'
