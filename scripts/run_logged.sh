#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "usage: scripts/run_logged.sh <phase> <commands-file>" >&2
  exit 2
fi

phase="$1"
commands="$2"
log_dir="outputs/logs/${phase}"
mkdir -p "$log_dir"
summary="$log_dir/summary.txt"
: > "$summary"

step=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  step=$((step + 1))
  name=$(printf '%02d' "$step")
  log="$log_dir/${name}.log"
  start=$(date +%s)
  echo "[$(date '+%F %T')] $name START $line" | tee -a "$summary"
  set +e
  { echo "\$ $line"; bash -c "$line" < /dev/null; } 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  elapsed=$(( $(date +%s) - start ))
  echo "[$(date '+%F %T')] $name END status=$status ${elapsed}s" | tee -a "$summary"
  if [ "$status" -ne 0 ]; then
    echo "stopped at step $name; see $log" | tee -a "$summary"
    exit "$status"
  fi
done < "$commands"
