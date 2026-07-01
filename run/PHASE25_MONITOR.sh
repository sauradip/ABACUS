#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/projects/u6fb/myprojects/UniCount"
cd "$REPO_DIR"

job_id="${1:-}"
if [[ -z "$job_id" ]]; then
  job_id="$(squeue --me --noheader --format '%A %j' | awk '$2 ~ /rankdpo_stage25/ {print $1; exit}')"
fi

if [[ -z "$job_id" ]]; then
  echo "No active rankdpo_stage25 job found."
  exit 1
fi

echo "Monitoring Stage2.5 job: $job_id"
echo "== sacct =="
sacct -j "$job_id" --format=JobID,JobName,State,Elapsed,ExitCode,Start -n | sed '/^$/d' || true

echo "== queue =="
squeue --me | awk -v id="$job_id" 'NR==1 || $1==id || $3 ~ /rankdpo_stage25/' || true

log_file="logs/rankdpo_stage25_${job_id}.log"
err_file="logs/rankdpo_stage25_${job_id}.err"

echo "== log tail ($log_file) =="
if [[ -f "$log_file" ]]; then
  tail -n 80 "$log_file"
else
  echo "Log not yet created"
fi

echo "== err tail ($err_file) =="
if [[ -f "$err_file" ]]; then
  tail -n 80 "$err_file"
else
  echo "Err log not yet created"
fi

echo "== metric grep =="
if [[ -f "$log_file" ]]; then
  grep -E "dpo_loss|rewards/accuracy|reward|loss|eval" "$log_file" | tail -n 40 || true
else
  echo "No metrics yet"
fi
