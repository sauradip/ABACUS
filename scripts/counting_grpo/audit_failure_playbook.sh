#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/fast/nobackup/scratch4weeks/am04485/Codes/UniCount}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/checkpoints/scaffold_sft_stage1}"
AUDIT_JSON_PATH="${AUDIT_JSON_PATH:-}"

red_alert() {
  echo "[RED ALERT] $*"
}

info() {
  echo "[INFO] $*"
}

pick_latest_file() {
  local pattern="$1"
  find "$LOG_DIR" -maxdepth 1 -type f -name "$pattern" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

resolve_audit_json() {
  if [[ -n "$AUDIT_JSON_PATH" && -f "$AUDIT_JSON_PATH" ]]; then
    echo "$AUDIT_JSON_PATH"
    return 0
  fi

  local candidates
  candidates=$(find "$CKPT_ROOT" -maxdepth 3 -type f \( -name 'zero_shot_audit_results.json' -o -name 'zero_shot_point_audit_step100.json' -o -name '*point_audit*.json' \) -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-)

  if [[ -n "$candidates" ]]; then
    echo "$candidates"
    return 0
  fi

  return 1
}

main() {
  if [[ ! -d "$LOG_DIR" ]]; then
    red_alert "Log directory not found: $LOG_DIR"
    exit 2
  fi

  local latest_log latest_err
  latest_log=$(pick_latest_file 'scaffold_sft_*.log')
  latest_err=$(pick_latest_file 'scaffold_sft_*.err')

  if [[ -z "${latest_log:-}" && -z "${latest_err:-}" ]]; then
    red_alert "No scaffold logs found in $LOG_DIR"
    exit 2
  fi

  info "Latest log: ${latest_log:-<missing>}"
  info "Latest err: ${latest_err:-<missing>}"

  echo
  info "==== Purity Guard Check (TRAINING:) ===="
  if [[ -n "${latest_log:-}" && -f "$latest_log" ]]; then
    grep -n 'TRAINING:' "$latest_log" || true
    if grep -Eiq 'TRAINING:.*(lm_head|latent_queries|q_former)' "$latest_log"; then
      red_alert "Forbidden trainable modules found in TRAINING block (lm_head/latent_queries/q_former)."
    else
      info "No forbidden modules found in TRAINING block."
    fi
  else
    red_alert "Missing latest log file, cannot inspect TRAINING block."
  fi

  echo
  info "==== Pivot Marker Scan ===="
  if [[ -n "${latest_log:-}" && -f "$latest_log" ]]; then
    grep -nE 'Launching Stage 1 pivot run|step\s*100|Zero-Shot Point Audit|Coordinate bounds|Token diversity|Audit passed|Proceeding with Stage 1 SFT' "$latest_log" || true
  fi

  echo
  info "==== Failure Signal Scan ===="
  if [[ -n "${latest_err:-}" && -f "$latest_err" ]]; then
    grep -nE 'Traceback|RuntimeError|Error\(|size mismatch|CUDA|OOM|No space left|FAILED' "$latest_err" || true
  fi

  echo
  info "==== Audit JSON Parse ===="
  local audit_json
  audit_json=""
  if audit_json=$(resolve_audit_json); then
    info "Audit JSON: $audit_json"
    python3 -c '
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
bounds = data.get("bounds_pass")
diversity = data.get("diversity_pass")
rows = data.get("rows", [])
print(f"bounds_pass={bounds}")
print(f"diversity_pass={diversity}")
print(f"num_rows={len(rows)}")
for idx, row in enumerate(rows[:5], start=1):
    pred = row.get("prediction_text", "")
    pred = pred.replace("\n", " ")
    if len(pred) > 240:
        pred = pred[:240] + "..."
    print(f"sample_{idx}.bounds_pass={row.get('"'"'bounds_pass'"'"')}")
    print(f"sample_{idx}.prediction_text={pred}")
' "$audit_json"
  else
    info "Audit JSON not found yet (searched under $CKPT_ROOT)."
  fi

  echo
  info "==== Checkpoint Sanity ===="
  local ckpt100="$CKPT_ROOT/checkpoint-100"
  local adapter_safetensors="$ckpt100/adapter_model.safetensors"
  local adapter_bin="$ckpt100/adapter_model.bin"
  if [[ -f "$adapter_safetensors" || -f "$adapter_bin" ]]; then
    info "checkpoint-100 adapter exists."
    if [[ -f "$adapter_safetensors" ]]; then
      info "Found: $adapter_safetensors"
    fi
    if [[ -f "$adapter_bin" ]]; then
      info "Found: $adapter_bin"
    fi
  else
    red_alert "checkpoint-100 adapter missing at $ckpt100 (expected adapter_model.safetensors or adapter_model.bin)."
    local latest_ckpt
    latest_ckpt=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
    if [[ -n "$latest_ckpt" ]]; then
      info "Latest available checkpoint directory: $latest_ckpt"
    else
      info "No checkpoint directories found under $CKPT_ROOT"
    fi
  fi

  echo
  info "Playbook run complete."
}

main "$@"