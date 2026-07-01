#!/bin/bash

# Phase 2 Sanity Audit Monitoring — Real-time Results Tracker
# Watches for job 4354629 completion and extracts key metrics

JOB_ID=4354629
OUTPUT_DIR="/projects/u6fb/myprojects/UniCount/checkpoints/scaffold_rex_stage15_4353822"
SANITY_JSON="${OUTPUT_DIR}/sanity_audit.json"
LOG_DIR="/projects/u6fb/myprojects/UniCount/logs"

echo "🟡 Phase 2 Sanity Audit Monitor — Job 4354629"
echo "=============================================="
echo "Watching for completion..."
echo ""

while true; do
  STATUS=$(sacct -j $JOB_ID --format=State --noheader | head -1 | xargs)
  ELAPSED=$(sacct -j $JOB_ID --format=Elapsed --noheader | head -1 | xargs)

  echo -ne "\r⏱️  Status: $STATUS | Elapsed: $ELAPSED"

  if [[ "$STATUS" == "COMPLETED" ]] || [[ "$STATUS" == "FAILED" ]]; then
    echo ""
    echo ""
    break
  fi

  sleep 5
done

echo ""
if [[ "$STATUS" == "FAILED" ]]; then
  echo "❌ Job FAILED"
  echo ""
  echo "=== STDERR ==="
  tail -50 "$LOG_DIR"/scaffold_rex_audit_v2_*.err 2>/dev/null || echo "No stderr yet"
  exit 1
fi

echo "✅ Job COMPLETED"
echo ""

# Wait for output file to exist
MAX_WAIT=30
WAITED=0
while [[ ! -f "$SANITY_JSON" ]] && [[ $WAITED -lt $MAX_WAIT ]]; do
  echo "⏳ Waiting for $SANITY_JSON to be written... ($WAITED/$MAX_WAIT)"
  sleep 2
  ((WAITED++))
done

if [[ ! -f "$SANITY_JSON" ]]; then
  echo "❌ Output JSON not found after 60s"
  echo ""
  echo "=== Available logs ==="
  ls -lah "$LOG_DIR"/scaffold_rex_audit_v2_* 2>/dev/null
  exit 1
fi

echo "✅ sanity_audit.json found"
echo ""

# Extract key results
echo "📊 AUDIT RESULTS"
echo "================="

TOTAL=$(jq '.meta.total_samples // 0' "$SANITY_JSON" 2>/dev/null)
PARSE_OK=$(jq '.meta.parse_ok // 0' "$SANITY_JSON" 2>/dev/null)
ZERO_COUNT=$(jq '.meta.zero_count // 0' "$SANITY_JSON" 2>/dev/null)

echo "Total Samples: $TOTAL"
echo "Parse Success: $PARSE_OK"
echo "Zero-Count Rows: $ZERO_COUNT"
echo ""

# Find 7.jpg result
RESULT_7=$(jq '.rows[] | select(.image | contains("7.jpg") or . == "7") | {image, gt_count, pred_count}' "$SANITY_JSON" 2>/dev/null)

if [[ -z "$RESULT_7" ]]; then
  echo "⚠️  Could not find 7.jpg in results"
  echo ""
  echo "Available images:"
  jq -r '.rows[].image' "$SANITY_JSON" 2>/dev/null | head -5
else
  echo "🎯 CRITICAL RESULT: 7.jpg"
  echo "$RESULT_7" | jq '.'

  PRED=$(echo "$RESULT_7" | jq '.pred_count // 0')
  GT=$(echo "$RESULT_7" | jq '.gt_count // 0')

  echo ""
  if [[ $PRED -gt 0 ]]; then
    echo "✅ SUCCESS: pred_count=$PRED (GT=$GT) — 0-COUNT BUG IS FIXED! 🎉"
  else
    echo "❌ FAILURE: pred_count=$PRED (GT=$GT) — Still returning 0"
  fi
fi

echo ""
echo "Full results saved to: $SANITY_JSON"
echo "Log files:"
ls -lah "$LOG_DIR"/scaffold_rex_audit_v2_* 2>/dev/null || echo "(not found yet)"
