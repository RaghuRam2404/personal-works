#!/usr/bin/env bash
# Step 13 daily performance monitor — runs at 4PM via launchd
# Fetches metrics for ALL published carousels within the 15-day window (any batch)
# Logs: /tmp/dadfit_step13.log

WORKSPACE="/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content"
VENV="$WORKSPACE/Carousels/.venv/bin/python3"
SCRIPT="$WORKSPACE/Carousels/scripts/step13_monitor.py"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') === Step 13 fetch (all batches, last 15 days) ===" >> /tmp/dadfit_step13.log

"$VENV" "$SCRIPT" fetch >> /tmp/dadfit_step13.log 2>&1

echo "=== done (exit $?) ===" >> /tmp/dadfit_step13.log
