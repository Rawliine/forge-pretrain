#!/bin/bash
set -euo pipefail

# ── Usage: ./experiment.sh "baseline adamw lr3e-4" ────────────────────────────
RUN_NAME="${1:?Usage: ./experiment.sh <run_name>}"
RESULTS_FILE="results.md"
EVAL_SHARDS="${EVAL_SHARDS:-/home/data/chunk_0049.bin}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
LOG_FILE="$LOG_DIR/${TIMESTAMP//[: ]/_}_${RUN_NAME// /_}.log"

# ── Init results.md if missing ────────────────────────────────────────────────
if [ ! -f "$RESULTS_FILE" ]; then
cat > "$RESULTS_FILE" << 'EOF'
# Experiment Results

| # | Run | Date | Params | Steps | Tokens/s | Train Loss | Val Loss | Perplexity | Time | Notes |
|---|-----|------|--------|-------|----------|------------|----------|------------|------|-------|
EOF
fi

echo "▶ Starting run: $RUN_NAME"
echo "  Log: $LOG_FILE"

# ── Run training, tee to log ──────────────────────────────────────────────────
START=$(date +%s)
./run_local.sh 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$(( END - START ))
ELAPSED_FMT=$(printf '%dm%02ds' $((ELAPSED/60)) $((ELAPSED%60)))

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "✗ Training failed (exit $TRAIN_EXIT)"
    RUN_NUM=$(grep -c '^|[0-9]' "$RESULTS_FILE" 2>/dev/null || echo 0)
    RUN_NUM=$((RUN_NUM + 1))
    echo "| $RUN_NUM | $RUN_NAME | $TIMESTAMP | - | - | - | - | - | - | $ELAPSED_FMT | **FAILED** |" >> "$RESULTS_FILE"
    exit 1
fi

# ── Extract metrics from training log ─────────────────────────────────────────
# Adjust these greps to match your train.py print format
LAST_STEP=$(grep -oP 'step\s*[:=]\s*\K[0-9]+' "$LOG_FILE" | tail -1 || echo "?")
TRAIN_LOSS=$(grep -oP 'loss\s*[:=]\s*\K[0-9]+\.[0-9]+' "$LOG_FILE" | tail -1 || echo "?")
TOKENS_SEC=$(grep -oP 'tok/s\s*[:=]\s*\K[0-9,.]+' "$LOG_FILE" | tail -1 || echo "?")

# ── Eval ──────────────────────────────────────────────────────────────────────
echo "▶ Evaluating checkpoint..."
VAL_OUTPUT=$(python eval_checkpoint.py --checkpoint checkpoint.pt --data "$EVAL_SHARDS" 2>&1)
echo "$VAL_OUTPUT"

PARAMS=$(echo "$VAL_OUTPUT" | grep -oP 'params_m\s*[:=]\s*\K[0-9]+\.[0-9]+M' || echo "?")
VAL_LOSS=$(echo "$VAL_OUTPUT" | grep -oP 'val_loss\s*[:=]\s*\K[0-9]+\.[0-9]+' || echo "?")
PPL=$(echo "$VAL_OUTPUT" | grep -oP 'perplexity\s*[:=]\s*\K[0-9]+\.[0-9]+' || echo "?")

# ── Append to results table ──────────────────────────────────────────────────
RUN_NUM=$(grep -c '^| [0-9]' "$RESULTS_FILE" 2>/dev/null || echo 0)
RUN_NUM=$((RUN_NUM + 1))

echo "| $RUN_NUM | $RUN_NAME | $TIMESTAMP | $PARAMS | $LAST_STEP | $TOKENS_SEC | $TRAIN_LOSS | $VAL_LOSS | $PPL | $ELAPSED_FMT | |" >> "$RESULTS_FILE"

echo ""
echo "✓ Done. Results appended to $RESULTS_FILE"
echo "  Params=$PARAMS  Steps=$LAST_STEP  Tok/s=$TOKENS_SEC  Train=$TRAIN_LOSS  Val=$VAL_LOSS  PPL=$PPL  Time=$ELAPSED_FMT"
