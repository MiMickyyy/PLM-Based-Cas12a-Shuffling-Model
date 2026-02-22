#!/bin/zsh
set -euo pipefail
ROOT="/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML"
RUN_ROOT="/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645"
PID_FILE="$RUN_ROOT/logs/full_exhaustive_rank.pid"
PIPE_LOG="$RUN_ROOT/logs/full_pipeline_post.log"
RANK_DIR="$RUN_ROOT/full/ranking/formal_exhaustive_seed13"
TOP_CSV="$RANK_DIR/candidate_top.csv"
FIG_DIR="$RUN_ROOT/full/figures/formal_exhaustive_seed13"
REF_CSV="/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_125422/smoke/ranking/rank_1771707281_101/candidate_top.csv"

exec >> "$PIPE_LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] post-run watcher started"

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting ranking pid=$PID"
  while kill -0 "$PID" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ranking pid finished"
fi

if [[ -f "$TOP_CSV" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] generating figures"
  cd "$ROOT"
  PYTHONPATH="$ROOT/cas12a_shuffling_model/src" \
  .venv/bin/python -m cas12a_shuffling_model.cli.make_figures \
    --ranking-csv "$TOP_CSV" \
    --ranking-csv-2 "$REF_CSV" \
    --out-dir "$FIG_DIR" \
    --top-n-heatmap 50 \
    --max-k-overlap 50 \
    --log-level INFO
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] figures done: $FIG_DIR"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] candidate_top.csv not found, skip figures"
fi
