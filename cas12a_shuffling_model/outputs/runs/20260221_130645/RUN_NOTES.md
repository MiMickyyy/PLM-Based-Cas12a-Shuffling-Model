# RUN_NOTES

- Run timestamp: 20260221_130645
- Goal: formal exhaustive ranking mode (4^11 full-space student scan + teacher rerank shortlist + Top-50 export).
- Primary config: /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/full/full_exhaustive.yaml
- Student checkpoint: /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260220_192317/full/student/run_1771645952/student_best.pt
- Calibration artifact: /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260220_192317/full/calibration/cal_1771646408

## Key parameters

- mode: exhaustive
- total combos: 4,194,304
- student_shortlist_size: 20,000
- teacher_rerank_size: 2,000
- final_top_k: 50
- student_batch_size: 1,024
- progress_every: 51,200
- checkpoint_every_batches: 50
- diversity min_hamming: 2

## Commands

- (pending) run exhaustive ranking
- (pending) generate figures

## Notes / decisions

- Use `mps` for student and teacher inference; if unavailable, fallback to CPU via CLI defaults.
- Keep resume/checkpoint enabled and never delete prior outputs.

## Restart / execution details

- 2026-02-21 13:14 local: initial exhaustive run started in foreground; checkpoint reached 153,600 combos.
- 2026-02-21 13:25 local: resumed with larger `progress_every` and `checkpoint_every_batches` to reduce checkpoint overhead.
- 2026-02-21 13:28 local: relaunched as detached background process (PPID=1) for autonomous completion.
- Active PID file: `/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/logs/full_exhaustive_rank.pid`
- Active log: `/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/logs/full_exhaustive_rank.log`

### Detached command

```bash
cd /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML && \
PYTHONPATH=/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/src \
OMP_NUM_THREADS=$(sysctl -n hw.ncpu) MKL_NUM_THREADS=$(sysctl -n hw.ncpu) \
.venv/bin/python -m cas12a_shuffling_model.cli.rank_candidates \
  --config /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/full/full_exhaustive.yaml \
  --out-dir /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/full/ranking \
  --run-id formal_exhaustive_seed13 \
  --mode exhaustive --resume --device mps --teacher-device mps \
  --student-checkpoint /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260220_192317/full/student/run_1771645952/student_best.pt \
  --calibration-model /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260220_192317/full/calibration/cal_1771646408/calibration_model.joblib \
  --calibration-meta /Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260220_192317/full/calibration/cal_1771646408/calibration_meta.json \
  --shortlist-size 20000 --teacher-rerank-size 2000 --top-k 50 \
  --student-batch-size 1024 --progress-every 102400 --checkpoint-every-batches 100 \
  --seed 13 --log-level INFO
```

- Post-run watcher launched (detached) to auto-generate figures when ranking completes.
  - watcher pid file: `/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/logs/post_watcher.pid`
  - watcher log: `/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/logs/full_pipeline_post.log`
  - watcher script: `/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/full/run_after_ranking.sh`

- 2026-02-21 13:41 local: started fresh exhaustive attempt in new folder `formal_exhaustive_seed13` (previous partial checkpoint preserved as `formal_exhaustive_seed13_attempt1_134136`).
- 2026-02-21 13:45 local: detached restart v2 with lower checkpoint frequency (`checkpoint_every_batches=1000`) to reduce I/O overhead.
- 2026-02-21 14:04 local: progress milestone `204800 / 4194304` logged.

## Engineering updates during run window

- Optimized exhaustive student heap update path in `src/cas12a_shuffling_model/cli/rank_candidates.py` to avoid materializing full row dicts for non-qualifying candidates.
- Added teacher batch API (`score_many`) and teacher-batch CLI/config plumbing for future rerank acceleration/testing.
- Verified test suite remains green (`22 passed`).

## Completion summary (2026-02-21)

- Exhaustive student scan completed: `4,194,304 / 4,194,304` (see `student_exhaustive_summary.json`).
- Student scan runtime: `18,807.41 s` (~5h 13m 27s).
- Teacher rerank + calibration + exports completed at `2026-02-21 20:05:53` local time.
- Figures auto-generated at `2026-02-21 20:06:49` local time by detached watcher.

### Final artifacts

- Ranking directory:
  `/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/full/ranking/formal_exhaustive_seed13`
- Figures directory:
  `/Users/mickysmacbookpro/Downloads/Domain_Shuffling_ML/cas12a_shuffling_model/outputs/runs/20260221_130645/full/figures/formal_exhaustive_seed13`

### Final counts

- student_shortlist.csv: `20,000` rows
- teacher_reranked.csv: `2,000` rows
- candidate_top.csv: `50` rows
- candidate_all_scored.csv: `2,000` rows

### Notable outputs

- Top-1 combo: `LLLFLFLLFLA`
- Top-50 calibrated_prob range: `[0.3413, 0.4884]`
- teacher cache hits during rerank: `1 / 2000` (~0.05%)
