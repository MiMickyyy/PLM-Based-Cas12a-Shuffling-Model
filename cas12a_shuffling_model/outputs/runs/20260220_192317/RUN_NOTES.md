# RUN_NOTES

- Run timestamp: 20260220_192317
- Scope: autonomous pipeline execution (student scorer, ranking, smoke e2e, full run)
- Decision: due to compute/runtime constraints, full candidate search uses large sampled subset + teacher rerank shortlist (not exhaustive 4^11).
- Smoke E2E decision: used 12 distill samples (larger than smoke default 6) to avoid unstable NaN correlations.
- Full run decision: use sampled search size=1500, shortlist=500, teacher_rerank=150, top_k=50 for practical runtime.
- Full run decision: build distill set with n=320 and train student for 5 epochs.
- Runtime decision: interrupted initial full ranking attempt (sample_size=1500) because student scoring was sequential and too slow.
- Fix applied: vectorized mini-batch student scoring in `score_student` for practical runtime; rerun full ranking afterward.
- Full run completed with timestamped outputs under `/outputs/runs/20260220_192317/full/`.
- Ranking executed twice (seed 23 and 31) for top-k overlap stability figure generation.
