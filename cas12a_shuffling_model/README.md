# PLM-guided ranking pipeline for Cas12a domain shuffling

Teacher: ProtGPT2 (PLM prior)  
Student: GRU autoregressive LM (distillation surrogate)  
Goal: score/rank chimeras in a combinatorial design space using interpretable global + junction-level consistency scores.

## Safety statement
This study is a pure mathematical/computational modeling project for a mathematical modeling course. Any wet-lab content or results referenced are pre-existing or fictionalized examples used only for modeling context. This work does not involve any activities that impact humans or present biological safety risks. The broader research context is conducted under ethics oversight and approval at the University of California, Riverside, and this computational task itself has no safety risk.

## Quickstart
Create a local venv and install deps:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Build validated domain peptides (translates SnapGene `.dna`, validates vs parental proteins):

```bash
PYTHONPATH=../cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.build_domains \
  --domains-dir ../domains \
  --parents As=../AsCas12a.prot Fn=../FnCas12a.prot Lb=../LbCas12a.prot Mb2=../Mb2Cas12a.prot \
  --out-dir ../cas12a_shuffling_model/data/processed
```

Or use the repo-root config:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.build_domains \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

Reconstruct active chimera AA sequences from `Sequence_Result.xlsx`:

```bash
PYTHONPATH=../cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.reconstruct_actives \
  --sequence-results ../Sequence_Result.xlsx \
  --validated-domains ../cas12a_shuffling_model/data/processed/validated_domain_peptides.csv \
  --out-dir ../cas12a_shuffling_model/data/active
```

Config mode:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.reconstruct_actives \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

Teacher scoring for an existing CSV (requires `sequence_aa` or `combo_compact`):

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.score_teacher \
  --config cas12a_shuffling_model/configs/smoke.yaml \
  --input-csv cas12a_shuffling_model/data/active/active_chimeras_reconstructed.csv \
  --output-csv cas12a_shuffling_model/data/processed/active_teacher_scores.csv
```

Build a sampled distillation teacher-score set:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.build_distill_set \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

### Cas12a teacher adaptation (new)
Continue pretraining ProtGPT2 on `cas12a.fasta` (supports `auto` -> LoRA preferred, fallback partial fine-tune):

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.train_teacher_adapter \
  --config cas12a_shuffling_model/configs/smoke.yaml \
  --fasta cas12a.fasta \
  --method auto \
  --epochs 1
```

Evaluate baseline vs adapted teacher on natural validation + active/background chimera diagnostics:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.eval_teacher_adaptation \
  --config cas12a_shuffling_model/configs/smoke.yaml \
  --adapted-model-path /path/to/adapted_teacher_model
```

### Mixed distillation set (chimera + natural; new)
Generate mixed-source distill data where natural rows provide `global_score` and nullable `junction_*`:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.build_distill_set \
  --config cas12a_shuffling_model/configs/smoke.yaml \
  --source-mode mixed \
  --chimera-samples 12 \
  --natural-samples 16
```

Train GRU student on distill teacher scores:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.train_student \
  --config cas12a_shuffling_model/configs/smoke.yaml
```
`train_student` supports mixed-source masked loss: global distillation on all rows, junction distillation only where junction targets are finite.

Score sequences with student model (single or batch):

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.score_student \
  --config cas12a_shuffling_model/configs/smoke.yaml \
  --checkpoint /path/to/student_best.pt \
  --input-csv /path/to/sequences.csv \
  --output-csv /path/to/student_scores.csv
```

Fit calibration artifact (27 actives + background distill set):

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.calibrate \
  --config cas12a_shuffling_model/configs/default.yaml
```

Rank candidates (student shortlist + teacher rerank + calibration + diversity):

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.rank_candidates \
  --config cas12a_shuffling_model/configs/default.yaml
```

To use adapted teacher in rerank/scoring CLIs, pass local model path:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.rank_candidates \
  --config cas12a_shuffling_model/configs/default.yaml \
  --teacher-model-source local \
  --teacher-model-name-or-path /path/to/adapted_teacher_model
```

## Notes on reproducibility
- Teacher cache keys are versioned by `sequence_hash + teacher_model_fingerprint + junction_window`, so baseline and adapted teacher scores do not collide.
- `cas12a.fasta` is used in two places: teacher adaptation and student mixed distillation data.
- Long runs are resumable via checkpoints (teacher adaptation and exhaustive ranking modes).

Generate figures from ranked outputs:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.make_figures \
  --ranking-csv /path/to/candidate_top.csv \
  --ranking-csv-2 /path/to/candidate_top_second_run.csv \
  --out-dir /path/to/figures
```

## New 3-stage search pipeline (T_family -> A_rank -> S_scan)

This repo now also supports a composition-first ranking/search flow where the final exhaustive scanner is a tiny slot-level model (`S_scan`) rather than a tiny sequence LM.

### Stage 1: Offline teacher export (`T_family`)

Export teacher labels/features for sampled or user-provided chimera tables:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.export_teacher_labels \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --n-samples 120000 \
  --out-table cas12a_shuffling_model/outputs/slot_search/teacher_export/teacher_labels.parquet
```

Output table includes canonical chimera representation:
- `chimera_id`
- `slot_code_11` (`A/L/F/M`, 11 slots)
- `slot_01 ... slot_11` as integers in `{0,1,2,3}`
- `full_protein_sequence`
- `teacher_seq_score_raw`
- `teacher_seq_score_norm` (robust normalized by length bins)
- optional junction features `teacher_junction_*`

### Stage 2: Train assistant ranker (`A_rank`)

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.train_assistant_ranker \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --data-table cas12a_shuffling_model/outputs/slot_search/teacher_export/teacher_labels.parquet
```

Loss is ranking-oriented (`L_top + L_corr + L_pair`) with near-tie filtering and easy/medium/hard pair sampling.
Best checkpoint selection uses ranking metrics (default `global_corr_chimera`) instead of plain `val_loss`.

### Stage 3: Train tiny slot scorer (`S_scan`) distilled from `A_rank`

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.train_slot_scorer \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --data-table cas12a_shuffling_model/outputs/slot_search/teacher_export/teacher_labels.parquet \
  --assistant-checkpoint /path/to/assistant_best.pt
```

`S_scan` architecture:
- per-slot main effects
- pairwise slot interaction matrices (`4x4` per slot pair)
- tiny nonlinear MLP head

### Stage 4: Exhaustive full-space scan (`4^11`)

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.scan_full_space \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --slot-scorer-checkpoint /path/to/slot_scorer_best.pt
```

This scores all 4,194,304 combinations in batches and exports:
- `s_scan_shortlist.csv`
- `s_scan_top.csv`
- `s_scan_summary.json`

### Stage 5: Optional assistant rerank of shortlist

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.rerank_shortlist \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --shortlist-table /path/to/s_scan_shortlist.csv \
  --assistant-checkpoint /path/to/assistant_best.pt
```

When `slot_search.rerank.teacher_audit=true`, rerank also performs an offline teacher audit on top candidates and exports:
- `assistant_teacher_audit_top.csv`
- `assistant_teacher_audit_summary.json`

### Active-first rerank upgrade (top-tail focused)

For the current bottleneck (Top50 local ordering), use active-first settings:
- dual-head assistant (`teacher_head` weak prior + `active_head` dominant)
- active-local hard negatives mined from rerank shortlist
- active similarity prior in rerank (`final_score = assistant_score + beta * sim_to_active`)

Hard-negative mining:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.mine_active_hard_negatives \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --base-table /path/to/teacher_labels.csv \
  --rerank-table /path/to/assistant_reranked_all.csv \
  --active-table Sequence_Result.xlsx \
  --out-table /path/to/assistant_train_active_local.csv
```

Train active-first assistant (dual-head):

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.train_assistant_ranker \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --data-table /path/to/assistant_train_active_local.csv \
  --objective-mode active_first \
  --dual-head
```

Rerank with active prior + beta sweep:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.rerank_shortlist \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --shortlist-table /path/to/s_scan_shortlist.csv \
  --assistant-checkpoint /path/to/assistant_best.pt \
  --active-table Sequence_Result.xlsx \
  --active-prior-mode kernel_density_over_actives \
  --active-prior-beta 0.15 \
  --active-prior-beta-sweep 0.0,0.05,0.1,0.15,0.2,0.25,0.3 \
  --dual-head-alpha 0.15
```

Optional local basin expansion (Hamming-1 + selective Hamming-2):

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.expand_local_neighbors \
  --config cas12a_shuffling_model/configs/slot_search.yaml \
  --seed-table /path/to/assistant_reranked_top.csv \
  --active-table Sequence_Result.xlsx \
  --out-table /path/to/local_expanded_candidates.csv
```

Compare rerank variants with active-focused metrics:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.report_active_rerank \
  --named-tables baseline=/path/to/baseline_top.csv,active_prior=/path/to/active_prior_top.csv,active_first=/path/to/active_first_top.csv,dual_head=/path/to/dual_head_top.csv \
  --active-table Sequence_Result.xlsx \
  --out-json /path/to/active_rerank_report.json \
  --out-csv /path/to/active_rerank_report.csv
```

### Notes / assumptions
- Final production search model is `S_scan` (slot-level), not a sequence autoregressive student.
- `T_family` is used offline for labeling sampled chimera data only.
- If parquet engine is unavailable, table outputs automatically fall back to CSV.
- `slot_search_smoke.yaml` limits scan range and sample size for quick functional checks.
