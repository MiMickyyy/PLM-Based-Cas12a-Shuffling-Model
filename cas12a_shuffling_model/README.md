# PLM-guided ranking pipeline for Cas12a domain shuffling

Teacher: ProtGPT2 (PLM prior)  
Student: GRU autoregressive LM (distillation surrogate)  
Goal: score/rank chimeras in a combinatorial design space using interpretable global + junction-level consistency scores.

## Safety statement
This study is a pure mathematical/computational modeling project for a mathematical modeling course. Any wet-lab content or results referenced are pre-existing or fictionalized examples used only for modeling context. This work does not involve any activities that impact humans or present biological safety risks. The broader research context is conducted under ethics oversight and approval at the University of California, Riverside, and this computational task itself has no safety risk.

## Quickstart (current milestone: data + reconstruction)
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

Train GRU student on distill teacher scores:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.train_student \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

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

Generate figures from ranked outputs:

```bash
PYTHONPATH=cas12a_shuffling_model/src .venv/bin/python -m cas12a_shuffling_model.cli.make_figures \
  --ranking-csv /path/to/candidate_top.csv \
  --ranking-csv-2 /path/to/candidate_top_second_run.csv \
  --out-dir /path/to/figures
```
