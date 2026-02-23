# PLM-Based Cas12a Domain-Shuffling Model  
### A reproducible computational pipeline for PLM-guided ranking of chimeric Cas12a designs

## Overview

This repository implements a **protein language model (PLM)-guided ranking pipeline** for **Cas12a domain shuffling**.  
The project combines:

- **Teacher model**: ProtGPT2 (family-level sequence prior / consistency scorer)
- **Student model**: GRU autoregressive language model (distilled surrogate for fast large-scale screening)
- **Calibration layer**: small-sample calibration using experimentally identified active chimeras
- **Search module**: candidate ranking in the combinatorial domain-shuffling design space (sampled or exhaustive-surrogate mode)

The overall goal is to rank domain-shuffled Cas12a chimeras using **interpretable sequence consistency signals**, including:

1. **Global consistency score** (full-length sequence prior)
2. **Junction-level consistency scores** (local compatibility at domain boundaries)

---

## Scientific Motivation

Domain shuffling enables recombination of functional modules from multiple parental Cas12a proteins, but the design space grows combinatorially and quickly exceeds practical experimental screening capacity. This repository provides a computational framework to:

- learn a **family prior** from natural Cas12a sequences,
- score candidate chimeras by sequence-level “naturalness” and junction compatibility,
- and prioritize a small set of candidates for downstream experimental evaluation.

---

## Pipeline Summary

The pipeline consists of six stages:

### 1) Domain fragment validation and peptide reconstruction
- Input: 44 DNA domain fragments (11 slots × 4 parents)
- Handles possible cloning/enzyme overhangs by ORF search
- Translates and validates each fragment against the corresponding parental protein
- Outputs a validated domain peptide table

### 2) Active chimera reconstruction
- Input: experimentally observed active chimera slot compositions (A/L/F/M per slot)
- Reconstructs full-length amino acid sequences from validated domains
- Exports a clean AA-level active chimera dataset

### 3) Teacher scoring (ProtGPT2)
- Scores sequences using a PLM-derived prior
- Produces:
  - `global_score`
  - `junction_01` ... `junction_10`
  - `junction_mean`
  - `junction_min`
- Supports score caching by `sequence_hash` for reuse across runs

### 4) Distillation set generation
- Samples candidate combinations
- Reconstructs AA sequences
- Uses teacher scoring to create a teacher-labeled distillation dataset
- This dataset is used to train a faster student model

### 5) Student training and calibration
- Trains a GRU autoregressive student model to approximate teacher scores
- Fits a calibration model using active chimeras + background examples
- Outputs calibrated scores (`calibrated_prob`) for ranking

### 6) Candidate ranking and figure generation
- Candidate ranking modes:
  - **sampled** (fast validation)
  - **exhaustive student scan + teacher rerank** (formal ranking mode)
- Final output includes:
  - ranked candidates (`candidate_top.csv`)
  - audit files (shortlist, reranked shortlist)
  - figures for interpretation and presentation

---

## Repository Structure

```text
PLM-Based-Cas12a-Shuffling-Model/
├── cas12a_shuffling_model/          # Python package (CLI + configs + src)
├── domains/                         # 44 domain DNA fragments (As/Lb/Fn/Mb2 × 11)
├── AsCas12a.prot                    # parental full-length protein
├── FnCas12a.prot
├── LbCas12a.prot
├── Mb2Cas12a.prot
├── Sequence_Result.xlsx             # active chimera slot compositions
├── cas12a.fasta                     # natural Cas12a family sequences (training corpus)
├── requirements.txt
├── pyproject.toml
└── README.md
````

---

## Installation

### Requirements

* Python 3.9+
* PyTorch-compatible environment
* Virtual environment recommended

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Quick Start

> Below are the core CLI steps for the end-to-end workflow.

### 1) Build validated domain peptides

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.build_domains \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

### 2) Reconstruct active chimeras

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.reconstruct_actives \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

### 3) Score active sequences with the teacher model (ProtGPT2)

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.score_teacher \
  --config cas12a_shuffling_model/configs/smoke.yaml \
  --input-csv cas12a_shuffling_model/data/active/active_chimeras_reconstructed.csv \
  --output-csv cas12a_shuffling_model/data/processed/active_teacher_scores.csv
```

### 4) Build a teacher-labeled distillation dataset

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.build_distill_set \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

### 5) Train the GRU student model

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.train_student \
  --config cas12a_shuffling_model/configs/smoke.yaml
```

### 6) Score sequences with the student model

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.score_student \
  --config cas12a_shuffling_model/configs/smoke.yaml \
  --checkpoint /path/to/student_best.pt \
  --input-csv /path/to/sequences.csv \
  --output-csv /path/to/student_scores.csv
```

### 7) Fit calibration artifacts

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.calibrate \
  --config cas12a_shuffling_model/configs/default.yaml
```

### 8) Rank candidates (student shortlist + teacher rerank + calibration + diversity)

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.rank_candidates \
  --config cas12a_shuffling_model/configs/default.yaml
```

### 9) Generate figures

```bash
PYTHONPATH=cas12a_shuffling_model/src python -m cas12a_shuffling_model.cli.make_figures \
  --ranking-csv /path/to/candidate_top.csv \
  --out-dir /path/to/figures
```

---

## Ranking Modes

### A) Sampled mode (fast validation)

Used for smoke testing and interface validation:

* sample candidate combinations
* score with student
* rerank shortlist with teacher
* export top candidates

### B) Exhaustive-surrogate mode (formal ranking)

Recommended for the final run:

* enumerate the full combinatorial space (4^11)
* score all combinations with the student model
* retain a large student shortlist
* rerank shortlist with the teacher model
* apply calibration and diversity filtering
* export final Top-K candidates

This mode avoids the cost of teacher scoring the full combinatorial space while preserving global coverage.

---

## Outputs

Typical output files include:

* `validated_domain_peptides.csv` — validated translated domains
* `active_chimeras_reconstructed.csv` — reconstructed active chimera AA sequences
* `distill_teacher_scores*.csv` — teacher-labeled distillation dataset
* `student_best.pt` — trained GRU checkpoint
* `calibration_model.pkl` / calibration artifacts
* `candidate_top.csv` — final ranked candidate list
* `candidate_all_scored.csv` — full scored shortlist/rerank audit table
* figures:

  * rank–score curve
  * slot × parent heatmap
  * calibrated probability vs score diagnostics
  * top-K overlap (stability)

---

## Reproducibility Notes

* **Sequence hash-based caching** is used for teacher scores to prevent recomputation.
* **Timestamped output directories** are generated for run isolation.
* **Config-driven execution** (`smoke.yaml`, `default.yaml`) supports reproducible runs.
* For long exhaustive runs, checkpointing/resume is recommended.

---

## Safety Statement

This repository is a **pure mathematical/computational modeling project** developed for a mathematical modeling course.
Any wet-lab content or results referenced are **pre-existing or fictionalized examples** used only as modeling context.

This computational work does **not** involve:

* activities affecting humans,
* pathogen handling,
* or biological safety risks.

The broader research context is conducted under institutional ethics oversight and approval at the University of California, Riverside, and this computational task itself presents no safety risk.

---

## Citation (Suggested)

If you use this repository in course work or internal research notes, please cite the repository and describe the pipeline as:

> PLM-guided ranking of Cas12a domain-shuffling chimeras using ProtGPT2 teacher scoring, GRU student distillation, and calibration-based candidate prioritization.

---

## License

MIT License.

```


```

[1]: https://github.com/MiMickyyy/PLM-Based-Cas12a-Shuffling-Model "GitHub - MiMickyyy/PLM-Based-Cas12a-Shuffling-Model"
