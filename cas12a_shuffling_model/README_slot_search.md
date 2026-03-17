# Slot-search pipeline (T_family -> A_rank -> S_scan)

This note documents the composition-first search path added to this repository.

## Objective

- Use `T_family` (Cas12a-adapted ProtGPT2) only offline to label sampled chimera sequences.
- Train `A_rank` to learn ranking structure from slot composition and optional teacher-derived features.
- Distill `A_rank` behavior into `S_scan`, a tiny slot-level scorer for exhaustive `4^11` search.

## Canonical chimera representation

Each record is normalized to:

- `chimera_id`
- `slot_code_11` (11 chars over `A/L/F/M`)
- `slot_01 ... slot_11` (integers in `{0,1,2,3}`)
- `full_protein_sequence`
- `length`

Implementation: `src/cas12a_shuffling_model/composition/chimera_repr.py`

## New entry points

1. `cas12a-export-teacher-labels`
2. `cas12a-train-assistant-ranker`
3. `cas12a-train-slot-scorer`
4. `cas12a-scan-full-space`
5. `cas12a-rerank-shortlist`

## Configs

- `configs/slot_search.yaml` (formal path)
- `configs/slot_search_smoke.yaml` (smoke path)

## Notes

- The final production scanner is `S_scan` (slot-level), not a tiny sequence LM.
- Ranking-focused losses are used (`top + corr + filtered pairwise`), with near-tie filtering and easy/medium/hard pair mixing.
- Best checkpoint selection is ranking-based (`global_corr_chimera` by default), not plain loss.
