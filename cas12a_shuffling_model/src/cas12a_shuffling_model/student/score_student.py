from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F

from cas12a_shuffling_model.io.loaders import sha256_text
from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    domain_lengths_from_combo,
    validate_combo_compact,
)
from cas12a_shuffling_model.student.gru_model import GRUAutoregressiveLM
from cas12a_shuffling_model.student.vocab import AminoAcidVocab, BOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from cas12a_shuffling_model.teacher.junction_scoring import (
    JunctionWindowConfig,
    compute_boundary_positions,
    score_windows_from_per_residue_ll,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StudentScore:
    sequence_hash: str
    global_score: float
    junction_scores: list[float]

    @property
    def junction_mean(self) -> float:
        vals = [v for v in self.junction_scores if v == v]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    @property
    def junction_min(self) -> float:
        vals = [v for v in self.junction_scores if v == v]
        return float(min(vals)) if vals else float("nan")


def detect_torch_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _vocab_from_stoi(stoi: dict[str, int]) -> AminoAcidVocab:
    if PAD_TOKEN not in stoi:
        raise ValueError("Checkpoint vocab missing <pad> token")
    if BOS_TOKEN not in stoi:
        raise ValueError("Checkpoint vocab missing <bos> token")
    if UNK_TOKEN not in stoi:
        raise ValueError("Checkpoint vocab missing X token")
    itos = {int(i): tok for tok, i in stoi.items()}
    return AminoAcidVocab(stoi={str(k): int(v) for k, v in stoi.items()}, itos=itos)


class StudentScorer:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        window: JunctionWindowConfig,
        device: str | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.window = window
        self.device = detect_torch_device(device)
        self.model: GRUAutoregressiveLM
        self.vocab: AminoAcidVocab
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if "model_config" not in ckpt or "model_state_dict" not in ckpt:
            raise ValueError(f"Invalid student checkpoint format: {self.checkpoint_path}")
        vocab_stoi = ckpt.get("vocab_stoi")
        if not isinstance(vocab_stoi, dict):
            raise ValueError(f"Checkpoint missing vocab_stoi: {self.checkpoint_path}")
        self.vocab = _vocab_from_stoi(vocab_stoi)

        mc = ckpt["model_config"]
        self.model = GRUAutoregressiveLM(
            vocab_size=len(self.vocab.stoi),
            embed_dim=int(mc["embed_dim"]),
            hidden_dim=int(mc["hidden_dim"]),
            num_layers=int(mc["num_layers"]),
            dropout=float(mc.get("dropout", 0.0)),
            pad_idx=self.vocab.pad_id,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded student checkpoint: %s (device=%s)", self.checkpoint_path, self.device)

    @torch.no_grad()
    def score_one(self, *, sequence_aa: str, domain_lengths: Sequence[int] | None = None) -> StudentScore:
        seq = str(sequence_aa).strip().upper()
        if not seq:
            raise ValueError("sequence_aa is empty")

        token_ids = self.vocab.encode(seq)
        input_ids = [self.vocab.bos_id] + token_ids[:-1]
        targets = token_ids
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        y = torch.tensor(targets, dtype=torch.long, device=self.device).unsqueeze(0)

        logits = self.model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)[0]
        token_ll_list = [float(v) for v in token_ll.detach().cpu().tolist()]

        global_score = float(sum(token_ll_list) / len(token_ll_list))
        per_residue_ll = [None] + token_ll_list[1:] if len(token_ll_list) > 1 else [None]

        if domain_lengths is None:
            L = len(seq)
            dlen = [L // 11] * 11
            dlen[-1] += max(0, L - sum(dlen))
            domain_lengths = dlen
        boundaries = compute_boundary_positions(domain_lengths)
        junction_scores = score_windows_from_per_residue_ll(
            per_residue_ll=per_residue_ll, boundary_positions=boundaries, window=self.window
        )
        if len(junction_scores) != 10:
            junction_scores = (junction_scores + [float("nan")] * 10)[:10]

        return StudentScore(
            sequence_hash=sha256_text(seq),
            global_score=global_score,
            junction_scores=junction_scores,
        )

    @torch.no_grad()
    def score_sequences(
        self,
        *,
        sequences_aa: Sequence[str],
        domain_lengths_list: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 16,
    ) -> list[StudentScore]:
        seqs = [str(s).strip().upper() for s in sequences_aa]
        if any(not s for s in seqs):
            raise ValueError("score_sequences got empty sequence")
        if domain_lengths_list is None:
            domain_lengths_list = [None] * len(seqs)
        if len(domain_lengths_list) != len(seqs):
            raise ValueError("domain_lengths_list length mismatch")

        scores: list[StudentScore] = []
        for start in range(0, len(seqs), batch_size):
            end = min(start + batch_size, len(seqs))
            chunk_seqs = seqs[start:end]
            chunk_dlen = domain_lengths_list[start:end]

            token_ids = [self.vocab.encode(s) for s in chunk_seqs]
            lengths = [len(t) for t in token_ids]
            max_len = max(lengths)

            x = torch.full(
                (len(chunk_seqs), max_len), fill_value=self.vocab.pad_id, dtype=torch.long, device=self.device
            )
            y = torch.full(
                (len(chunk_seqs), max_len), fill_value=self.vocab.pad_id, dtype=torch.long, device=self.device
            )
            mask = torch.zeros((len(chunk_seqs), max_len), dtype=torch.bool, device=self.device)

            for i, toks in enumerate(token_ids):
                n = len(toks)
                x[i, :n] = torch.tensor([self.vocab.bos_id] + toks[:-1], dtype=torch.long, device=self.device)
                y[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
                mask[i, :n] = True

            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            token_ll = log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

            token_ll_cpu = token_ll.detach().cpu()
            mask_cpu = mask.detach().cpu()

            for i, seq in enumerate(chunk_seqs):
                n = int(mask_cpu[i].sum().item())
                vals = [float(v) for v in token_ll_cpu[i, :n].tolist()]
                global_score = float(sum(vals) / len(vals))
                per_residue_ll = [None] + vals[1:] if len(vals) > 1 else [None]

                dlen = chunk_dlen[i]
                if dlen is None:
                    L = len(seq)
                    dtmp = [L // 11] * 11
                    dtmp[-1] += max(0, L - sum(dtmp))
                    dlen = dtmp

                boundaries = compute_boundary_positions(dlen)
                junction_scores = score_windows_from_per_residue_ll(
                    per_residue_ll=per_residue_ll,
                    boundary_positions=boundaries,
                    window=self.window,
                )
                if len(junction_scores) != 10:
                    junction_scores = (junction_scores + [float("nan")] * 10)[:10]
                scores.append(
                    StudentScore(
                        sequence_hash=sha256_text(seq),
                        global_score=global_score,
                        junction_scores=junction_scores,
                    )
                )
        return scores

    def score_batch_rows(
        self,
        *,
        rows_df: pd.DataFrame,
        validated_domains: dict[tuple[str, int], str] | None = None,
        combo_col: str = "combo_compact",
        seq_col: str = "sequence_aa",
        batch_size: int = 16,
    ) -> pd.DataFrame:
        prepared_rows = []
        seqs: list[str] = []
        dlen_list: list[Sequence[int] | None] = []
        for _, row in rows_df.iterrows():
            combo = None
            if combo_col in rows_df.columns and pd.notna(row.get(combo_col)):
                combo = validate_combo_compact(str(row.get(combo_col)))

            seq = str(row.get(seq_col, "")).strip().upper() if seq_col in rows_df.columns else ""
            if not seq:
                if combo is None or validated_domains is None:
                    raise ValueError("Missing sequence_aa and cannot reconstruct without combo+validated_domains")
                seq = build_sequence_from_combo(combo, validated_domains)

            dlen = None
            if combo is not None and validated_domains is not None:
                dlen = domain_lengths_from_combo(combo, validated_domains)
            prepared_rows.append((row.to_dict(), seq))
            seqs.append(seq)
            dlen_list.append(dlen)

        scores = self.score_sequences(
            sequences_aa=seqs, domain_lengths_list=dlen_list, batch_size=int(batch_size)
        )
        out_rows = []
        for (row_dict, seq), score in zip(prepared_rows, scores):
            rec = dict(row_dict)
            rec["sequence_aa"] = seq
            rec["sequence_hash"] = score.sequence_hash
            rec["global_score"] = score.global_score
            rec["junction_mean"] = score.junction_mean
            rec["junction_min"] = score.junction_min
            for i, v in enumerate(score.junction_scores, start=1):
                rec[f"junction_{i:02d}"] = v
            out_rows.append(rec)
        return pd.DataFrame(out_rows)
