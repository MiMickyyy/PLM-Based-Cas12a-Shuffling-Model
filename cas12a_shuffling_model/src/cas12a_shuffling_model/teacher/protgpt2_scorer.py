from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from cas12a_shuffling_model.io.loaders import sha256_text
from cas12a_shuffling_model.teacher.junction_scoring import (
    JunctionWindowConfig,
    compute_boundary_positions,
    score_windows_from_per_residue_ll,
)
from cas12a_shuffling_model.teacher.score_cache import ScoreCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProtGPT2Config:
    model_name: str = "nferruz/ProtGPT2"
    model_name_or_path: str | None = None
    model_source: str = "hf"  # hf | local
    model_revision: str | None = None
    adapter_path: str | None = None  # PEFT adapter path
    add_spaces: bool = True
    max_length: Optional[int] = None  # truncate if set

    @property
    def resolved_model_name_or_path(self) -> str:
        return str(self.model_name_or_path or self.model_name)

    @property
    def resolved_adapter_path(self) -> str | None:
        if self.adapter_path is None:
            return None
        p = str(self.adapter_path).strip()
        return p if p else None


@dataclass(frozen=True)
class TeacherScore:
    seq_hash: str
    seq_len: int
    global_score: float
    junction_scores: list[float]
    from_cache: bool = False

    @property
    def junction_mean(self) -> float:
        vals = [v for v in self.junction_scores if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def junction_min(self) -> float:
        vals = [v for v in self.junction_scores if v == v]
        return min(vals) if vals else float("nan")


def _format_sequence(seq_aa: str, add_spaces: bool) -> str:
    s = seq_aa.strip().upper()
    if add_spaces:
        return " ".join(list(s))
    return s


def detect_torch_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _path_fingerprint(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return f"missing:{p}"
    if p.is_file():
        st = p.stat()
        payload = f"file:{p.resolve()}:{st.st_size}:{int(st.st_mtime)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    tracked = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    parts = []
    root = p.resolve()
    for rel in tracked:
        fp = root / rel
        if fp.exists():
            st = fp.stat()
            parts.append(f"{rel}:{st.st_size}:{int(st.st_mtime)}")
    if not parts:
        payload = f"dir:{root}"
    else:
        payload = f"dir:{root}|" + "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_teacher_model_fingerprint(cfg: ProtGPT2Config) -> str:
    base = cfg.resolved_model_name_or_path
    revision = cfg.model_revision or ""
    model_source = str(cfg.model_source or "hf").lower()
    if Path(base).exists():
        base_sig = f"local:{_path_fingerprint(base)}"
    else:
        base_sig = f"hf:{base}@{revision}"

    adapter = cfg.resolved_adapter_path
    if adapter:
        if Path(adapter).exists():
            adapter_sig = f"adapter:{_path_fingerprint(adapter)}"
        else:
            adapter_sig = f"adapter:{adapter}"
    else:
        adapter_sig = "adapter:none"

    payload = (
        f"source={model_source}|base={base_sig}|{adapter_sig}|"
        f"spaces={int(cfg.add_spaces)}|maxlen={cfg.max_length}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _map_token_ll_to_residue_ll(token_ll: list[float], seq_len: int) -> list[float | None]:
    # per-residue list is length seq_len; residue 0 has no previous context.
    if seq_len <= 0:
        return []
    if seq_len == 1:
        return [None]
    if not token_ll:
        return [None] + [float("nan")] * (seq_len - 1)

    out: list[float | None] = [None]
    target_n = seq_len - 1
    src_n = len(token_ll)
    for i in range(target_n):
        # Linear index mapping from residue position into token_ll index.
        src_idx = int(i * src_n / target_n)
        if src_idx >= src_n:
            src_idx = src_n - 1
        out.append(float(token_ll[src_idx]))
    return out


class ProtGPT2Scorer:
    def __init__(
        self,
        *,
        config: ProtGPT2Config,
        window: JunctionWindowConfig,
        cache: ScoreCache,
        device: str | None = None,
    ):
        self.config = config
        self.window = window
        self.cache = cache
        self.device = detect_torch_device(device)
        self.model_fingerprint = build_teacher_model_fingerprint(config)
        self._model = None
        self._tokenizer = None
        self._model_max_positions: int | None = None

    def _fallback_to_cpu(self, reason: Exception | str) -> None:
        if self.device != "mps":
            raise RuntimeError(reason) if isinstance(reason, Exception) else RuntimeError(str(reason))
        assert self._model is not None
        logger.warning("MPS teacher forward failed; fallback to CPU. reason=%s", reason)
        self._model.to("cpu")
        self.device = "cpu"

    def _forward_with_fallback(self, *, input_ids, attention_mask=None):
        assert self._model is not None
        try:
            return self._model(input_ids=input_ids, attention_mask=attention_mask).logits
        except RuntimeError as e:
            if self.device != "mps":
                raise
            self._fallback_to_cpu(e)
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            return self._model(input_ids=input_ids, attention_mask=attention_mask).logits

    def _cache_key(self, seq_hash: str) -> str:
        return (
            f"protgpt2:v2|fp={self.model_fingerprint}|"
            f"wL={self.window.left}|wR={self.window.right}|{seq_hash}"
        )

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ProtGPT2 scoring requires `torch` and `transformers` installed."
            ) from e

        model_ref = self.config.resolved_model_name_or_path
        logger.info("Loading teacher model: %s", model_ref)
        tokenizer_kwargs = {"use_fast": True}
        if self.config.model_revision:
            tokenizer_kwargs["revision"] = self.config.model_revision
        self._tokenizer = AutoTokenizer.from_pretrained(model_ref, **tokenizer_kwargs)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model_kwargs = {"use_safetensors": True}
        if self.config.model_revision:
            model_kwargs["revision"] = self.config.model_revision
        try:
            self._model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
        except Exception:
            # Fallback for local checkpoints saved as .bin.
            model_kwargs.pop("use_safetensors", None)
            self._model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)

        adapter_path = self.config.resolved_adapter_path
        if adapter_path:
            try:
                from peft import PeftModel
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "adapter_path was provided but `peft` is not installed. "
                    "Install peft or remove teacher.adapter_path."
                ) from e
            logger.info("Loading teacher adapter: %s", adapter_path)
            self._model = PeftModel.from_pretrained(self._model, adapter_path)
            if hasattr(self._model, "merge_and_unload"):
                try:
                    self._model = self._model.merge_and_unload()
                except Exception:
                    # Keep adapter-wrapped model if merge fails.
                    pass

        if self._model.config.pad_token_id is None and self._tokenizer.pad_token_id is not None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
        model_max_positions = getattr(self._model.config, "n_positions", None) or getattr(
            self._model.config, "max_position_embeddings", None
        )
        self._model_max_positions = int(model_max_positions) if model_max_positions else None
        self._model.to(self.device)
        self._model.eval()
        logger.info(
            "Teacher device: %s | model_fingerprint=%s | model_max_positions=%s",
            self.device,
            self.model_fingerprint,
            self._model_max_positions,
        )

    def score_one(
        self,
        *,
        seq_aa: str,
        domain_lengths: Optional[Sequence[int]] = None,
    ) -> TeacherScore:
        seq_aa = seq_aa.strip().upper()
        if self.config.max_length is not None and len(seq_aa) > self.config.max_length:
            seq_aa = seq_aa[: self.config.max_length]

        seq_hash = sha256_text(seq_aa)
        key = self._cache_key(seq_hash)
        cached = self.cache.get(key)
        if cached is not None:
            return TeacherScore(
                seq_hash=cached.seq_hash,
                seq_len=cached.seq_len,
                global_score=cached.global_score,
                junction_scores=cached.junction_scores,
                from_cache=True,
            )

        self._lazy_load()
        assert self._model is not None and self._tokenizer is not None

        import torch
        import torch.nn.functional as F

        text = _format_sequence(seq_aa, self.config.add_spaces)
        token_max_length = self._model_max_positions
        enc = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=token_max_length is not None,
            max_length=token_max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)

        with torch.no_grad():
            logits = self._forward_with_fallback(input_ids=input_ids)  # [B, T, V]

        # Next-token log-probabilities for positions 1..T-1.
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        token_ll = token_ll[0].detach().cpu().tolist()

        # Tokenization and residue count are not always 1:1 for BPE models.
        # Use deterministic linear mapping to residue indices for junction windows.
        per_residue_ll = _map_token_ll_to_residue_ll(token_ll, len(seq_aa))

        global_score = float(sum(x for x in token_ll) / max(1, len(token_ll)))

        if domain_lengths is None:
            # If caller doesn't provide domain lengths, use uniform boundaries by splitting into 11 equal-ish parts.
            # This is only for debugging; the main pipeline should pass true lengths.
            L = len(seq_aa)
            approx = [L // 11] * 11
            approx[-1] += max(0, L - sum(approx))
            domain_lengths = approx

        boundaries = compute_boundary_positions(domain_lengths)
        junction_scores = score_windows_from_per_residue_ll(per_residue_ll, boundaries, self.window)

        self.cache.set(
            key=key,
            seq_hash=seq_hash,
            seq_len=len(seq_aa),
            global_score=global_score,
            junction_scores=junction_scores,
            meta={
                "model_name_or_path": self.config.resolved_model_name_or_path,
                "adapter_path": self.config.resolved_adapter_path,
                "model_fingerprint": self.model_fingerprint,
                "add_spaces": self.config.add_spaces,
                "device": self.device,
            },
        )

        return TeacherScore(
            seq_hash=seq_hash,
            seq_len=len(seq_aa),
            global_score=global_score,
            junction_scores=junction_scores,
            from_cache=False,
        )

    def score_many(
        self,
        *,
        seqs_aa: Sequence[str],
        domain_lengths_list: Optional[Sequence[Optional[Sequence[int]]]] = None,
        batch_size: int = 4,
    ) -> list[TeacherScore]:
        if domain_lengths_list is None:
            domain_lengths_list = [None] * len(seqs_aa)
        if len(domain_lengths_list) != len(seqs_aa):
            raise ValueError("domain_lengths_list length mismatch with seqs_aa")

        seqs: list[str] = []
        dlen_list: list[Optional[Sequence[int]]] = []
        keys: list[str] = []
        seq_hashes: list[str] = []
        results: list[TeacherScore | None] = [None] * len(seqs_aa)
        uncached_idx: list[int] = []

        for i, (raw_seq, domain_lengths) in enumerate(zip(seqs_aa, domain_lengths_list)):
            seq = str(raw_seq).strip().upper()
            if self.config.max_length is not None and len(seq) > self.config.max_length:
                seq = seq[: self.config.max_length]
            seq_hash = sha256_text(seq)
            key = self._cache_key(seq_hash)

            cached = self.cache.get(key)
            if cached is not None:
                results[i] = TeacherScore(
                    seq_hash=cached.seq_hash,
                    seq_len=cached.seq_len,
                    global_score=cached.global_score,
                    junction_scores=cached.junction_scores,
                    from_cache=True,
                )
            else:
                uncached_idx.append(i)

            seqs.append(seq)
            dlen_list.append(domain_lengths)
            keys.append(key)
            seq_hashes.append(seq_hash)

        if uncached_idx:
            self._lazy_load()
            assert self._model is not None and self._tokenizer is not None

            import torch
            import torch.nn.functional as F

            for start in range(0, len(uncached_idx), batch_size):
                end = min(start + batch_size, len(uncached_idx))
                idxs = uncached_idx[start:end]

                chunk_seqs = [seqs[i] for i in idxs]
                texts = [_format_sequence(s, self.config.add_spaces) for s in chunk_seqs]
                enc = self._tokenizer(
                    texts,
                    add_special_tokens=False,
                    padding=True,
                    truncation=self._model_max_positions is not None,
                    max_length=self._model_max_positions,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                with torch.no_grad():
                    logits = self._forward_with_fallback(
                        input_ids=input_ids, attention_mask=attention_mask
                    )

                logits = logits[:, :-1, :]
                targets = input_ids[:, 1:]
                target_mask = attention_mask[:, 1:].bool()
                log_probs = F.log_softmax(logits, dim=-1)
                token_ll_all = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

                token_ll_cpu = token_ll_all.detach().cpu()
                target_mask_cpu = target_mask.detach().cpu()

                for local_i, global_i in enumerate(idxs):
                    token_ll = [
                        float(v)
                        for v in token_ll_cpu[local_i][target_mask_cpu[local_i]].tolist()
                    ]
                    seq = seqs[global_i]
                    per_residue_ll = _map_token_ll_to_residue_ll(token_ll, len(seq))
                    global_score = float(sum(token_ll) / max(1, len(token_ll)))

                    domain_lengths = dlen_list[global_i]
                    if domain_lengths is None:
                        L = len(seq)
                        approx = [L // 11] * 11
                        approx[-1] += max(0, L - sum(approx))
                        domain_lengths = approx

                    boundaries = compute_boundary_positions(domain_lengths)
                    junction_scores = score_windows_from_per_residue_ll(
                        per_residue_ll, boundaries, self.window
                    )

                    self.cache.set(
                        key=keys[global_i],
                        seq_hash=seq_hashes[global_i],
                        seq_len=len(seq),
                        global_score=global_score,
                        junction_scores=junction_scores,
                        meta={
                            "model_name_or_path": self.config.resolved_model_name_or_path,
                            "adapter_path": self.config.resolved_adapter_path,
                            "model_fingerprint": self.model_fingerprint,
                            "add_spaces": self.config.add_spaces,
                            "device": self.device,
                            "batched": True,
                        },
                    )

                    results[global_i] = TeacherScore(
                        seq_hash=seq_hashes[global_i],
                        seq_len=len(seq),
                        global_score=global_score,
                        junction_scores=junction_scores,
                        from_cache=False,
                    )

        out: list[TeacherScore] = []
        for i, score in enumerate(results):
            if score is None:
                raise RuntimeError(f"score_many missing output at index={i}")
            out.append(score)
        return out
