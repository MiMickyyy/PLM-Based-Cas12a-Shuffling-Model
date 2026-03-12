from __future__ import annotations

import json
import logging
import math
import random
import time
from functools import partial
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cas12a_shuffling_model.io.cas12a_corpus import (
    Cas12aSequence,
    read_cas12a_fasta,
    sample_sequences,
    split_train_val_indices,
)
from cas12a_shuffling_model.teacher.protgpt2_scorer import detect_torch_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherAdaptConfig:
    seed: int = 13
    base_model_name_or_path: str = "nferruz/ProtGPT2"
    base_model_revision: str | None = None
    add_spaces: bool = True
    method: str = "auto"  # auto | lora | partial | full
    partial_last_n_layers: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    min_len: int = 300
    max_len_filter: int | None = None
    deduplicate: bool = False
    train_val_fraction: float = 0.1
    max_train_sequences: int | None = None
    max_val_sequences: int | None = None
    max_length: int | None = 2048
    batch_size: int = 1
    eval_batch_size: int = 1
    grad_accum_steps: int = 8
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 0
    cpu_threads: int | None = None
    interop_threads: int | None = None
    log_every_steps: int = 10


class _TokenizedCausalDataset(Dataset):
    def __init__(self, token_ids_list: Sequence[list[int]]):
        self.token_ids_list = list(token_ids_list)

    def __len__(self) -> int:
        return len(self.token_ids_list)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        ids = self.token_ids_list[idx]
        return {"input_ids": ids}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpu_threads(cfg: TeacherAdaptConfig) -> None:
    if cfg.cpu_threads is not None and int(cfg.cpu_threads) > 0:
        torch.set_num_threads(int(cfg.cpu_threads))
    if cfg.interop_threads is not None and int(cfg.interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(cfg.interop_threads))
        except RuntimeError:
            # Can only be set once per process; safe to ignore.
            pass


def _format_sequence(seq_aa: str, add_spaces: bool) -> str:
    seq = str(seq_aa).strip().upper()
    if add_spaces:
        return " ".join(list(seq))
    return seq


def _tokenize_sequences(
    sequences: Sequence[Cas12aSequence],
    *,
    tokenizer,
    add_spaces: bool,
    max_length: int | None,
) -> list[list[int]]:
    tokenized: list[list[int]] = []
    for rec in sequences:
        text = _format_sequence(rec.sequence_aa, add_spaces=add_spaces)
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
        )
        ids = [int(x) for x in enc["input_ids"]]
        if len(ids) < 2:
            continue
        tokenized.append(ids)
    return tokenized


def _collate_clm_batch(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), fill_value=pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _count_trainable_params(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _maybe_apply_lora(model, cfg: TeacherAdaptConfig) -> tuple[Any, str]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception:
        return model, "unavailable"

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    wrapped = get_peft_model(model, lora_cfg)
    return wrapped, "applied"


def _freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _apply_partial_finetune(model, cfg: TeacherAdaptConfig) -> None:
    _freeze_all(model)

    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True

    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        for p in model.transformer.ln_f.parameters():
            p.requires_grad = True

    n_last = max(0, int(cfg.partial_last_n_layers))
    if (
        n_last > 0
        and hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
    ):
        blocks = model.transformer.h
        for block in blocks[-n_last:]:
            for p in block.parameters():
                p.requires_grad = True


def _resolve_adapt_method(model, cfg: TeacherAdaptConfig) -> tuple[Any, str]:
    method = str(cfg.method).strip().lower()
    if method not in {"auto", "lora", "partial", "full"}:
        raise ValueError(f"Unknown teacher adapt method: {cfg.method}")

    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model, "full"

    if method == "partial":
        _apply_partial_finetune(model, cfg)
        return model, "partial"

    if method == "lora":
        wrapped, status = _maybe_apply_lora(model, cfg)
        if status != "applied":
            raise RuntimeError("Requested method=lora but peft is unavailable")
        return wrapped, "lora"

    # auto
    wrapped, status = _maybe_apply_lora(model, cfg)
    if status == "applied":
        return wrapped, "lora"
    _apply_partial_finetune(model, cfg)
    return model, "partial"


@torch.no_grad()
def _evaluate_lm_loss(model, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        out = model(**inputs)
        losses.append(float(out.loss.item()))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def _extract_trainable_state(model) -> dict[str, torch.Tensor]:
    trainable_keys = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    out = {}
    for key, tensor in model.state_dict().items():
        if key in trainable_keys:
            out[key] = tensor.detach().cpu()
    return out


def train_teacher_adapter(
    *,
    fasta_path: str,
    cfg: TeacherAdaptConfig,
    out_dir: str,
    device: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    _set_seed(cfg.seed)
    _set_cpu_threads(cfg)
    device_name = detect_torch_device(device)
    logger.info("Teacher adaptation device: %s", device_name)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    sequences = read_cas12a_fasta(
        fasta_path,
        min_len=cfg.min_len,
        max_len=cfg.max_len_filter,
        deduplicate=bool(cfg.deduplicate),
    )
    train_idx, val_idx = split_train_val_indices(
        len(sequences), val_fraction=cfg.train_val_fraction, seed=cfg.seed
    )
    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    if cfg.max_train_sequences is not None:
        train_seqs = sample_sequences(
            train_seqs, n=int(cfg.max_train_sequences), seed=cfg.seed + 17
        )
    if cfg.max_val_sequences is not None:
        val_seqs = sample_sequences(
            val_seqs, n=int(cfg.max_val_sequences), seed=cfg.seed + 29
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs = {"use_fast": True}
    model_kwargs = {"use_safetensors": True}
    if cfg.base_model_revision:
        tokenizer_kwargs["revision"] = cfg.base_model_revision
        model_kwargs["revision"] = cfg.base_model_revision

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, **model_kwargs)
    except Exception:
        model_kwargs.pop("use_safetensors", None)
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, **model_kwargs)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model, method_used = _resolve_adapt_method(model, cfg)
    model.to(device_name)

    trainable_params, total_params = _count_trainable_params(model)
    logger.info(
        "Teacher adaptation method=%s trainable=%d total=%d (%.4f%%)",
        method_used,
        trainable_params,
        total_params,
        100.0 * trainable_params / max(1, total_params),
    )
    if trainable_params <= 0:
        raise RuntimeError("Teacher adaptation resolved to 0 trainable parameters.")

    train_ids = _tokenize_sequences(
        train_seqs, tokenizer=tokenizer, add_spaces=cfg.add_spaces, max_length=cfg.max_length
    )
    val_ids = _tokenize_sequences(
        val_seqs, tokenizer=tokenizer, add_spaces=cfg.add_spaces, max_length=cfg.max_length
    )
    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("Teacher adaptation dataset is empty after tokenization/filtering.")

    train_loader = DataLoader(
        _TokenizedCausalDataset(train_ids),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=partial(_collate_clm_batch, pad_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        _TokenizedCausalDataset(val_ids),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=partial(_collate_clm_batch, pad_id=tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    last_ckpt = out_path / "checkpoints" / "checkpoint_last.pt"
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    history_rows: list[dict[str, float | int]] = []
    if resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        history_rows = list(ckpt.get("history_rows", []))
        logger.info("Resumed teacher adapter from epoch=%d", start_epoch - 1)

    started_at = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train(True)
        running = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            inputs = {
                "input_ids": batch["input_ids"].to(device_name),
                "attention_mask": batch["attention_mask"].to(device_name),
                "labels": batch["labels"].to(device_name),
            }
            out = model(**inputs)
            loss = out.loss / max(1, int(cfg.grad_accum_steps))
            loss.backward()
            running.append(float(out.loss.item()))

            if step % max(1, int(cfg.grad_accum_steps)) == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % max(1, int(cfg.log_every_steps)) == 0:
                    logger.info(
                        "Epoch %d step %d global_step=%d train_loss=%.4f",
                        epoch,
                        step,
                        global_step,
                        float(np.mean(running[-10:])),
                    )

        train_loss = float(np.mean(running)) if running else float("nan")
        val_loss = _evaluate_lm_loss(model, val_loader, device_name)
        train_ppl = float(math.exp(min(20.0, train_loss))) if math.isfinite(train_loss) else float("nan")
        val_ppl = float(math.exp(min(20.0, val_loss))) if math.isfinite(val_loss) else float("nan")
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_ppl": float(train_ppl),
            "val_ppl": float(val_ppl),
        }
        history_rows.append(row)
        logger.info(
            "Teacher adapt epoch=%d/%d train_loss=%.4f val_loss=%.4f val_ppl=%.4f",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            val_ppl,
        )

        trainable_state = _extract_trainable_state(model)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "optimizer_state_dict": optimizer.state_dict(),
                "trainable_state_dict": trainable_state,
                "history_rows": history_rows,
                "method_used": method_used,
            },
            last_ckpt,
        )
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "trainable_state_dict": trainable_state,
                    "method_used": method_used,
                },
                out_path / "checkpoints" / "checkpoint_best.pt",
            )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_path / "train_history.csv", index=False)

    adapter_path = None
    adapted_model_path = None
    if method_used == "lora":
        adapter_dir = out_path / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        adapter_path = str(adapter_dir)
        if hasattr(model, "merge_and_unload"):
            merged_model = model.merge_and_unload()
            adapted_model_dir = out_path / "adapted_teacher_model"
            adapted_model_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(adapted_model_dir)
            tokenizer.save_pretrained(adapted_model_dir)
            adapted_model_path = str(adapted_model_dir)
    else:
        adapted_model_dir = out_path / "adapted_teacher_model"
        adapted_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapted_model_dir)
        tokenizer.save_pretrained(adapted_model_dir)
        adapted_model_path = str(adapted_model_dir)

    finished_at = time.time()
    summary = {
        "fasta_path": str(fasta_path),
        "out_dir": str(out_path),
        "device": device_name,
        "method_requested": cfg.method,
        "method_used": method_used,
        "base_model_name_or_path": cfg.base_model_name_or_path,
        "base_model_revision": cfg.base_model_revision,
        "adapted_model_path": adapted_model_path,
        "adapter_path": adapter_path,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "n_sequences_total": len(sequences),
        "n_sequences_train": len(train_seqs),
        "n_sequences_val": len(val_seqs),
        "n_tokenized_train": len(train_ids),
        "n_tokenized_val": len(val_ids),
        "epochs": cfg.epochs,
        "global_steps": global_step,
        "best_val_loss": float(best_val_loss),
        "best_val_ppl": float(math.exp(min(20.0, best_val_loss))),
        "runtime_seconds": float(finished_at - started_at),
        "config": asdict(cfg),
    }
    (out_path / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
