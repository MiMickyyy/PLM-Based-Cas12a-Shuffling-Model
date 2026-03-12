from pathlib import Path

from cas12a_shuffling_model.teacher.protgpt2_scorer import (
    ProtGPT2Config,
    ProtGPT2Scorer,
    build_teacher_model_fingerprint,
)
from cas12a_shuffling_model.teacher.junction_scoring import JunctionWindowConfig
from cas12a_shuffling_model.teacher.score_cache import ScoreCache
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    with_teacher_overrides,
)


def test_teacher_cache_key_differs_by_model_name(tmp_path: Path):
    cache = ScoreCache(tmp_path / "scores.sqlite")
    scorer_a = ProtGPT2Scorer(
        config=ProtGPT2Config(model_name="nferruz/ProtGPT2"),
        window=JunctionWindowConfig(left=25, right=25),
        cache=cache,
    )
    scorer_b = ProtGPT2Scorer(
        config=ProtGPT2Config(model_name="example/other-model"),
        window=JunctionWindowConfig(left=25, right=25),
        cache=cache,
    )
    seq_hash = "abc123"
    assert scorer_a._cache_key(seq_hash) != scorer_b._cache_key(seq_hash)


def test_teacher_cache_key_differs_by_adapter_path(tmp_path: Path):
    cache = ScoreCache(tmp_path / "scores.sqlite")
    scorer_base = ProtGPT2Scorer(
        config=ProtGPT2Config(model_name="nferruz/ProtGPT2"),
        window=JunctionWindowConfig(left=25, right=25),
        cache=cache,
    )
    scorer_adapted = ProtGPT2Scorer(
        config=ProtGPT2Config(model_name="nferruz/ProtGPT2", adapter_path="/tmp/adapter"),
        window=JunctionWindowConfig(left=25, right=25),
        cache=cache,
    )
    seq_hash = "abc123"
    assert scorer_base._cache_key(seq_hash) != scorer_adapted._cache_key(seq_hash)


def test_local_model_fingerprint_changes_with_files(tmp_path: Path):
    model_dir = tmp_path / "teacher_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    config_path.write_text('{"model_type":"gpt2"}', encoding="utf-8")
    fp1 = build_teacher_model_fingerprint(
        ProtGPT2Config(model_source="local", model_name_or_path=str(model_dir))
    )
    config_path.write_text('{"model_type":"gpt2","n_layer":24}', encoding="utf-8")
    fp2 = build_teacher_model_fingerprint(
        ProtGPT2Config(model_source="local", model_name_or_path=str(model_dir))
    )
    assert fp1 != fp2


def test_with_teacher_overrides_and_local_config(tmp_path: Path):
    cache_path = tmp_path / "teacher_cache.sqlite"
    cfg = {
        "paths": {"out_processed_dir": str(tmp_path)},
        "teacher": {
            "model_name": "nferruz/ProtGPT2",
            "cache_sqlite": str(cache_path),
            "junction_window": {"left": 11, "right": 22},
        },
    }
    updated = with_teacher_overrides(
        cfg,
        model_name_or_path="/models/adapted_teacher",
        model_source="local",
        adapter_path="/models/adapter",
        model_revision="main",
    )
    scorer = build_teacher_scorer_from_config(updated)
    assert scorer.config.model_source == "local"
    assert scorer.config.resolved_model_name_or_path == "/models/adapted_teacher"
    assert scorer.config.resolved_adapter_path == "/models/adapter"
    assert scorer.config.model_revision == "main"
    assert scorer.window.left == 11
    assert scorer.window.right == 22
