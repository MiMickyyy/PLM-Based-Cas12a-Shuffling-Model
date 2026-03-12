from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.teacher.adaptation_eval import (
    TeacherAdaptEvalConfig,
    run_teacher_adaptation_eval,
)
from cas12a_shuffling_model.teacher.scoring_utils import (
    load_validated_domains_dict,
    resolve_validated_domains_path,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--adapted-model-path", required=True)
    ap.add_argument("--active-csv", default=None)
    ap.add_argument("--natural-fasta", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--natural-eval-size", type=int, default=None)
    ap.add_argument("--background-size", type=int, default=None)
    ap.add_argument("--score-batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    e_cfg = cfg.get("teacher_adapt_eval", {})
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 13))

    active_csv = args.active_csv
    if not active_csv:
        active_dir = cfg.get("paths", {}).get("out_active_dir")
        if active_dir:
            active_csv = str(Path(active_dir) / "active_chimeras_reconstructed.csv")
    if not active_csv:
        raise SystemExit("Missing --active-csv and no paths.out_active_dir configured")

    natural_fasta = args.natural_fasta or cfg.get("paths", {}).get("atlas_fasta")
    if not natural_fasta:
        raise SystemExit("Missing --natural-fasta and no paths.atlas_fasta configured")

    out_dir = args.out_dir or e_cfg.get(
        "output_dir", "cas12a_shuffling_model/outputs/teacher_adapt_eval"
    )
    run_dir = Path(out_dir) / f"eval_{int(time.time())}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = TeacherAdaptEvalConfig(
        seed=seed,
        natural_eval_size=int(
            args.natural_eval_size
            if args.natural_eval_size is not None
            else e_cfg.get("natural_eval_size", 128)
        ),
        background_size=int(
            args.background_size
            if args.background_size is not None
            else e_cfg.get("background_size", 128)
        ),
        score_batch_size=int(
            args.score_batch_size
            if args.score_batch_size is not None
            else e_cfg.get("score_batch_size", 1)
        ),
        make_plots=not bool(args.no_plots),
    )

    vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
    validated_domains = load_validated_domains_dict(vd_path)

    report = run_teacher_adaptation_eval(
        root_config=cfg,
        adapted_model_path=args.adapted_model_path,
        validated_domains=validated_domains,
        out_dir=str(run_dir),
        cfg=eval_cfg,
        active_csv=str(active_csv),
        natural_fasta=str(natural_fasta),
        device=args.device,
    )
    (run_dir / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Teacher adaptation evaluation done. out_dir=%s", run_dir)


if __name__ == "__main__":
    main()

