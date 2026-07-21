#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm_ddpo.sft import (  # noqa: E402
    CountConditioningSFTTrainer,
    CountSFTConfig,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = CountSFTConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Native MDM diffusion SFT for explicit step-count conditioning."
        )
    )
    parser.add_argument("--mdm-root", default=defaults.mdm_root)
    parser.add_argument("--motionrft-root", default=defaults.motionrft_root)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--model-args-path", default=defaults.model_args_path)
    parser.add_argument(
        "--prediction-type",
        choices=["auto", "x_start", "epsilon"],
        default=defaults.prediction_type,
    )
    parser.add_argument("--data-cache-dir", default=defaults.data_cache_dir)
    parser.add_argument("--step-data-manifest", default=defaults.step_data_manifest)
    parser.add_argument("--step-motion-root", default=defaults.step_motion_root)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--resume", default="")

    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--precision",
        choices=["no", "fp16", "bf16"],
        default=defaults.precision,
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=defaults.allow_tf32,
    )
    parser.add_argument("--data-workers", type=int, default=defaults.data_workers)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=defaults.pin_memory,
    )

    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=defaults.steps_per_epoch,
        help="Optimizer updates per epoch; 0 uses one balanced step-data pass.",
    )
    parser.add_argument(
        "--human-batch-size",
        type=int,
        default=defaults.human_batch_size,
    )
    parser.add_argument(
        "--step-batch-size",
        type=int,
        default=defaults.step_batch_size,
    )
    parser.add_argument(
        "--human-loss-weight",
        type=float,
        default=defaults.human_loss_weight,
    )
    parser.add_argument(
        "--step-loss-weight",
        type=float,
        default=defaults.step_loss_weight,
    )
    parser.add_argument(
        "--lora-learning-rate",
        type=float,
        default=defaults.lora_learning_rate,
    )
    parser.add_argument(
        "--count-learning-rate",
        type=float,
        default=defaults.count_learning_rate,
    )
    parser.add_argument("--adam-beta1", type=float, default=defaults.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=defaults.adam_beta2)
    parser.add_argument(
        "--adam-weight-decay",
        type=float,
        default=defaults.adam_weight_decay,
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=defaults.adam_epsilon,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=defaults.max_grad_norm,
    )
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora-alpha", type=float, default=defaults.lora_alpha)
    parser.add_argument(
        "--lora-target-regex",
        default=defaults.lora_target_regex,
    )

    parser.add_argument("--step-targets", default=defaults.step_targets)
    parser.add_argument(
        "--step-split-seed",
        type=int,
        default=defaults.step_split_seed,
    )
    parser.add_argument(
        "--step-prompt-seed",
        type=int,
        default=defaults.step_prompt_seed,
    )
    parser.add_argument(
        "--step-eval-samples-per-target",
        type=int,
        default=defaults.step_eval_samples_per_target,
    )
    parser.add_argument(
        "--step-min-frames",
        type=int,
        default=defaults.step_min_frames,
    )
    parser.add_argument(
        "--step-max-frames",
        type=int,
        default=defaults.step_max_frames,
    )
    parser.add_argument("--length-bins", type=int, default=defaults.length_bins)
    parser.add_argument(
        "--max-samples-per-bin-target",
        type=int,
        default=defaults.max_samples_per_bin_target,
    )
    parser.add_argument(
        "--anti-jitter-lambda",
        type=float,
        default=defaults.anti_jitter_lambda,
    )
    parser.add_argument(
        "--anti-jitter-auto-grad-ratio",
        type=float,
        default=defaults.anti_jitter_auto_grad_ratio,
    )
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    config = CountSFTConfig(**vars(args))
    trainer = CountConditioningSFTTrainer(config)
    checkpoint = trainer.train()
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "global_step": trainer.global_step,
                "anti_jitter_lambda": trainer.anti_jitter_lambda_effective,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
