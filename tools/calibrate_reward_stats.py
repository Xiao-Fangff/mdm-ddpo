#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm_ddpo.calibration import (  # noqa: E402
    MIN_CALIBRATION_PROMPTS,
    MIN_CALIBRATION_SAMPLES_PER_PROMPT,
    compute_reward_calibration,
    save_reward_calibration,
)
from mdm_ddpo.config import TrainConfig  # noqa: E402
from mdm_ddpo.rewards import MotionReward  # noqa: E402
from mdm_ddpo.runtime import (  # noqa: E402
    bootstrap_external_repositories,
    build_data_loader,
    build_dataset,
    build_mdm,
    build_policy_model,
    diffusion_runtime_metadata,
    load_model_args,
    resolve_device,
    resolve_reward_device,
    seed_everything,
)
from mdm_ddpo.trainer import (  # noqa: E402
    DDPOTrainer,
    FixedEvalPool,
    load_fixed_eval_pool,
    save_fixed_eval_pool,
    validate_fixed_eval_pool,
)


LOGGER = logging.getLogger("calibrate_reward_stats")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fixed retrieval/M2M reward scales on the original MDM."
        )
    )
    parser.add_argument("--output", default="reward_calibration.json")
    parser.add_argument("--pool-path", default="")
    parser.add_argument("--samples-output", default="")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--prompts", type=int, default=1024)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--ddim-eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reward-device", default="same")
    parser.add_argument(
        "--precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--data-cache-dir", default=TrainConfig.data_cache_dir)
    parser.add_argument("--mdm-root", default=TrainConfig.mdm_root)
    parser.add_argument("--motionrft-root", default=TrainConfig.motionrft_root)
    parser.add_argument("--model-path", default=TrainConfig.model_path)
    parser.add_argument("--model-args-path", default=TrainConfig.model_args_path)
    parser.add_argument(
        "--prediction-type",
        choices=["auto", "x_start", "epsilon"],
        default="auto",
    )
    parser.add_argument(
        "--reward-backbone-path",
        default=TrainConfig.reward_backbone_path,
    )
    parser.add_argument("--reward-t5-path", default=TrainConfig.reward_t5_path)
    parser.add_argument("--retrieval-weight", type=float, default=1.0)
    parser.add_argument("--m2m-weight", type=float, default=1.0)
    parser.add_argument(
        "--allow-small-run",
        action="store_true",
        help="Allow a non-production calibration for smoke testing only.",
    )
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.samples_per_prompt < 2:
        parser.error("--samples-per-prompt must be at least 2.")
    if args.batch_size % args.samples_per_prompt != 0:
        parser.error("--batch-size must be divisible by --samples-per-prompt.")
    if args.prompts <= 0:
        parser.error("--prompts must be positive.")
    if not args.allow_small_run and (
        args.prompts < MIN_CALIBRATION_PROMPTS
        or args.samples_per_prompt < MIN_CALIBRATION_SAMPLES_PER_PROMPT
    ):
        parser.error(
            "A production calibration requires at least "
            f"{MIN_CALIBRATION_PROMPTS} prompts and "
            f"{MIN_CALIBRATION_SAMPLES_PER_PROMPT} samples per prompt."
        )


def _build_config(args: argparse.Namespace, output: Path) -> TrainConfig:
    config = TrainConfig(
        mdm_root=args.mdm_root,
        motionrft_root=args.motionrft_root,
        model_path=args.model_path,
        model_args_path=args.model_args_path,
        prediction_type=args.prediction_type,
        reward_backbone_path=args.reward_backbone_path,
        reward_t5_path=args.reward_t5_path,
        output_dir=str(output.parent),
        split="train",
        data_cache_dir=args.data_cache_dir,
        data_workers=args.data_workers,
        seed=args.seed,
        device=args.device,
        reward_device=args.reward_device,
        precision=args.precision,
        sample_steps=args.sample_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        rollout_batch_size=args.batch_size,
        rollout_batches_per_epoch=1,
        samples_per_prompt=args.samples_per_prompt,
        train_batch_size=args.batch_size,
        retrieval_weight=args.retrieval_weight,
        m2m_weight=args.m2m_weight,
        reward_embedding_mode="mean",
        fixed_eval_every=0,
        fixed_eval_prompts=args.prompts,
        fixed_eval_samples_per_prompt=args.samples_per_prompt,
        use_swanlab=False,
    )
    config.validate()
    return config


def _load_or_create_pool(
    config: TrainConfig,
    *,
    split: str,
    prompts: int,
    seed: int,
    path: Path,
) -> FixedEvalPool:
    if path.exists():
        pool = load_fixed_eval_pool(path)
    else:
        seed_everything(seed)
        dataset = build_dataset(config, split=split)
        if len(dataset) < prompts:
            raise ValueError(
                f"Calibration split {split!r} has {len(dataset)} samples, "
                f"fewer than requested prompts={prompts}."
            )
        generator = torch.Generator().manual_seed(seed)
        dataset_indices = torch.randperm(
            len(dataset),
            generator=generator,
        )[:prompts]
        from data_loaders.get_data import get_collate_fn

        items = [dataset[int(index)] for index in dataset_indices]
        collate_fn = get_collate_fn(
            config.dataset,
            hml_mode="train",
            batch_size=prompts,
        )
        motion, condition = collate_fn(items)
        pool = validate_fixed_eval_pool(
            FixedEvalPool(
                dataset_indices=dataset_indices,
                motion=motion,
                lengths=condition["y"]["lengths"],
                texts=list(condition["y"]["text"]),
                split=split,
                noise_seed=seed,
                prompt_noise_seeds=(
                    torch.arange(prompts, dtype=torch.long) * 1_000_003
                    + seed
                ),
            )
        )
        save_fixed_eval_pool(pool, path)
        LOGGER.info("Created calibration prompt pool: %s", path)
    actual = (pool.split, pool.prompt_count, pool.noise_seed)
    expected = (split, prompts, seed)
    if actual != expected:
        raise ValueError(
            "Existing calibration pool does not match requested settings: "
            f"expected={expected}, actual={actual}."
        )
    return pool


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    output = Path(args.output).expanduser().resolve()
    pool_path = (
        Path(args.pool_path).expanduser().resolve()
        if args.pool_path
        else output.with_name("reward_calibration_pool.pt")
    )
    samples_output = (
        Path(args.samples_output).expanduser().resolve()
        if args.samples_output
        else output.with_name("reward_calibration_samples.pt")
    )
    config = _build_config(args, output)
    bootstrap_external_repositories(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    reward_device = resolve_reward_device(config, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32

    data_loader = build_data_loader(config)
    model_args = load_model_args(config)
    model, diffusion, _, sample_steps = build_mdm(
        config,
        model_args,
        data_loader,
        device,
    )
    policy_model = build_policy_model(model, config.guidance_scale)
    reward_model = MotionReward(config, reward_device)
    pool = _load_or_create_pool(
        config,
        split=args.split,
        prompts=args.prompts,
        seed=args.seed,
        path=pool_path,
    )

    evaluator = DDPOTrainer.__new__(DDPOTrainer)
    evaluator.config = config
    evaluator.device = device
    evaluator.model = model
    evaluator.diffusion = diffusion
    evaluator.policy_model = policy_model
    evaluator.reward_model = reward_model
    evaluator.fixed_eval_pool = pool
    evaluation = evaluator.evaluate_fixed_pool()
    if (
        evaluation.retrieval_by_prompt is None
        or evaluation.m2m_by_prompt is None
    ):
        raise RuntimeError("Calibration evaluation did not retain per-sample scores.")

    metadata = {
        "model_path": str(Path(config.model_path).expanduser().resolve()),
        "model_args_path": str(
            Path(config.model_args_path).expanduser().resolve()
        ),
        "policy": "original_mdm_without_lora",
        "dataset": config.dataset,
        "split": pool.split,
        "pool_id": pool.pool_id,
        "pool_path": str(pool_path),
        "seed": args.seed,
        "sample_steps": sample_steps,
        "guidance_scale": config.guidance_scale,
        "ddim_eta": config.ddim_eta,
        "precision": config.precision,
        "reward_embedding_mode": "mean",
        "mdm_diffusion": diffusion_runtime_metadata(model_args, diffusion),
    }
    payload = compute_reward_calibration(
        evaluation.retrieval_by_prompt,
        evaluation.m2m_by_prompt,
        retrieval_weight=config.retrieval_weight,
        m2m_weight=config.m2m_weight,
        metadata=metadata,
    )
    save_reward_calibration(payload, output)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "calibration_id": payload["calibration_id"],
            "pool_id": pool.pool_id,
            "retrieval": evaluation.retrieval_by_prompt,
            "m2m": evaluation.m2m_by_prompt,
        },
        samples_output,
    )
    LOGGER.info("Saved reward calibration: %s", output)
    LOGGER.info("Saved raw reward samples: %s", samples_output)
    print(
        json.dumps(
            {
                "calibration_id": payload["calibration_id"],
                "full_calibration": payload["full_calibration"],
                "components": payload["components"],
                "relationships": payload["relationships"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
