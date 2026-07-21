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

from mdm_ddpo.config import TrainConfig  # noqa: E402
from mdm_ddpo.count_conditioning import count_conditioning_signature  # noqa: E402
from mdm_ddpo.policy_io import (  # noqa: E402
    configure_and_load_policy_checkpoint,
    load_policy_checkpoint_payload,
    policy_uses_count_conditioning,
)
from mdm_ddpo.rewards import RewardOutput  # noqa: E402
from mdm_ddpo.runtime import (  # noqa: E402
    bootstrap_external_repositories,
    build_data_loader,
    build_mdm,
    build_policy_model,
    diffusion_runtime_metadata,
    load_model_args,
    resolve_device,
    seed_everything,
)
from mdm_ddpo.step_calibration import (  # noqa: E402
    MIN_STEP_CALIBRATION_PROMPTS,
    MIN_STEP_CALIBRATION_SAMPLES_PER_PROMPT,
    compute_step_reward_calibration,
    compute_target_error_scales,
    save_step_reward_calibration,
)
from mdm_ddpo.step_data import (  # noqa: E402
    create_fixed_step_eval_pool,
    create_synthetic_fixed_step_eval_pool,
    create_synthetic_step_records,
    load_fixed_step_eval_pool,
    load_humanml_stats,
    load_step_manifest,
    save_fixed_step_eval_pool,
    stratified_step_split,
)
from mdm_ddpo.step_reward import (  # noqa: E402
    HardStepDetector,
    compute_step_count_reward,
)
from mdm_ddpo.trainer import DDPOTrainer  # noqa: E402


LOGGER = logging.getLogger("calibrate_step_reward_stats")


class _ZeroMotionReward:
    """Avoid loading MotionReward when calibrating only hard step counts."""

    embedding_mode = "mean"

    @staticmethod
    def score(
        *,
        texts: list[str],
        generated_motion: torch.Tensor,
        lengths: torch.Tensor,
        gt_motion: torch.Tensor,
    ) -> RewardOutput:
        del texts, lengths, gt_motion
        zeros = torch.zeros(
            generated_motion.shape[0],
            device=generated_motion.device,
        )
        return RewardOutput(total=zeros, retrieval=zeros, m2m=zeros)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate fixed hard/soft step reward scales on the original MDM."
        )
    )
    parser.add_argument("--output", default="step_reward_k16_calibration.json")
    parser.add_argument("--pool-path", default="")
    parser.add_argument("--samples-output", default="")
    parser.add_argument("--prompts", type=int, default=384)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--ddim-eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--split-seed", type=int, default=20260600)
    parser.add_argument("--prompt-seed", type=int, default=20260612)
    parser.add_argument("--step-targets", default="1,2,3,4,5,6")
    parser.add_argument(
        "--step-pool-source",
        choices=["reference", "synthetic"],
        default="reference",
    )
    parser.add_argument("--step-data-manifest", default=TrainConfig.step_data_manifest)
    parser.add_argument("--step-motion-root", default=TrainConfig.step_motion_root)
    parser.add_argument("--step-detector-root", default=TrainConfig.step_detector_root)
    parser.add_argument(
        "--step-detector-backend",
        choices=["progressive", "rgdno"],
        default="progressive",
    )
    parser.add_argument("--step-detector-fps", type=int, default=20)
    parser.add_argument("--step-detector-lead-threshold", type=float, default=0.138)
    parser.add_argument("--step-detector-rgdno-threshold", type=float, default=0.005)
    parser.add_argument("--step-soft-lead-temperature", type=float, default=1.0)
    parser.add_argument("--step-soft-length-temperature", type=float, default=1.0)
    parser.add_argument("--step-soft-progress-temperature", type=float, default=1.0)
    parser.add_argument("--step-soft-cluster-gap-seconds", type=float, default=0.15)
    parser.add_argument(
        "--step-ankle-high-frequency-cutoff-hz",
        type=float,
        default=4.0,
    )
    parser.add_argument("--step-soft-huber-delta", type=float, default=1.0)
    parser.add_argument("--step-soft-exact-bonus", type=float, default=0.15)
    parser.add_argument("--step-soft-target-scale-floor", type=float, default=0.25)
    parser.add_argument(
        "--step-reward-mode",
        choices=[
            "exp",
            "linear",
            "exact",
            "negative_l1",
            "soft_huber_exact",
        ],
        default="exp",
    )
    parser.add_argument("--step-reward-temperature", type=float, default=1.0)
    parser.add_argument("--step-reward-linear-tolerance", type=float, default=3.0)
    parser.add_argument("--step-min-frames", type=int, default=40)
    parser.add_argument("--step-max-frames", type=int, default=196)
    parser.add_argument("--device", default="cuda:0")
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
        "--policy-checkpoint",
        default="",
        help=(
            "Optional count-SFT/DDPO policy checkpoint to calibrate instead "
            "of the zero-LoRA base MDM."
        ),
    )
    parser.add_argument(
        "--prediction-type",
        choices=["auto", "x_start", "epsilon"],
        default="auto",
    )
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
    targets = TrainConfig(step_targets=args.step_targets).step_target_values
    if args.prompts <= 0 or args.prompts % len(targets) != 0:
        parser.error(
            "--prompts must be positive and divisible by the number of "
            "--step-targets for stratified calibration."
        )
    if args.samples_per_prompt < 2:
        parser.error("--samples-per-prompt must be at least 2.")
    if args.batch_size % args.samples_per_prompt != 0:
        parser.error("--batch-size must be divisible by --samples-per-prompt.")
    step_samples = args.batch_size * 0.25
    if not step_samples.is_integer():
        parser.error(
            "--batch-size must make the fixed 25% step sample allocation an "
            "integer."
        )
    if int(step_samples) % args.samples_per_prompt != 0:
        parser.error(
            "At --step-data-ratio=0.25, the step sample allocation in "
            "--batch-size must be divisible by --samples-per-prompt. "
            "For K=16 use --batch-size 64 (or another compatible size)."
        )
    if (args.batch_size - int(step_samples)) % 4 != 0:
        parser.error(
            "The remaining HumanML allocation must be divisible by its fixed K=4."
        )
    if not args.allow_small_run and (
        args.prompts < MIN_STEP_CALIBRATION_PROMPTS
        or args.samples_per_prompt < MIN_STEP_CALIBRATION_SAMPLES_PER_PROMPT
    ):
        parser.error(
            "A production step calibration requires at least "
            f"{MIN_STEP_CALIBRATION_PROMPTS} prompts and "
            f"{MIN_STEP_CALIBRATION_SAMPLES_PER_PROMPT} samples per prompt."
        )


def _build_config(
    args: argparse.Namespace,
    output: Path,
    *,
    enable_count_conditioning: bool,
) -> TrainConfig:
    config = TrainConfig(
        mdm_root=args.mdm_root,
        motionrft_root=args.motionrft_root,
        model_path=args.model_path,
        model_args_path=args.model_args_path,
        prediction_type=args.prediction_type,
        enable_count_conditioning=enable_count_conditioning,
        train_count_conditioning=enable_count_conditioning,
        output_dir=str(output.parent),
        data_cache_dir=args.data_cache_dir,
        data_workers=args.data_workers,
        seed=args.seed,
        device=args.device,
        reward_device="same",
        precision=args.precision,
        sample_steps=args.sample_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        rollout_batch_size=args.batch_size,
        rollout_batches_per_epoch=1,
        samples_per_prompt=4,
        step_samples_per_prompt=args.samples_per_prompt,
        train_batch_size=args.batch_size,
        advantage_mode="group_centered",
        fixed_eval_every=0,
        fixed_step_eval_samples_per_prompt=args.samples_per_prompt,
        enable_step_reward=True,
        step_data_ratio=0.25,
        step_rollout_source=args.step_pool_source,
        step_use_m2m_reward=False,
        step_targets=args.step_targets,
        step_split_seed=args.split_seed,
        step_prompt_seed=args.prompt_seed,
        step_eval_samples_per_target=(
            args.prompts // len(TrainConfig(step_targets=args.step_targets).step_target_values)
        ),
        step_min_frames=args.step_min_frames,
        step_max_frames=args.step_max_frames,
        step_data_manifest=args.step_data_manifest,
        step_motion_root=args.step_motion_root,
        step_detector_root=args.step_detector_root,
        step_detector_backend=args.step_detector_backend,
        step_detector_fps=args.step_detector_fps,
        step_detector_lead_threshold=args.step_detector_lead_threshold,
        step_detector_rgdno_threshold=args.step_detector_rgdno_threshold,
        step_soft_lead_temperature=args.step_soft_lead_temperature,
        step_soft_length_temperature=args.step_soft_length_temperature,
        step_soft_progress_temperature=args.step_soft_progress_temperature,
        step_soft_cluster_gap_seconds=args.step_soft_cluster_gap_seconds,
        step_ankle_high_frequency_cutoff_hz=(
            args.step_ankle_high_frequency_cutoff_hz
        ),
        step_soft_huber_delta=args.step_soft_huber_delta,
        step_soft_exact_bonus=args.step_soft_exact_bonus,
        step_soft_target_scale_floor=args.step_soft_target_scale_floor,
        step_reward_mode=args.step_reward_mode,
        step_reward_temperature=args.step_reward_temperature,
        step_reward_linear_tolerance=args.step_reward_linear_tolerance,
        allow_uncalibrated_soft_step_reward=True,
    )
    config.validate()
    return config


def _load_or_create_pool(
    config: TrainConfig,
    *,
    prompts: int,
    path: Path,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    mean, std = load_humanml_stats(config.mdm_root)
    if path.exists():
        pool = load_fixed_step_eval_pool(path)
    else:
        records = load_step_manifest(
            config.step_data_manifest,
            motion_root=config.step_motion_root,
            targets=config.step_target_values,
            min_frames=config.step_min_frames,
            max_frames=config.step_max_frames,
        )
        training, selected = stratified_step_split(
            records,
            eval_per_target=prompts // len(config.step_target_values),
            split_seed=config.step_split_seed,
            prompt_seed=config.step_prompt_seed,
        )
        if config.step_rollout_source == "synthetic":
            synthetic, _ = create_synthetic_step_records(
                training,
                targets=config.step_target_values,
                seed=config.step_synthetic_seed,
                samples_per_target=prompts // len(config.step_target_values),
            )
            pool = create_synthetic_fixed_step_eval_pool(
                synthetic,
                max_frames=config.step_max_frames,
                noise_seed=config.seed + 104729,
                detector_backend=config.step_detector_backend,
            )
        else:
            pool = create_fixed_step_eval_pool(
                selected,
                mean=mean,
                std=std,
                max_frames=config.step_max_frames,
                noise_seed=config.seed + 104729,
                detector_backend=config.step_detector_backend,
            )
        save_fixed_step_eval_pool(pool, path)
        LOGGER.info("Created step calibration pool: %s", path)
    expected = (
        prompts,
        config.seed + 104729,
        config.step_detector_backend,
        "synthetic" if config.step_rollout_source == "synthetic" else "val",
    )
    actual = (
        pool.prompt_count,
        pool.noise_seed,
        pool.detector_backend,
        pool.split,
    )
    if actual != expected:
        raise ValueError(
            "Existing step calibration pool does not match requested settings: "
            f"expected={expected}, actual={actual}."
        )
    return pool, torch.from_numpy(mean), torch.from_numpy(std)


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
        else output.with_name("step_reward_calibration_pool.pt")
    )
    samples_output = (
        Path(args.samples_output).expanduser().resolve()
        if args.samples_output
        else output.with_name("step_reward_calibration_samples.pt")
    )
    policy_payload = (
        load_policy_checkpoint_payload(args.policy_checkpoint)
        if args.policy_checkpoint
        else None
    )
    config = _build_config(
        args,
        output,
        enable_count_conditioning=(
            policy_uses_count_conditioning(policy_payload)
            if policy_payload is not None
            else False
        ),
    )
    bootstrap_external_repositories(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32

    data_loader = build_data_loader(
        config,
        prompt_batch_size=config.humanml_prompts_per_rollout_batch,
    )
    model_args = load_model_args(config)
    model, diffusion, _, sample_steps = build_mdm(
        config,
        model_args,
        data_loader,
        device,
    )
    runtime_diffusion = diffusion_runtime_metadata(model_args, diffusion)
    policy_structure: dict[str, object] | None = None
    if policy_payload is not None:
        policy_structure = configure_and_load_policy_checkpoint(
            model,
            policy_payload,
            diffusion_metadata=runtime_diffusion,
            model_path=config.model_path,
            source="Step calibration policy",
        )
    policy_model = build_policy_model(model, config.guidance_scale)
    pool, mean, std = _load_or_create_pool(
        config,
        prompts=args.prompts,
        path=pool_path,
    )
    evaluator = DDPOTrainer.__new__(DDPOTrainer)
    evaluator.config = config
    evaluator.device = device
    evaluator.model = model
    evaluator.diffusion = diffusion
    evaluator.policy_model = policy_model
    evaluator.reward_model = _ZeroMotionReward()
    evaluator.step_detector = HardStepDetector(
        backend=config.step_detector_backend,
        fps=config.step_detector_fps,
        motion_rule_root=config.step_detector_root,
        lead_threshold=config.step_detector_lead_threshold,
        rgdno_threshold=config.step_detector_rgdno_threshold,
        soft_lead_temperature=config.step_soft_lead_temperature,
        soft_length_temperature=config.step_soft_length_temperature,
        soft_progress_temperature=config.step_soft_progress_temperature,
        soft_cluster_gap_seconds=config.step_soft_cluster_gap_seconds,
        ankle_high_frequency_cutoff_hz=(
            config.step_ankle_high_frequency_cutoff_hz
        ),
    )
    evaluator.step_mdm_mean = mean.to(device=device, dtype=torch.float32)
    evaluator.step_mdm_std = std.to(device=device, dtype=torch.float32)
    evaluator.fixed_step_eval_pool = pool
    evaluation = evaluator.evaluate_fixed_step_pool(calibrating=True)
    if (
        evaluation.step_reward_by_prompt is None
        or evaluation.detected_steps_by_prompt is None
    ):
        raise RuntimeError("Step calibration did not retain per-sample values.")
    targets = pool.target_steps[:, None].expand_as(
        evaluation.detected_steps_by_prompt
    )
    if evaluation.soft_count_by_prompt is None:
        raise RuntimeError("Step calibration did not retain soft-count values.")
    target_error_scales = compute_target_error_scales(
        evaluation.soft_count_by_prompt,
        targets,
        minimum_scale=config.step_soft_target_scale_floor,
    )
    calibrated_reward = evaluation.step_reward_by_prompt
    if config.step_reward_mode == "soft_huber_exact":
        target_scale = torch.tensor(
            [
                target_error_scales[str(int(value))]
                for value in targets.reshape(-1)
            ],
            dtype=torch.float32,
        ).reshape_as(targets)
        calibrated_reward = compute_step_count_reward(
            evaluation.detected_steps_by_prompt,
            targets,
            mode=config.step_reward_mode,
            soft_count=evaluation.soft_count_by_prompt,
            target_scale=target_scale,
            huber_delta=config.step_soft_huber_delta,
            exact_bonus=config.step_soft_exact_bonus,
        ).reward.reshape_as(targets)
    metadata = {
        "model_path": str(Path(config.model_path).expanduser().resolve()),
        "policy": (
            "loaded_trainable_policy"
            if policy_payload is not None
            else "original_mdm_without_lora"
        ),
        "policy_checkpoint": (
            str(Path(args.policy_checkpoint).expanduser().resolve())
            if args.policy_checkpoint
            else ""
        ),
        "policy_structure": policy_structure,
        "policy_id": (
            str(policy_structure["policy_id"])
            if policy_structure is not None
            else ""
        ),
        "count_conditioning": count_conditioning_signature(model),
        "pool_id": pool.pool_id,
        "pool_path": str(pool_path),
        "seed": config.seed,
        "sample_steps": sample_steps,
        "guidance_scale": config.guidance_scale,
        "ddim_eta": config.ddim_eta,
        "precision": config.precision,
        "step_targets": list(config.step_target_values),
        "step_samples_per_prompt": config.step_samples_per_prompt,
        "step_pool_source": config.step_rollout_source,
        "mdm_diffusion": runtime_diffusion,
    }
    payload = compute_step_reward_calibration(
        calibrated_reward,
        evaluation.detected_steps_by_prompt,
        targets,
        detector_config=config.step_detector_config(),
        reward_config=config.step_reward_config(),
        metadata=metadata,
        soft_counts=evaluation.soft_count_by_prompt,
        target_error_scales=target_error_scales,
        detection_diagnostics={
            "candidate_count": evaluation.candidate_count_by_prompt,
            "candidate_spacing": evaluation.candidate_spacing_by_prompt,
            "ankle_high_frequency_ratio": (
                evaluation.ankle_high_frequency_ratio_by_prompt
            ),
        },
    )
    save_step_reward_calibration(payload, output)
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "calibration_id": payload["calibration_id"],
            "pool_id": pool.pool_id,
            "step_reward": calibrated_reward,
            "detected_steps": evaluation.detected_steps_by_prompt,
            "soft_count": evaluation.soft_count_by_prompt,
            "target_steps": targets,
        },
        samples_output,
    )
    LOGGER.info("Saved step reward calibration: %s", output)
    LOGGER.info("Saved raw step reward samples: %s", samples_output)
    print(
        json.dumps(
            {
                "calibration_id": payload["calibration_id"],
                "full_calibration": payload["full_calibration"],
                "component": payload["component"],
                "metrics": payload["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
