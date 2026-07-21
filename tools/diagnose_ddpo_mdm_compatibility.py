#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm_ddpo.config import TrainConfig  # noqa: E402
from mdm_ddpo.diffusion import (  # noqa: E402
    ddim_step_with_logprob,
    ddim_transition_mean_std,
)
from mdm_ddpo.policy_diagnostics import (  # noqa: E402
    TimestepBucket,
    advantage_logprob_alignment,
    epsilon_ddim_score_sensitivity,
    effective_lora_delta_norm,
    parse_timestep_buckets,
    summarize_sensitivity_buckets,
    summarize_timestep_log_ratios,
    xstart_ddim_score_sensitivity,
)
from mdm_ddpo.rewards import (  # noqa: E402
    apply_step_m2m_policy,
)
from mdm_ddpo.runtime import (  # noqa: E402
    autocast_context,
    build_model_kwargs,
    diffusion_prediction_type,
    diffusion_runtime_metadata,
)
from mdm_ddpo.step_reward import compute_step_count_reward  # noqa: E402
from mdm_ddpo.trainer import DDPOTrainer, Trajectory  # noqa: E402


LOGGER = logging.getLogger("diagnose_ddpo_mdm_compatibility")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit parameterization-specific timestep scaling and overfit one fixed DDPO "
            "trajectory under exactly replayed diffusion noise."
        )
    )
    parser.add_argument(
        "--output",
        default="reports/ddpo_mdm_compatibility_diagnostic.json",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/mdm-ddpo-compatibility-diagnostic",
    )
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--clip-range", type=float, default=1.0e-3)
    parser.add_argument("--gradient-pairs-per-bucket", type=int, default=32)
    parser.add_argument(
        "--step-only-overfit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Zero HumanML advantages during the fixed-trajectory updates so "
            "the test isolates step-count credit assignment."
        ),
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the full JSON report instead of a compact result summary.",
    )
    parser.add_argument(
        "--timestep-buckets",
        default="1-2,3-5,6-15,16-30,31-49",
    )
    parser.add_argument(
        "--reward-calibration-path",
        default=str(PROJECT_ROOT / "reward_calibration.json"),
    )
    parser.add_argument(
        "--step-reward-calibration-path",
        default=str(PROJECT_ROOT / "step_reward_k8_soft_huber_calibration.json"),
    )
    parser.add_argument("--mdm-root", default=TrainConfig.mdm_root)
    parser.add_argument("--motionrft-root", default=TrainConfig.motionrft_root)
    parser.add_argument("--model-path", default=TrainConfig.model_path)
    parser.add_argument("--model-args-path", default=TrainConfig.model_args_path)
    parser.add_argument(
        "--prediction-type",
        choices=["auto", "x_start", "epsilon"],
        default="auto",
    )
    parser.add_argument("--data-cache-dir", default=TrainConfig.data_cache_dir)
    return parser


def _build_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(
        mdm_root=args.mdm_root,
        motionrft_root=args.motionrft_root,
        model_path=args.model_path,
        model_args_path=args.model_args_path,
        prediction_type=args.prediction_type,
        data_cache_dir=args.data_cache_dir,
        output_dir=str(Path(args.work_dir).expanduser().resolve()),
        data_workers=0,
        pin_memory=True,
        seed=args.seed,
        device=args.device,
        reward_device="same",
        precision=args.precision,
        epochs=1,
        sample_steps=args.sample_steps,
        rollout_batch_size=args.rollout_batch_size,
        rollout_batches_per_epoch=1,
        samples_per_prompt=4,
        step_samples_per_prompt=8,
        train_batch_size=args.train_batch_size,
        inner_epochs=1,
        timestep_fraction=1.0,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        advantage_mode="component_shrink",
        advantage_std_floor_quantile="p25",
        advantage_retrieval_weight=0.5,
        advantage_m2m_weight=0.5,
        step_advantage_retrieval_weight=0.0,
        step_advantage_m2m_weight=0.0,
        step_advantage_step_weight=1.0,
        reward_calibration_path=str(
            Path(args.reward_calibration_path).expanduser().resolve()
        ),
        enable_step_reward=True,
        step_data_ratio=0.5,
        step_rollout_source="synthetic",
        step_balanced_sampling=True,
        step_use_m2m_reward=False,
        step_reward_weight=1.0,
        step_reward_mode="soft_huber_exact",
        step_reward_calibration_path=str(
            Path(args.step_reward_calibration_path).expanduser().resolve()
        ),
        fixed_eval_every=0,
        early_stop_patience=0,
        use_swanlab=False,
    )
    config.validate()
    return config


def _model_kwargs(
    trainer: DDPOTrainer,
    trajectory: Trajectory,
    sample_indices: torch.Tensor | None = None,
) -> dict[str, dict[str, Any]]:
    if sample_indices is None:
        sample_indices = torch.arange(len(trajectory.texts))
    indices = sample_indices.tolist()
    return build_model_kwargs(
        trainer.model,
        [trajectory.texts[index] for index in indices],
        trajectory.lengths[sample_indices],
        trajectory.latents.shape[-1],
        device=trainer.device,
        guidance_scale=trainer.config.guidance_scale,
        cached_text_embeddings=[
            trajectory.text_embeddings[index] for index in indices
        ],
        target_steps=(
            trajectory.target_steps[sample_indices]
            if trajectory.target_steps is not None
            else None
        ),
    )


@torch.no_grad()
def _recompute_log_probs(
    trainer: DDPOTrainer,
    trajectory: Trajectory,
) -> torch.Tensor:
    model_kwargs = _model_kwargs(trainer, trajectory)
    parts: list[torch.Tensor] = []
    for position in range(trajectory.timesteps.shape[1]):
        current = trajectory.latents[:, position].to(trainer.device)
        previous = trajectory.next_latents[:, position].to(trainer.device)
        timestep = trajectory.timesteps[:, position].to(trainer.device)
        with autocast_context(trainer.device, trainer.config.precision):
            _, log_prob, _ = ddim_step_with_logprob(
                trainer.diffusion,
                trainer.policy_model,
                current,
                timestep,
                model_kwargs=model_kwargs,
                eta=trainer.config.ddim_eta,
                prev_sample=previous,
                mask=model_kwargs["y"]["mask"],
                clip_denoised=trainer.config.clip_denoised,
            )
        parts.append(log_prob.detach().float().cpu())
    return torch.stack(parts, dim=1)


@torch.no_grad()
def _extract_transition_noise(
    trainer: DDPOTrainer,
    trajectory: Trajectory,
) -> tuple[list[torch.Tensor], float]:
    model_kwargs = _model_kwargs(trainer, trajectory)
    noise_parts: list[torch.Tensor] = []
    maximum_error = 0.0
    for position in range(trajectory.timesteps.shape[1]):
        current = trajectory.latents[:, position].to(trainer.device)
        previous = trajectory.next_latents[:, position].to(trainer.device)
        timestep = trajectory.timesteps[:, position].to(trainer.device)
        with autocast_context(trainer.device, trainer.config.precision):
            mean, std, _ = ddim_transition_mean_std(
                trainer.diffusion,
                trainer.policy_model,
                current,
                timestep,
                model_kwargs=model_kwargs,
                eta=trainer.config.ddim_eta,
                clip_denoised=trainer.config.clip_denoised,
            )
        noise = (previous - mean).div(std.clamp_min(1.0e-12))
        reconstructed = mean + std * noise
        maximum_error = max(
            maximum_error,
            reconstructed.sub(previous).abs().max().float().item(),
        )
        noise_parts.append(noise.detach().float().cpu())
    return noise_parts, maximum_error


@torch.no_grad()
def _fixed_noise_generation(
    trainer: DDPOTrainer,
    trajectory: Trajectory,
    transition_noise: list[torch.Tensor],
) -> torch.Tensor:
    model_kwargs = _model_kwargs(trainer, trajectory)
    current = trajectory.latents[:, 0].to(trainer.device)
    for position, noise in enumerate(transition_noise):
        timestep = trajectory.timesteps[:, position].to(trainer.device)
        with autocast_context(trainer.device, trainer.config.precision):
            current, _, _ = ddim_step_with_logprob(
                trainer.diffusion,
                trainer.policy_model,
                current,
                timestep,
                model_kwargs=model_kwargs,
                eta=trainer.config.ddim_eta,
                mask=model_kwargs["y"]["mask"],
                clip_denoised=trainer.config.clip_denoised,
                noise=noise.to(trainer.device),
            )
    final_timestep = torch.zeros(
        current.shape[0],
        device=trainer.device,
        dtype=torch.long,
    )
    with autocast_context(trainer.device, trainer.config.precision):
        current, _, _ = ddim_step_with_logprob(
            trainer.diffusion,
            trainer.policy_model,
            current,
            final_timestep,
            model_kwargs=model_kwargs,
            eta=trainer.config.ddim_eta,
            mask=model_kwargs["y"]["mask"],
            clip_denoised=trainer.config.clip_denoised,
            noise=torch.zeros_like(current),
        )
    return current.squeeze(2).permute(0, 2, 1).contiguous()


@torch.no_grad()
def _score_fixed_generation(
    trainer: DDPOTrainer,
    trajectory: Trajectory,
    generated_motion: torch.Tensor,
) -> dict[str, float]:
    base = trainer.reward_model.score(
        texts=trajectory.texts,
        generated_motion=generated_motion,
        lengths=trajectory.lengths,
        gt_motion=trajectory.gt_motion.to(trainer.device),
    )
    assert trajectory.step_mask is not None
    assert trajectory.target_steps is not None
    base = apply_step_m2m_policy(
        base,
        step_mask=trajectory.step_mask.to(trainer.device),
        m2m_weight=trainer.config.m2m_weight,
        enabled=trainer.config.step_use_m2m_reward,
    )
    step_mask = trajectory.step_mask.bool()
    active = step_mask.to(trainer.device)
    step_reward = torch.zeros(len(step_mask), device=trainer.device)
    detected = torch.full(
        (len(step_mask),),
        -1,
        device=trainer.device,
        dtype=torch.long,
    )
    soft_count = torch.full_like(detected, -1, dtype=torch.float32)
    if active.any():
        assert trainer.step_detector is not None
        assert trainer.step_mdm_mean is not None
        assert trainer.step_mdm_std is not None
        assert trainer.step_reward_calibration is not None
        detection = trainer.step_detector.detect_normalized(
            generated_motion[active],
            trajectory.lengths[step_mask],
            mean=trainer.step_mdm_mean,
            std=trainer.step_mdm_std,
        )
        targets = trajectory.target_steps[step_mask].to(trainer.device)
        output = compute_step_count_reward(
            detection.hard_count,
            targets,
            mode=trainer.config.step_reward_mode,
            temperature=trainer.config.step_reward_temperature,
            linear_tolerance=trainer.config.step_reward_linear_tolerance,
            soft_count=detection.soft_count,
            target_scale=trainer.step_reward_calibration.target_error_scales(
                targets
            ),
            huber_delta=trainer.config.step_soft_huber_delta,
            exact_bonus=trainer.config.step_soft_exact_bonus,
        )
        step_reward[active] = output.reward
        detected[active] = output.detected_steps
        soft_count[active] = detection.soft_count
    absolute_error = (
        detected[active] - trajectory.target_steps[step_mask].to(trainer.device)
    ).abs().float()
    total = base.total + trainer.config.step_reward_weight * step_reward
    metrics = {
        "reward_total": total.mean().item(),
        "reward_retrieval": base.retrieval.mean().item(),
        "reward_m2m": base.m2m.mean().item(),
        "step_reward": step_reward[active].mean().item(),
        "step_soft_count_mean": soft_count[active].mean().item(),
        "step_hard_count_mean": detected[active].float().mean().item(),
        "step_mae": absolute_error.mean().item(),
        "step_exact_fraction": (absolute_error == 0).float().mean().item(),
        "step_within_one_fraction": (
            absolute_error <= 1
        ).float().mean().item(),
        "step_target_mean": (
            trajectory.target_steps[step_mask].float().mean().item()
        ),
    }
    return metrics


def _gradient_l2_norm(gradients: list[torch.Tensor | None]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared += gradient.detach().double().square().sum().cpu()
    return squared.sqrt().item()


def _bucket_gradient_diagnostics(
    trainer: DDPOTrainer,
    trajectory: Trajectory,
    buckets: tuple[TimestepBucket, ...],
    *,
    pairs_per_bucket: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if trajectory.advantages is None:
        raise ValueError("Trajectory advantages are required.")
    if pairs_per_bucket <= 0:
        raise ValueError("Gradient pairs per bucket must be positive.")
    trainable = [
        parameter for parameter in trainer.model.parameters()
        if parameter.requires_grad
    ]
    generator = torch.Generator().manual_seed(seed)
    output: dict[str, dict[str, float]] = {}
    flat_timesteps = trajectory.timesteps.reshape(-1)
    transition_count = trajectory.timesteps.shape[1]
    for bucket in buckets:
        candidates = torch.nonzero(
            bucket.contains(flat_timesteps),
            as_tuple=False,
        ).reshape(-1)
        if not len(candidates):
            continue
        order = torch.randperm(len(candidates), generator=generator)
        selected = candidates[order[: min(len(candidates), pairs_per_bucket)]]
        sample_indices = torch.div(
            selected,
            transition_count,
            rounding_mode="floor",
        )
        time_indices = selected.remainder(transition_count)
        model_kwargs = _model_kwargs(trainer, trajectory, sample_indices)
        current = trajectory.latents[sample_indices, time_indices].to(
            trainer.device
        )
        previous = trajectory.next_latents[sample_indices, time_indices].to(
            trainer.device
        )
        timesteps = trajectory.timesteps[sample_indices, time_indices].to(
            trainer.device
        )
        old_log_probs = trajectory.old_log_probs[
            sample_indices,
            time_indices,
        ].to(trainer.device)
        advantages = trajectory.advantages[sample_indices].to(trainer.device)
        with autocast_context(trainer.device, trainer.config.precision):
            _, new_log_probs, _ = ddim_step_with_logprob(
                trainer.diffusion,
                trainer.policy_model,
                current,
                timesteps,
                model_kwargs=model_kwargs,
                eta=trainer.config.ddim_eta,
                prev_sample=previous,
                mask=model_kwargs["y"]["mask"],
                clip_denoised=trainer.config.clip_denoised,
            )
            log_ratio = new_log_probs - old_log_probs
            loss = -(advantages * log_ratio.clamp(-20.0, 20.0).exp()).mean()
        gradients = list(
            torch.autograd.grad(
                loss,
                trainable,
                allow_unused=True,
            )
        )
        output[bucket.label] = {
            "pairs": float(len(selected)),
            "advantage_mean": advantages.detach().float().mean().item(),
            "advantage_abs_mean": advantages.detach().float().abs().mean().item(),
            "loss": loss.detach().float().item(),
            "gradient_norm": _gradient_l2_norm(gradients),
            "initial_log_prob_abs_diff_max": (
                log_ratio.detach().float().abs().max().item()
            ),
        }
    return output


def _round_record(
    *,
    update: int,
    trainer: DDPOTrainer,
    trajectory: Trajectory,
    original_log_probs: torch.Tensor,
    previous_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    fixed_reward: dict[str, float],
    optimization: dict[str, float] | None,
    buckets: tuple[TimestepBucket, ...],
) -> dict[str, Any]:
    cumulative = current_log_probs - original_log_probs
    local = current_log_probs - previous_log_probs
    assert trajectory.advantages is not None
    return {
        "update": update,
        "effective_lora_delta_norm": effective_lora_delta_norm(trainer.model),
        "fixed_reward": fixed_reward,
        "logprob_alignment": advantage_logprob_alignment(
            trajectory.advantages,
            cumulative.mean(dim=1),
        ),
        "local_timestep_metrics": summarize_timestep_log_ratios(
            trajectory.timesteps,
            local,
            clip_range=trainer.config.clip_range,
            buckets=buckets,
        ),
        "cumulative_timestep_metrics": summarize_timestep_log_ratios(
            trajectory.timesteps,
            cumulative,
            clip_range=trainer.config.clip_range,
            buckets=buckets,
        ),
        "optimizer": optimization or {},
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive.")
    config = _build_config(args)
    trainer = DDPOTrainer(config)
    actual_prediction_type = diffusion_prediction_type(trainer.diffusion)
    actual_diffusion_metadata = diffusion_runtime_metadata(
        trainer.model_args,
        trainer.diffusion,
    )
    maximum_timestep = trainer.diffusion.num_timesteps - 1
    buckets = parse_timestep_buckets(
        args.timestep_buckets,
        minimum=1,
        maximum=maximum_timestep,
    )

    LOGGER.info("Collecting one fixed rollout trajectory.")
    trajectory = trainer.collect_rollouts(epoch=0)
    assert trajectory.advantages is not None
    if args.step_only_overfit:
        assert trajectory.step_mask is not None
        trajectory.advantages = trajectory.advantages.clone().masked_fill(
            ~trajectory.step_mask.bool(),
            0.0,
        )
    original_log_probs = trajectory.old_log_probs.clone()
    recomputed = _recompute_log_probs(trainer, trajectory)
    initial_audit_max = recomputed.sub(original_log_probs).abs().max().item()
    if initial_audit_max > config.log_prob_audit_tolerance:
        raise RuntimeError(
            "Initial full-trajectory log-probability audit failed: "
            f"{initial_audit_max:.6g}."
        )

    LOGGER.info("Extracting exact transition noise for deterministic replay.")
    transition_noise, replay_reconstruction_error = _extract_transition_noise(
        trainer,
        trajectory,
    )
    generated = _fixed_noise_generation(trainer, trajectory, transition_noise)
    initial_fixed_reward = _score_fixed_generation(
        trainer,
        trajectory,
        generated,
    )
    initial_reward_replay_error = max(
        abs(initial_fixed_reward["reward_total"] - trajectory.rewards.mean().item()),
        abs(
            initial_fixed_reward["step_reward"]
            - trajectory.step_rewards[trajectory.step_mask.bool()].mean().item()
        ),
    )

    sensitivity = xstart_ddim_score_sensitivity(
        trainer.diffusion,
        eta=config.ddim_eta,
    )
    epsilon_sensitivity = epsilon_ddim_score_sensitivity(
        trainer.diffusion,
        eta=config.ddim_eta,
    )
    actual_sensitivity = (
        epsilon_sensitivity
        if actual_prediction_type == "epsilon"
        else sensitivity
    )
    LOGGER.info("Computing advantage-weighted gradient norms by timestep bucket.")
    gradient_buckets = _bucket_gradient_diagnostics(
        trainer,
        trajectory,
        buckets,
        pairs_per_bucket=args.gradient_pairs_per_bucket,
        seed=args.seed,
    )

    records = [
        _round_record(
            update=0,
            trainer=trainer,
            trajectory=trajectory,
            original_log_probs=original_log_probs,
            previous_log_probs=original_log_probs,
            current_log_probs=recomputed,
            fixed_reward=initial_fixed_reward,
            optimization=None,
            buckets=buckets,
        )
    ]
    previous_log_probs = recomputed
    for update in range(1, args.updates + 1):
        LOGGER.info("Fixed-trajectory optimizer update %d/%d.", update, args.updates)
        trajectory.old_log_probs = previous_log_probs.clone()
        optimization = trainer.optimize(trajectory)
        current_log_probs = _recompute_log_probs(trainer, trajectory)
        generated = _fixed_noise_generation(
            trainer,
            trajectory,
            transition_noise,
        )
        fixed_reward = _score_fixed_generation(
            trainer,
            trajectory,
            generated,
        )
        records.append(
            _round_record(
                update=update,
                trainer=trainer,
                trajectory=trajectory,
                original_log_probs=original_log_probs,
                previous_log_probs=previous_log_probs,
                current_log_probs=current_log_probs,
                fixed_reward=fixed_reward,
                optimization=optimization,
                buckets=buckets,
            )
        )
        previous_log_probs = current_log_probs

    score_values = [
        value["score_sensitivity"] for value in sensitivity.values()
    ]
    epsilon_score_values = [
        value["score_sensitivity"] for value in epsilon_sensitivity.values()
    ]
    report = {
        "schema_version": 1,
        "config": {
            "seed": args.seed,
            "device": str(trainer.device),
            "precision": config.precision,
            "prediction_type": actual_prediction_type,
            "mdm_diffusion": actual_diffusion_metadata,
            "sample_steps": trainer.diffusion.num_timesteps,
            "rollout_batch_size": config.rollout_batch_size,
            "train_batch_size": config.train_batch_size,
            "learning_rate": config.learning_rate,
            "clip_range": config.clip_range,
            "updates": args.updates,
            "step_only_overfit": bool(args.step_only_overfit),
            "timestep_fraction": config.timestep_fraction,
            "timestep_buckets": [bucket.label for bucket in buckets],
        },
        "trajectory": {
            "samples": float(len(trajectory.rewards)),
            "transitions_per_sample": float(trajectory.timesteps.shape[1]),
            "initial_full_logprob_audit_max": initial_audit_max,
            "transition_noise_reconstruction_error_max": (
                replay_reconstruction_error
            ),
            "initial_reward_replay_error_max": initial_reward_replay_error,
        },
        "xstart_sensitivity": {
            "minimum": min(score_values),
            "maximum": max(score_values),
            "max_min_ratio": max(score_values) / max(min(score_values), 1.0e-30),
            "by_bucket": summarize_sensitivity_buckets(
                sensitivity,
                buckets,
            ),
            "by_timestep": {
                str(key): value for key, value in sensitivity.items()
            },
        },
        "hypothetical_epsilon_sensitivity": {
            "minimum": min(epsilon_score_values),
            "maximum": max(epsilon_score_values),
            "max_min_ratio": (
                max(epsilon_score_values)
                / max(min(epsilon_score_values), 1.0e-30)
            ),
            "by_bucket": summarize_sensitivity_buckets(
                epsilon_sensitivity,
                buckets,
            ),
            "by_timestep": {
                str(key): value
                for key, value in epsilon_sensitivity.items()
            },
        },
        "actual_sensitivity": {
            "prediction_type": actual_prediction_type,
            "minimum": min(
                value["score_sensitivity"]
                for value in actual_sensitivity.values()
            ),
            "maximum": max(
                value["score_sensitivity"]
                for value in actual_sensitivity.values()
            ),
            "by_bucket": summarize_sensitivity_buckets(
                actual_sensitivity,
                buckets,
            ),
        },
        "initial_gradient_by_timestep_bucket": gradient_buckets,
        "fixed_overfit": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    temporary.replace(output)
    LOGGER.info("Saved DDPO/MDM compatibility diagnostic: %s", output)
    if args.print_report:
        printed = report
    else:
        initial = records[0]
        final = records[-1]
        printed = {
            "output": str(output),
            "prediction_type": actual_prediction_type,
            "xstart_sensitivity_max_min_ratio": (
                report["xstart_sensitivity"]["max_min_ratio"]
            ),
            "epsilon_sensitivity_max_min_ratio": (
                report["hypothetical_epsilon_sensitivity"]["max_min_ratio"]
            ),
            "initial_gradient_by_timestep_bucket": gradient_buckets,
            "fixed_step_reward_initial": initial["fixed_reward"]["step_reward"],
            "fixed_step_reward_final": final["fixed_reward"]["step_reward"],
            "final_logprob_alignment": final["logprob_alignment"],
            "final_effective_lora_delta_norm": (
                final["effective_lora_delta_norm"]
            ),
        }
    print(json.dumps(printed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
