from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .calibration import RewardCalibration, load_reward_calibration
from .config import TrainConfig
from .count_conditioning import (
    count_conditioning_metadata,
    count_conditioning_signature,
    set_count_conditioning_trainable,
    validate_count_conditioning_signature,
)
from .diffusion import ddim_step_with_logprob
from .lora import (
    LoRAReport,
    configure_trainable_policy,
    load_trainable_state_dict,
    parameter_counts,
    set_lora_trainable,
    trainable_state_dict,
)
from .policy_io import policy_checkpoint_id
from .rewards import (
    MotionReward,
    RewardOutput,
    add_step_reward,
    apply_step_m2m_policy,
)
from .runtime import (
    autocast_context,
    bootstrap_external_repositories,
    build_data_loader,
    build_dataset,
    build_mdm,
    build_model_kwargs,
    build_policy_model,
    CachedTextEmbedding,
    diffusion_prediction_type,
    diffusion_runtime_metadata,
    load_model_args,
    resolve_device,
    resolve_reward_device,
    validate_diffusion_runtime_metadata,
    seed_everything,
    split_text_embeddings,
)
from .tracking import SwanLabTracker, format_training_metrics
from .step_calibration import (
    StepRewardCalibration,
    compute_target_error_scales,
    load_step_reward_calibration,
)
from .step_data import (
    FixedStepEvalPool,
    StepMotionDataset,
    SyntheticStepConditionDataset,
    build_step_data_loader,
    create_fixed_step_eval_pool,
    create_synthetic_step_records,
    load_fixed_step_eval_pool,
    load_humanml_stats,
    load_step_manifest,
    save_fixed_step_eval_pool,
    stratified_step_split,
    target_histogram,
)
from .step_reward import HardStepDetector, compute_step_count_reward


LOGGER = logging.getLogger(__name__)


def apply_optimizer_hyperparameters(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> None:
    """Make current CLI optimizer settings authoritative after resume."""
    for parameter_group in optimizer.param_groups:
        parameter_group.update(
            {
                "lr": config.learning_rate,
                "betas": (config.adam_beta1, config.adam_beta2),
                "weight_decay": config.adam_weight_decay,
                "eps": config.adam_epsilon,
            }
        )


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    checkpoint: dict[str, Any],
    config: TrainConfig,
) -> bool:
    """Restore optimizer state unless this resume is an algorithm migration."""
    if config.reset_optimizer_on_resume:
        return False
    optimizer.load_state_dict(checkpoint["optimizer"])
    apply_optimizer_hyperparameters(optimizer, config)
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return True


def shuffled_sample_minibatches(
    num_samples: int,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Shuffle individual rollout samples before forming equal PPO batches."""
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("Sample count and PPO batch size must be positive.")
    if num_samples % batch_size != 0:
        raise ValueError(
            "The PPO sample count must be divisible by the train batch size."
        )
    permutation = torch.randperm(num_samples, generator=generator)
    return list(permutation.split(batch_size))


def log_prob_consistency_metrics(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    tolerance: float,
) -> dict[str, float]:
    """Audit rollout/recomputed log probabilities before policy mutation."""
    if old_log_probs.shape != new_log_probs.shape:
        raise ValueError("Old and new log-probability tensors must match.")
    if old_log_probs.numel() == 0:
        raise ValueError("Cannot audit empty log-probability tensors.")
    if tolerance <= 0:
        raise ValueError("Log-probability audit tolerance must be positive.")

    log_ratio = new_log_probs.detach().float() - old_log_probs.detach().float()
    if not torch.isfinite(log_ratio).all():
        raise FloatingPointError(
            "Non-finite old/new log-probability difference before optimization."
        )
    absolute = log_ratio.abs()
    ratio = log_ratio.exp()
    maximum = absolute.max().item()
    metrics = {
        "initial_log_prob_abs_diff_mean": absolute.mean().item(),
        "initial_log_prob_abs_diff_max": maximum,
        "initial_log_ratio_mean": log_ratio.mean().item(),
        "initial_log_ratio_std": log_ratio.std(unbiased=False).item(),
        "initial_log_ratio_max": maximum,
        "initial_ratio_mean": ratio.mean().item(),
        "initial_ratio_std": ratio.std(unbiased=False).item(),
        "initial_ratio_abs_deviation_max": (ratio - 1.0).abs().max().item(),
    }
    if maximum > tolerance:
        raise RuntimeError(
            "Old/new log-probability consistency audit failed before the "
            f"first optimizer update: max_abs_diff={maximum:.6g} exceeds "
            f"tolerance={tolerance:.6g}."
        )
    return metrics


def merge_log_prob_audit_metrics(
    records: list[dict[str, float]],
) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot merge an empty log-probability audit.")
    names = set(records[0])
    if any(set(record) != names for record in records[1:]):
        raise ValueError("Log-probability audit records have different fields.")
    return {
        name: (
            max(record[name] for record in records)
            if name.endswith("_max")
            else float(np.mean([record[name] for record in records]))
        )
        for name in names
    }


def _tensor_collection_l2_norm(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    squared_norm = sum(
        tensor.detach().float().square().sum()
        for tensor in tensors
    )
    return squared_norm.sqrt().item()


def gradient_l2_norm(
    gradients: list[torch.Tensor | None],
    *,
    scale: float = 1.0,
) -> float:
    if scale <= 0:
        raise ValueError("Gradient scale must be positive.")
    values = [
        gradient.detach().float() / scale
        for gradient in gradients
        if gradient is not None
    ]
    return _tensor_collection_l2_norm(values)


def calibrate_anchor_lambda(
    ppo_grad_norm: float,
    anchor_grad_norm: float,
    target_ratio: float,
    *,
    epsilon: float = 1.0e-12,
) -> float:
    if target_ratio < 0:
        raise ValueError("Anchor gradient target ratio cannot be negative.")
    if (
        not math.isfinite(ppo_grad_norm)
        or not math.isfinite(anchor_grad_norm)
        or ppo_grad_norm <= epsilon
        or anchor_grad_norm <= epsilon
    ):
        raise ValueError(
            "Cannot calibrate anchor lambda from zero or non-finite gradients."
        )
    return target_ratio * ppo_grad_norm / anchor_grad_norm


class NativeMDMTrainingModel(torch.nn.Module):
    """Transparent adapter expected by MDM's native training_losses code."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(*args, **kwargs)


FIXED_EVAL_POOL_VERSION = 1


@dataclass(frozen=True)
class FixedEvalPool:
    dataset_indices: torch.Tensor
    motion: torch.Tensor
    lengths: torch.Tensor
    texts: list[str]
    split: str
    noise_seed: int
    prompt_noise_seeds: torch.Tensor
    pool_id: str = ""

    @property
    def prompt_count(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class FixedEvalResult:
    metrics: dict[str, Any]
    total_per_prompt: torch.Tensor
    retrieval_per_prompt: torch.Tensor
    m2m_per_prompt: torch.Tensor
    total_by_prompt: torch.Tensor | None = None
    retrieval_by_prompt: torch.Tensor | None = None
    m2m_by_prompt: torch.Tensor | None = None


@dataclass(frozen=True)
class FixedStepEvalResult:
    metrics: dict[str, Any]
    total_per_prompt: torch.Tensor
    retrieval_per_prompt: torch.Tensor
    m2m_per_prompt: torch.Tensor
    step_reward_per_prompt: torch.Tensor
    exact_per_prompt: torch.Tensor
    within_one_per_prompt: torch.Tensor
    mae_per_prompt: torch.Tensor
    detected_mean_per_prompt: torch.Tensor
    soft_count_mean_per_prompt: torch.Tensor
    soft_error_mean_per_prompt: torch.Tensor
    soft_mae_per_prompt: torch.Tensor
    candidate_count_mean_per_prompt: torch.Tensor
    candidate_spacing_mean_per_prompt: torch.Tensor
    ankle_high_frequency_ratio_per_prompt: torch.Tensor
    total_by_prompt: torch.Tensor | None = None
    retrieval_by_prompt: torch.Tensor | None = None
    m2m_by_prompt: torch.Tensor | None = None
    step_reward_by_prompt: torch.Tensor | None = None
    detected_steps_by_prompt: torch.Tensor | None = None
    soft_count_by_prompt: torch.Tensor | None = None
    candidate_count_by_prompt: torch.Tensor | None = None
    candidate_spacing_by_prompt: torch.Tensor | None = None
    ankle_high_frequency_ratio_by_prompt: torch.Tensor | None = None
    target_error_scales: dict[str, float] | None = None


def _update_hash_with_tensor(
    digest: Any,
    tensor: torch.Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())


def fixed_eval_pool_id(pool: FixedEvalPool) -> str:
    digest = hashlib.sha256()
    digest.update(str(FIXED_EVAL_POOL_VERSION).encode("utf-8"))
    digest.update(pool.split.encode("utf-8"))
    digest.update(str(pool.noise_seed).encode("utf-8"))
    _update_hash_with_tensor(digest, pool.dataset_indices)
    _update_hash_with_tensor(digest, pool.motion)
    _update_hash_with_tensor(digest, pool.lengths)
    _update_hash_with_tensor(digest, pool.prompt_noise_seeds)
    for value in pool.texts:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_fixed_eval_pool(pool: FixedEvalPool) -> FixedEvalPool:
    prompt_count = pool.prompt_count
    if prompt_count <= 0:
        raise ValueError("Fixed-eval pool cannot be empty.")
    if pool.dataset_indices.shape != (prompt_count,):
        raise ValueError("Fixed-eval dataset indices have an invalid shape.")
    if pool.lengths.shape != (prompt_count,):
        raise ValueError("Fixed-eval lengths have an invalid shape.")
    if pool.motion.ndim != 4 or pool.motion.shape[0] != prompt_count:
        raise ValueError("Fixed-eval GT motion has an invalid shape.")
    if pool.prompt_noise_seeds.shape != (prompt_count,):
        raise ValueError("Fixed-eval prompt noise seeds have an invalid shape.")
    normalized = FixedEvalPool(
        dataset_indices=pool.dataset_indices.detach().cpu().long(),
        motion=pool.motion.detach().cpu().float(),
        lengths=pool.lengths.detach().cpu().long(),
        texts=list(pool.texts),
        split=pool.split,
        noise_seed=int(pool.noise_seed),
        prompt_noise_seeds=pool.prompt_noise_seeds.detach().cpu().long(),
    )
    calculated_id = fixed_eval_pool_id(normalized)
    if pool.pool_id and pool.pool_id != calculated_id:
        raise ValueError(
            "Fixed-eval pool checksum mismatch; the persisted pool is corrupt."
        )
    return FixedEvalPool(
        dataset_indices=normalized.dataset_indices,
        motion=normalized.motion,
        lengths=normalized.lengths,
        texts=normalized.texts,
        split=normalized.split,
        noise_seed=normalized.noise_seed,
        prompt_noise_seeds=normalized.prompt_noise_seeds,
        pool_id=calculated_id,
    )


def save_fixed_eval_pool(pool: FixedEvalPool, path: Path) -> Path:
    pool = validate_fixed_eval_pool(pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FIXED_EVAL_POOL_VERSION,
        "pool_id": pool.pool_id,
        "split": pool.split,
        "dataset_indices": pool.dataset_indices,
        "texts": pool.texts,
        "lengths": pool.lengths,
        "gt_motion": pool.motion,
        "noise_seed": pool.noise_seed,
        "prompt_noise_seeds": pool.prompt_noise_seeds,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    return path


def load_fixed_eval_pool(path: Path) -> FixedEvalPool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != FIXED_EVAL_POOL_VERSION:
        raise ValueError(
            f"Unsupported fixed-eval pool version in {path}: "
            f"{payload.get('version')!r}."
        )
    required = {
        "pool_id",
        "split",
        "dataset_indices",
        "texts",
        "lengths",
        "gt_motion",
        "noise_seed",
        "prompt_noise_seeds",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"Fixed-eval pool is missing fields: {missing}")
    return validate_fixed_eval_pool(
        FixedEvalPool(
            dataset_indices=payload["dataset_indices"],
            motion=payload["gt_motion"],
            lengths=payload["lengths"],
            texts=list(payload["texts"]),
            split=str(payload["split"]),
            noise_seed=int(payload["noise_seed"]),
            prompt_noise_seeds=payload["prompt_noise_seeds"],
            pool_id=str(payload["pool_id"]),
        )
    )


def bootstrap_standard_error(
    values: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> float:
    values = values.detach().float().cpu().reshape(-1)
    if values.numel() < 2:
        return 0.0
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(),
        (samples, values.numel()),
        generator=generator,
    )
    bootstrap_means = values[indices].mean(dim=1)
    return bootstrap_means.std(unbiased=True).item()


def summarize_fixed_eval_component(
    name: str,
    current: torch.Tensor,
    baseline: torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    current = current.detach().float().cpu().reshape(-1)
    baseline = baseline.detach().float().cpu().reshape(-1)
    if current.shape != baseline.shape or current.numel() == 0:
        raise ValueError("Fixed-eval current and baseline values must match.")
    delta = current - baseline
    return {
        name: current.mean().item(),
        f"{name}_median": torch.quantile(current, 0.5).item(),
        f"{name}_bootstrap_se": bootstrap_standard_error(
            current,
            samples=bootstrap_samples,
            seed=seed,
        ),
        f"{name}_baseline": baseline.mean().item(),
        f"{name}_delta": delta.mean().item(),
        f"{name}_delta_median": torch.quantile(delta, 0.5).item(),
        f"{name}_improvement_fraction": (delta > 0).float().mean().item(),
        f"{name}_delta_bootstrap_se": bootstrap_standard_error(
            delta,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
    }


def summarize_fixed_eval_error(
    name: str,
    current: torch.Tensor,
    baseline: torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    """Summarize a paired validation metric where lower is better."""
    current = current.detach().float().cpu().reshape(-1)
    baseline = baseline.detach().float().cpu().reshape(-1)
    if current.shape != baseline.shape or current.numel() == 0:
        raise ValueError("Fixed-eval current and baseline errors must match.")
    delta = current - baseline
    return {
        name: current.mean().item(),
        f"{name}_median": torch.quantile(current, 0.5).item(),
        f"{name}_bootstrap_se": bootstrap_standard_error(
            current,
            samples=bootstrap_samples,
            seed=seed,
        ),
        f"{name}_baseline": baseline.mean().item(),
        f"{name}_delta": delta.mean().item(),
        f"{name}_delta_median": torch.quantile(delta, 0.5).item(),
        f"{name}_improvement_fraction": (delta < 0).float().mean().item(),
        f"{name}_delta_bootstrap_se": bootstrap_standard_error(
            delta,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
    }


def compute_balanced_validation_metrics(
    retrieval_current: torch.Tensor,
    retrieval_baseline: torch.Tensor,
    m2m_current: torch.Tensor,
    m2m_baseline: torch.Tensor,
    *,
    retrieval_scale: float,
    m2m_scale: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    """Compute paired, calibration-normalized balanced validation deltas."""
    tensors = [
        value.detach().float().cpu().reshape(-1)
        for value in (
            retrieval_current,
            retrieval_baseline,
            m2m_current,
            m2m_baseline,
        )
    ]
    if any(value.shape != tensors[0].shape for value in tensors[1:]):
        raise ValueError("Balanced validation component tensors must match.")
    if tensors[0].numel() == 0:
        raise ValueError("Balanced validation cannot use empty tensors.")
    if retrieval_scale <= 0 or m2m_scale <= 0:
        raise ValueError("Balanced validation scales must be positive.")
    retrieval_delta = tensors[0] - tensors[1]
    m2m_delta = tensors[2] - tensors[3]
    normalized_retrieval = retrieval_delta / retrieval_scale
    normalized_m2m = m2m_delta / m2m_scale
    balanced = 0.5 * normalized_retrieval + 0.5 * normalized_m2m
    return {
        "eval_normalized_retrieval_delta": normalized_retrieval.mean().item(),
        "eval_normalized_m2m_delta": normalized_m2m.mean().item(),
        "eval_balanced_score": balanced.mean().item(),
        "eval_balanced_score_median": torch.quantile(balanced, 0.5).item(),
        "eval_balanced_score_bootstrap_se": bootstrap_standard_error(
            balanced,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }


def repeat_prompt_batch(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    texts: list[str],
    samples_per_prompt: int | list[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Repeat conditioning items contiguously for grouped DDPO rollouts.

    ``samples_per_prompt`` may be one shared group size or one size per
    prompt. The latter is used for mixed HumanML (K=4) and step (K=16)
    rollouts while preserving complete, contiguous advantage groups.
    """
    prompt_count = motion.shape[0]
    if lengths.shape[0] != prompt_count or len(texts) != prompt_count:
        raise ValueError("Motion, length, and text prompt counts must match.")
    if isinstance(samples_per_prompt, int):
        repeat_counts = torch.full(
            (prompt_count,),
            samples_per_prompt,
            dtype=torch.long,
        )
    else:
        repeat_counts = torch.as_tensor(samples_per_prompt, dtype=torch.long)
    if repeat_counts.ndim != 1 or repeat_counts.shape[0] != prompt_count:
        raise ValueError(
            "samples_per_prompt must be one integer or one count for each "
            "motion/text prompt."
        )
    if (repeat_counts < 2).any():
        raise ValueError("Grouped DDPO requires at least two samples per prompt.")
    repeat_counts = repeat_counts.to(device=motion.device)
    repeated_motion = motion.repeat_interleave(repeat_counts, dim=0)
    repeated_lengths = lengths.repeat_interleave(repeat_counts, dim=0)
    repeated_texts = [
        text
        for text, count in zip(texts, repeat_counts.tolist())
        for _ in range(count)
    ]
    prompt_ids = torch.arange(
        prompt_count,
        device=motion.device,
    ).repeat_interleave(repeat_counts)
    return repeated_motion, repeated_lengths, repeated_texts, prompt_ids


def compute_grouped_advantages(
    rewards: torch.Tensor,
    prompt_ids: torch.Tensor,
    epsilon: float,
    mode: str = "group_whiten",
    std_floor: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Remove prompt difficulty with a selectable within-group reward scale."""
    if rewards.ndim != 1 or prompt_ids.shape != rewards.shape:
        raise ValueError("Rewards and prompt_ids must be matching 1-D tensors.")
    if epsilon <= 0:
        raise ValueError("Advantage epsilon must be positive.")
    if mode not in {"group_centered", "group_whiten", "group_shrink"}:
        raise ValueError(f"Unknown grouped advantage mode: {mode!r}.")
    if mode == "group_shrink" and (std_floor is None or std_floor <= 0):
        raise ValueError("group_shrink requires a positive fixed std floor.")

    centered_rewards = torch.zeros_like(rewards)
    group_means: list[torch.Tensor] = []
    group_stds: list[torch.Tensor] = []
    unique_prompt_ids = torch.unique(prompt_ids, sorted=True)
    for prompt_id in unique_prompt_ids:
        mask = prompt_ids == prompt_id
        group_rewards = rewards[mask]
        if group_rewards.numel() < 2:
            raise ValueError(
                f"Prompt group {int(prompt_id)} has fewer than two samples."
            )
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False)
        centered_rewards[mask] = group_rewards - group_mean
        group_means.append(group_mean)
        group_stds.append(group_std)

    means = torch.stack(group_means)
    stds = torch.stack(group_stds)
    centered_std = centered_rewards.std(unbiased=False)
    if mode == "group_whiten":
        advantages = torch.zeros_like(rewards)
        for prompt_id, group_std in zip(unique_prompt_ids, stds):
            mask = prompt_ids == prompt_id
            advantages[mask] = centered_rewards[mask] / (group_std + epsilon)
    elif mode == "group_centered":
        advantages = centered_rewards / (centered_std + epsilon)
    else:
        assert std_floor is not None
        advantages = torch.zeros_like(rewards)
        for prompt_id, group_std in zip(unique_prompt_ids, stds):
            mask = prompt_ids == prompt_id
            denominator = torch.sqrt(group_std.square() + std_floor**2)
            advantages[mask] = centered_rewards[mask] / denominator

    stats = {
        "unique_prompts": float(len(unique_prompt_ids)),
        "reward_within_prompt_std": stds.mean().item(),
        "reward_between_prompt_std": means.std(unbiased=False).item(),
        "reward_centered_std": centered_std.item(),
        "reward_group_std_min": stds.min().item(),
        "reward_group_std_median": stds.median().item(),
        "reward_group_std_max": stds.max().item(),
        "potential_group_whiten_scale_max": (
            1.0 / stds.min().clamp_min(epsilon)
        ).item(),
        "zero_variance_prompt_fraction": (
            (stds < epsilon).float().mean().item()
        ),
    }
    if mode == "group_shrink":
        assert std_floor is not None
        stats.update(
            {
                "advantage_std_floor": float(std_floor),
                "effective_shrink_scale_max": (
                    1.0 / torch.sqrt(stds.min().square() + std_floor**2)
                ).item(),
            }
        )
    return advantages, stats


def _pearson_tensor(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().float().reshape(-1)
    second = second.detach().float().reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if denominator <= 0:
        return 0.0
    return (first * second).sum().div(denominator).item()


def grouped_reward_information_metrics(
    rewards: torch.Tensor,
    prompt_ids: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon: float,
    prefix: str,
    tie_tolerance: float = 1.0e-6,
) -> dict[str, float]:
    """Measure how much within-prompt ranking information a reward carries."""

    if rewards.ndim != 1 or prompt_ids.shape != rewards.shape:
        raise ValueError("Grouped reward information tensors must be matching 1-D.")
    if advantages.shape != rewards.shape:
        raise ValueError("Grouped reward advantages must match rewards.")
    if tie_tolerance <= 0:
        raise ValueError("Reward tie tolerance must be positive.")
    unique_levels: list[float] = []
    concentrations: list[float] = []
    tied_pairs = 0
    total_pairs = 0
    for prompt_id in torch.unique(prompt_ids, sorted=True):
        active = prompt_ids == prompt_id
        group_rewards = rewards[active].detach().float()
        group_advantages = advantages[active].detach().float()
        quantized = torch.round(group_rewards / tie_tolerance)
        unique_levels.append(float(torch.unique(quantized).numel()))
        if len(group_rewards) >= 2:
            differences = (
                group_rewards[:, None] - group_rewards[None, :]
            ).abs()
            upper = torch.triu(
                torch.ones_like(differences, dtype=torch.bool),
                diagonal=1,
            )
            tied_pairs += int((differences[upper] <= tie_tolerance).sum())
            total_pairs += int(upper.sum())
        positive = group_advantages.clamp_min(0.0)
        positive_sum = positive.sum()
        concentrations.append(
            positive.max().div(positive_sum).item()
            if positive_sum > epsilon
            else 0.0
        )
    levels = torch.tensor(unique_levels, dtype=torch.float32)
    concentration = torch.tensor(concentrations, dtype=torch.float32)
    return {
        f"{prefix}_unique_reward_levels_mean": levels.mean().item(),
        f"{prefix}_unique_reward_levels_median": levels.median().item(),
        f"{prefix}_unique_reward_levels_min": levels.min().item(),
        f"{prefix}_pairwise_reward_tie_fraction": (
            tied_pairs / total_pairs if total_pairs else 0.0
        ),
        f"{prefix}_nonzero_advantage_sample_fraction": (
            (advantages.abs() > epsilon).float().mean().item()
        ),
        f"{prefix}_top1_advantage_concentration": (
            concentration.mean().item()
        ),
    }


def compute_component_shrink_advantages(
    retrieval_rewards: torch.Tensor,
    m2m_rewards: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    epsilon: float,
    retrieval_std_floor: float,
    m2m_std_floor: float,
    retrieval_weight: float = 0.5,
    m2m_weight: float = 0.5,
    step_retrieval_weight: float | None = None,
    step_m2m_weight: float | None = None,
    step_rewards: torch.Tensor | None = None,
    step_mask: torch.Tensor | None = None,
    step_std_floor: float | None = None,
    step_weight: float = 0.0,
    step_use_m2m_reward: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shrink reward components independently before fixed-weight combining."""
    if retrieval_rewards.shape != m2m_rewards.shape:
        raise ValueError("Retrieval and M2M reward tensors must match.")
    if retrieval_rewards.ndim != 1 or prompt_ids.shape != retrieval_rewards.shape:
        raise ValueError("Component rewards and prompt ids must be matching 1-D tensors.")
    if retrieval_std_floor <= 0 or m2m_std_floor <= 0:
        raise ValueError("Component shrinkage floors must be positive.")
    resolved_step_retrieval_weight = (
        retrieval_weight
        if step_retrieval_weight is None
        else float(step_retrieval_weight)
    )
    resolved_step_m2m_weight = (
        m2m_weight
        if step_m2m_weight is None
        else float(step_m2m_weight)
    )
    if (
        retrieval_weight < 0
        or m2m_weight < 0
        or resolved_step_retrieval_weight < 0
        or resolved_step_m2m_weight < 0
        or step_weight < 0
    ):
        raise ValueError("Component advantage weights must be non-negative.")
    if retrieval_weight == 0 and m2m_weight == 0 and step_weight == 0:
        raise ValueError("At least one component advantage weight must be non-zero.")
    if (step_rewards is None) != (step_mask is None):
        raise ValueError("Step rewards and step mask must be provided together.")
    active_step_mask: torch.Tensor | None = None
    if step_rewards is not None:
        if step_rewards.shape != retrieval_rewards.shape:
            raise ValueError("Step rewards must match retrieval rewards.")
        assert step_mask is not None
        if step_mask.shape != retrieval_rewards.shape:
            raise ValueError("Step mask must match component rewards.")
        active_step_mask = step_mask.bool()
        for prompt_id in torch.unique(prompt_ids, sorted=True):
            prompt_active = active_step_mask[prompt_ids == prompt_id]
            if prompt_active.any() and not prompt_active.all():
                raise ValueError(
                    "Step reward mask must be constant within every prompt group."
                )
        if step_weight > 0 and (step_std_floor is None or step_std_floor <= 0):
            raise ValueError(
                "A positive step component weight requires a fixed std floor."
            )

    retrieval_advantages, retrieval_stats = compute_grouped_advantages(
        retrieval_rewards,
        prompt_ids,
        epsilon,
        mode="group_shrink",
        std_floor=retrieval_std_floor,
    )
    m2m_advantages, m2m_stats = compute_grouped_advantages(
        m2m_rewards,
        prompt_ids,
        epsilon,
        mode="group_shrink",
        std_floor=m2m_std_floor,
    )
    retrieval_contribution = retrieval_weight * retrieval_advantages
    m2m_contribution = m2m_weight * m2m_advantages
    if active_step_mask is not None:
        retrieval_contribution[active_step_mask] = (
            resolved_step_retrieval_weight
            * retrieval_advantages[active_step_mask]
        )
        m2m_contribution[active_step_mask] = (
            resolved_step_m2m_weight
            * m2m_advantages[active_step_mask]
        )
        if not step_use_m2m_reward:
            m2m_contribution = m2m_contribution.masked_fill(
                active_step_mask,
                0.0,
            )
    combined = retrieval_contribution + m2m_contribution
    comparable = (
        retrieval_advantages.abs() > epsilon
    ) & (m2m_advantages.abs() > epsilon)
    conflict = comparable & (retrieval_advantages * m2m_advantages < 0)
    stats = {
        "component_advantage_correlation": _pearson_tensor(
            retrieval_advantages,
            m2m_advantages,
        ),
        "component_advantage_conflict_fraction": (
            conflict.float().sum().div(comparable.float().sum()).item()
            if comparable.any()
            else 0.0
        ),
        "component_advantage_retrieval_weight": float(retrieval_weight),
        "component_advantage_m2m_weight": float(m2m_weight),
        "component_advantage_step_retrieval_weight": float(
            resolved_step_retrieval_weight
        ),
        "component_advantage_step_m2m_weight": float(
            resolved_step_m2m_weight
        ),
        "component_advantage_step_m2m_enabled": float(step_use_m2m_reward),
        "component_advantage_retrieval_std_floor": float(
            retrieval_std_floor
        ),
        "component_advantage_m2m_std_floor": float(m2m_std_floor),
        "component_advantage_retrieval_std": (
            retrieval_advantages.std(unbiased=False).item()
        ),
        "component_advantage_m2m_std": (
            m2m_advantages.std(unbiased=False).item()
        ),
        "component_advantage_retrieval_contribution_mean_abs": (
            retrieval_contribution.abs().mean().item()
        ),
        "component_advantage_m2m_contribution_mean_abs": (
            m2m_contribution.abs().mean().item()
        ),
        "component_advantage_step_m2m_contribution_mean_abs": (
            m2m_contribution[active_step_mask].abs().mean().item()
            if active_step_mask is not None and active_step_mask.any()
            else 0.0
        ),
        "component_advantage_step_retrieval_contribution_mean_abs": (
            retrieval_contribution[active_step_mask].abs().mean().item()
            if active_step_mask is not None and active_step_mask.any()
            else 0.0
        ),
        "component_advantage_retrieval_group_std_median": (
            retrieval_stats["reward_group_std_median"]
        ),
        "component_advantage_m2m_group_std_median": (
            m2m_stats["reward_group_std_median"]
        ),
        "component_advantage_retrieval_effective_scale_max": (
            retrieval_stats["effective_shrink_scale_max"]
        ),
        "component_advantage_m2m_effective_scale_max": (
            m2m_stats["effective_shrink_scale_max"]
        ),
    }
    if step_rewards is not None and step_weight > 0:
        assert step_mask is not None
        assert step_std_floor is not None
        active = step_mask.bool()
        if active.any():
            active_advantages, step_stats = compute_grouped_advantages(
                step_rewards[active],
                prompt_ids[active],
                epsilon,
                mode="group_shrink",
                std_floor=step_std_floor,
            )
            step_advantages = torch.zeros_like(step_rewards)
            step_advantages[active] = active_advantages
            step_contribution = step_weight * step_advantages
            combined = combined + step_contribution
            retrieval_step_comparable = (
                retrieval_advantages[active].abs() > epsilon
            ) & (active_advantages.abs() > epsilon)
            m2m_step_comparable = (
                m2m_advantages[active].abs() > epsilon
            ) & (active_advantages.abs() > epsilon)

            def conflict_fraction(
                first: torch.Tensor,
                second: torch.Tensor,
                comparable_mask: torch.Tensor,
            ) -> float:
                if not comparable_mask.any():
                    return 0.0
                conflict_mask = comparable_mask & (first * second < 0)
                return conflict_mask.float().sum().div(
                    comparable_mask.float().sum()
                ).item()

            stats.update(
                {
                    "component_advantage_step_weight": float(step_weight),
                    "component_advantage_step_std_floor": float(step_std_floor),
                    "component_advantage_step_std": (
                        active_advantages.std(unbiased=False).item()
                    ),
                    "component_advantage_step_contribution_mean_abs": (
                        step_contribution[active].abs().mean().item()
                    ),
                    "component_advantage_step_group_std_median": (
                        step_stats["reward_group_std_median"]
                    ),
                    "component_advantage_step_zero_variance_prompt_fraction": (
                        step_stats["zero_variance_prompt_fraction"]
                    ),
                    "component_advantage_step_effective_scale_max": (
                        step_stats["effective_shrink_scale_max"]
                    ),
                    "component_advantage_retrieval_step_correlation": (
                        _pearson_tensor(
                            retrieval_advantages[active],
                            active_advantages,
                        )
                    ),
                    "component_advantage_m2m_step_correlation": (
                        _pearson_tensor(
                            m2m_advantages[active],
                            active_advantages,
                        )
                    ),
                    "component_advantage_retrieval_step_conflict_fraction": (
                        conflict_fraction(
                            retrieval_advantages[active],
                            active_advantages,
                            retrieval_step_comparable,
                        )
                    ),
                    "component_advantage_m2m_step_conflict_fraction": (
                        conflict_fraction(
                            m2m_advantages[active],
                            active_advantages,
                            m2m_step_comparable,
                        )
                    ),
                    "component_advantage_step_samples": float(active.sum()),
                }
            )
            stats.update(
                grouped_reward_information_metrics(
                    step_rewards[active],
                    prompt_ids[active],
                    active_advantages,
                    epsilon=epsilon,
                    prefix="component_advantage_step",
                )
            )
        else:
            stats["component_advantage_step_samples"] = 0.0
    return combined, stats


@dataclass
class Trajectory:
    latents: torch.Tensor
    next_latents: torch.Tensor
    timesteps: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    retrieval_rewards: torch.Tensor
    m2m_rewards: torch.Tensor
    texts: list[str]
    text_embeddings: list[CachedTextEmbedding]
    lengths: torch.Tensor
    gt_motion: torch.Tensor
    prompt_ids: torch.Tensor
    step_rewards: torch.Tensor | None = None
    step_mask: torch.Tensor | None = None
    detected_steps: torch.Tensor | None = None
    target_steps: torch.Tensor | None = None
    step_absolute_error: torch.Tensor | None = None
    soft_step_count: torch.Tensor | None = None
    soft_step_error: torch.Tensor | None = None
    step_raw_candidate_count: torch.Tensor | None = None
    step_candidate_count: torch.Tensor | None = None
    step_candidate_spacing_mean: torch.Tensor | None = None
    step_candidate_spacing_min: torch.Tensor | None = None
    step_ankle_high_frequency_ratio: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    group_stats: dict[str, float] | None = None

    @classmethod
    def concatenate(cls, batches: list["Trajectory"]) -> "Trajectory":
        if not batches:
            raise ValueError("Cannot concatenate an empty rollout list.")
        prompt_id_parts: list[torch.Tensor] = []
        prompt_offset = 0
        for batch in batches:
            unique_ids, local_ids = torch.unique(
                batch.prompt_ids,
                sorted=True,
                return_inverse=True,
            )
            prompt_id_parts.append(local_ids + prompt_offset)
            prompt_offset += len(unique_ids)

        def concatenate_optional(name: str) -> torch.Tensor | None:
            values = [getattr(batch, name) for batch in batches]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Trajectory field {name!r} is missing from some batches."
                )
            return torch.cat(values, dim=0)  # type: ignore[arg-type]

        return cls(
            latents=torch.cat([batch.latents for batch in batches], dim=0),
            next_latents=torch.cat(
                [batch.next_latents for batch in batches], dim=0
            ),
            timesteps=torch.cat([batch.timesteps for batch in batches], dim=0),
            old_log_probs=torch.cat(
                [batch.old_log_probs for batch in batches], dim=0
            ),
            rewards=torch.cat([batch.rewards for batch in batches], dim=0),
            retrieval_rewards=torch.cat(
                [batch.retrieval_rewards for batch in batches], dim=0
            ),
            m2m_rewards=torch.cat(
                [batch.m2m_rewards for batch in batches], dim=0
            ),
            texts=[text for batch in batches for text in batch.texts],
            text_embeddings=[
                embedding
                for batch in batches
                for embedding in batch.text_embeddings
            ],
            lengths=torch.cat([batch.lengths for batch in batches], dim=0),
            gt_motion=torch.cat([batch.gt_motion for batch in batches], dim=0),
            prompt_ids=torch.cat(prompt_id_parts, dim=0),
            step_rewards=concatenate_optional("step_rewards"),
            step_mask=concatenate_optional("step_mask"),
            detected_steps=concatenate_optional("detected_steps"),
            target_steps=concatenate_optional("target_steps"),
            step_absolute_error=concatenate_optional("step_absolute_error"),
            soft_step_count=concatenate_optional("soft_step_count"),
            soft_step_error=concatenate_optional("soft_step_error"),
            step_raw_candidate_count=concatenate_optional(
                "step_raw_candidate_count"
            ),
            step_candidate_count=concatenate_optional("step_candidate_count"),
            step_candidate_spacing_mean=concatenate_optional(
                "step_candidate_spacing_mean"
            ),
            step_candidate_spacing_min=concatenate_optional(
                "step_candidate_spacing_min"
            ),
            step_ankle_high_frequency_ratio=concatenate_optional(
                "step_ankle_high_frequency_ratio"
            ),
        )


class DDPOTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        bootstrap_external_repositories(config)
        seed_everything(config.seed)

        self.device = resolve_device(config.device)
        self.reward_device = resolve_reward_device(config, self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32

        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2, sort_keys=True)

        self.model_args = load_model_args(config)
        self.data_loader = build_data_loader(
            config,
            prompt_batch_size=config.humanml_prompts_per_rollout_batch,
        )
        self.data_iterator: Any | None = None
        self.step_data_loader: Any | None = None
        self.step_data_iterator: Any | None = None
        self.step_eval_records: list[Any] = []
        (
            self.model,
            self.diffusion,
            self.base_diffusion,
            self.sample_steps,
        ) = build_mdm(
            config,
            self.model_args,
            self.data_loader,
            self.device,
        )
        self.prediction_type = diffusion_prediction_type(self.diffusion)
        self.diffusion_metadata = diffusion_runtime_metadata(
            self.model_args,
            self.diffusion,
        )

        self.lora_report: LoRAReport | None = configure_trainable_policy(
            self.model,
            mode=config.train_mode,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_target_regex=config.lora_target_regex,
        )
        if self.lora_report is not None and not config.train_lora:
            set_lora_trainable(self.model, False)
        if config.enable_count_conditioning:
            set_count_conditioning_trainable(
                self.model,
                config.train_count_conditioning,
            )
        self.initial_policy_id = ""
        if config.initial_policy_path:
            self._load_initial_policy(config.initial_policy_path)
        self.count_conditioning_metadata = count_conditioning_metadata(self.model)
        with open(
            self.output_dir / "runtime_metadata.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "model_path": str(
                        Path(config.model_path).expanduser().resolve()
                    ),
                    "model_args_path": str(
                        Path(config.model_args_path).expanduser().resolve()
                    ),
                    "mdm_diffusion": self.diffusion_metadata,
                    "count_conditioning": self.count_conditioning_metadata,
                    "initial_policy_path": str(
                        Path(config.initial_policy_path).expanduser().resolve()
                    ) if config.initial_policy_path else "",
                    "ddpo_sampler": "stochastic_ddim",
                    "ddim_eta": config.ddim_eta,
                    "guidance_scale": config.guidance_scale,
                    "clip_denoised": config.clip_denoised,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        self.model.eval()
        self.policy_model = build_policy_model(
            self.model,
            config.guidance_scale,
        )
        self.policy_model.eval()
        self.anchor_enabled = bool(
            config.anchor_lambda > 0
            or config.anchor_auto_grad_ratio > 0
        )
        self.anchor_model = (
            NativeMDMTrainingModel(self.model)
            if self.anchor_enabled
            else None
        )
        self.anchor_lambda_effective = float(config.anchor_lambda)
        self.anchor_lambda_calibrated = bool(
            config.anchor_auto_grad_ratio <= 0
        )

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("The DDPO policy has no trainable parameters.")
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and config.precision == "fp16"
        )

        self.reward_model = MotionReward(config, self.reward_device)
        self.reward_calibration: RewardCalibration | None = (
            load_reward_calibration(config.reward_calibration_path)
            if config.reward_calibration_path
            else None
        )
        self._validate_calibration_mdm(
            self.reward_calibration,
            source="Reward calibration",
        )
        self.step_reward_calibration: StepRewardCalibration | None = (
            load_step_reward_calibration(config.step_reward_calibration_path)
            if config.step_reward_calibration_path
            else None
        )
        self._validate_calibration_mdm(
            self.step_reward_calibration,
            source="Step reward calibration",
        )
        self.step_detector: HardStepDetector | None = None
        self.step_mdm_mean: torch.Tensor | None = None
        self.step_mdm_std: torch.Tensor | None = None
        if config.enable_step_reward:
            if self.step_reward_calibration is not None:
                self.step_reward_calibration.validate_settings(
                    detector_config=config.step_detector_config(),
                    reward_config=config.step_reward_config(),
                    samples_per_prompt=config.step_samples_per_prompt,
                )
            self.step_detector = HardStepDetector(
                backend=config.step_detector_backend,
                fps=config.step_detector_fps,
                motion_rule_root=config.step_detector_root,
                lead_threshold=config.step_detector_lead_threshold,
                rgdno_threshold=config.step_detector_rgdno_threshold,
                soft_lead_temperature=config.step_soft_lead_temperature,
                soft_length_temperature=config.step_soft_length_temperature,
                soft_progress_temperature=(
                    config.step_soft_progress_temperature
                ),
                soft_cluster_gap_seconds=(
                    config.step_soft_cluster_gap_seconds
                ),
                ankle_high_frequency_cutoff_hz=(
                    config.step_ankle_high_frequency_cutoff_hz
                ),
            )
            self._initialize_step_training_data()
        if (
            config.advantage_mode == "group_shrink"
            and self.reward_calibration is not None
            and (
                not math.isclose(
                    self.reward_calibration.reward_weight("retrieval"),
                    config.retrieval_weight,
                )
                or not math.isclose(
                    self.reward_calibration.reward_weight("m2m"),
                    config.m2m_weight,
                )
            )
        ):
            raise ValueError(
                "group_shrink total reward weights must match the reward "
                "weights stored in reward_calibration.json."
            )
        self.start_epoch = 0
        self.global_step = 0
        self.fixed_eval_baseline: dict[str, Any] | None = None
        self.fixed_eval_baseline_per_prompt: dict[str, torch.Tensor] | None = None
        self.checkpoint_fixed_eval_pool_id: str | None = None
        self.best_balanced_score: float | None = None
        self.best_balanced_epoch: int | None = None
        self.best_retrieval_delta: float | None = None
        self.best_retrieval_epoch: int | None = None
        self.best_m2m_delta: float | None = None
        self.best_m2m_epoch: int | None = None
        self.fixed_step_eval_baseline_per_prompt: (
            dict[str, torch.Tensor] | None
        ) = None
        self.best_step_reward_delta: float | None = None
        self.best_step_epoch: int | None = None
        self.best_step_acceptance_score: float | None = None
        self.best_step_acceptance_epoch: int | None = None
        self.evals_without_improvement = 0
        self.fixed_step_eval_pool: FixedStepEvalPool | None = None
        self.checkpoint_fixed_step_eval_pool_id: str | None = None
        if config.resume:
            self._load_checkpoint(config.resume)
        self.fixed_eval_pool = (
            self._load_or_create_fixed_eval_pool()
            if config.fixed_eval_every > 0
            else None
        )
        self.fixed_step_eval_pool = (
            self._load_or_create_fixed_step_eval_pool()
            if config.enable_step_reward and config.fixed_eval_every > 0
            else None
        )
        if (
            self.fixed_step_eval_pool is not None
            and self.checkpoint_fixed_step_eval_pool_id is not None
            and self.fixed_step_eval_pool.pool_id
            != self.checkpoint_fixed_step_eval_pool_id
        ):
            raise ValueError(
                "Resume checkpoint fixed step-eval pool does not match "
                "fixed_step_eval_pool.pt."
            )
        if (
            self.fixed_eval_pool is not None
            and self.checkpoint_fixed_eval_pool_id is not None
            and self.fixed_eval_pool.pool_id
            != self.checkpoint_fixed_eval_pool_id
        ):
            raise ValueError(
                "Resume checkpoint fixed-eval pool does not match "
                "fixed_eval_pool.pt."
            )
        if config.resume:
            resume_dir = Path(config.resume).expanduser().resolve().parent
            for name in (
                "best_balanced.pt",
                "best_retrieval.pt",
                "best_m2m.pt",
                "best_step.pt",
                "best_step_acceptance.pt",
            ):
                source = resume_dir / name
                target = self.output_dir / name
                if source.exists() and source != target and not target.exists():
                    shutil.copy2(source, target)
                    LOGGER.info("Copied resumed best checkpoint: %s", target)

        total, trainable = parameter_counts(self.model)
        LOGGER.info(
            "Policy parameters: trainable=%s / total=%s (%.3f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
        )
        if self.lora_report is not None:
            LOGGER.info(
                "Injected %d LoRA adapters with %s trainable parameters.",
                self.lora_report.adapters,
                f"{self.lora_report.trainable_parameters:,}",
            )
        LOGGER.info(
            "Diffusion prediction type=%s, steps=%d, "
            "policy transitions per sample=%d, "
            "policy device=%s, reward device=%s",
            self.prediction_type,
            self.sample_steps,
            self.sample_steps - 1,
            self.device,
            self.reward_device,
        )
        LOGGER.info(
            "MDM diffusion configuration: %s",
            json.dumps(self.diffusion_metadata, sort_keys=True),
        )
        if (
            config.precision != "no"
            and config.train_batch_size != config.rollout_batch_size
        ):
            LOGGER.warning(
                "Low-precision old/new log-probability agreement is best when "
                "--train-batch-size equals --rollout-batch-size."
            )

    def _validate_calibration_mdm(
        self,
        calibration: RewardCalibration | StepRewardCalibration | None,
        *,
        source: str,
    ) -> None:
        if calibration is None:
            return
        metadata = calibration.payload.get("metadata", {})
        artifact_count_conditioning = metadata.get("count_conditioning")
        if artifact_count_conditioning is not None:
            validate_count_conditioning_signature(
                self.model,
                artifact_count_conditioning,
                source=source,
            )
        if not self.config.resume:
            artifact_policy_id = str(metadata.get("policy_id", ""))
            if artifact_policy_id != self.initial_policy_id:
                raise ValueError(
                    f"{source} policy id does not match the initialized "
                    "LoRA/count policy. Generate calibration from the exact "
                    "--initial-policy-path checkpoint."
                )
        artifact_diffusion = metadata.get("mdm_diffusion")
        # Legacy calibration files predate this audit and remain loadable.
        if artifact_diffusion is None:
            return
        validate_diffusion_runtime_metadata(
            artifact_diffusion,
            self.diffusion_metadata,
            source=source,
        )
        artifact_model_path = metadata.get("model_path")
        if artifact_model_path is not None:
            calibrated_model = Path(artifact_model_path).expanduser().resolve()
            current_model = Path(self.config.model_path).expanduser().resolve()
            if calibrated_model != current_model:
                raise ValueError(
                    f"{source} was generated for model {calibrated_model}, "
                    f"but the current policy uses {current_model}."
                )

    def preflight_summary(self) -> dict[str, Any]:
        total, trainable = parameter_counts(self.model)
        return {
            "dataset_samples": len(self.data_loader.dataset),
            "prediction_type": self.prediction_type,
            "mdm_diffusion": dict(self.diffusion_metadata),
            "count_conditioning": count_conditioning_metadata(self.model),
            "diffusion_steps": self.sample_steps,
            "policy_transitions": self.sample_steps - 1,
            "policy_parameters": total,
            "trainable_parameters": trainable,
            "lora_adapters": (
                self.lora_report.adapters if self.lora_report is not None else 0
            ),
            "train_lora": self.config.train_lora,
            "reward_t5_tensors_loaded_separately": len(
                self.reward_model.missing_checkpoint_keys
            ),
            "reward_backbone_missing_tensors": 0,
            "reward_normalization_mean_max_delta": (
                self.reward_model.normalization_mean_delta
            ),
            "reward_normalization_std_max_delta": (
                self.reward_model.normalization_std_delta
            ),
            "policy_device": str(self.device),
            "reward_device": str(self.reward_device),
            "prompts_per_rollout_batch": (
                self.config.prompts_per_rollout_batch
            ),
            "samples_per_prompt": self.config.samples_per_prompt,
            "humanml_samples_per_prompt": (
                self.config.human_samples_per_prompt
            ),
            "step_samples_per_prompt": self.config.step_samples_per_prompt,
            "humanml_rollout_samples": self.config.humanml_rollout_samples,
            "step_rollout_samples": self.config.step_rollout_samples,
            "step_motion_ratio": (
                self.config.step_rollout_samples
                / self.config.rollout_batch_size
            ),
            "advantage_mode": self.config.advantage_mode,
            "advantage_std_floor_quantile": (
                self.config.advantage_std_floor_quantile
            ),
            "advantage_total_std_floor": (
                self._advantage_std_floor("total")
                if self.config.advantage_mode == "group_shrink"
                else 0.0
            ),
            "advantage_retrieval_std_floor": (
                self._advantage_std_floor("retrieval")
                if self.config.advantage_mode == "component_shrink"
                else 0.0
            ),
            "advantage_m2m_std_floor": (
                self._advantage_std_floor("m2m")
                if self.config.advantage_mode == "component_shrink"
                else 0.0
            ),
            "advantage_step_std_floor": (
                self._advantage_std_floor("step")
                if self.config.enable_step_reward
                and self.config.advantage_mode == "component_shrink"
                and self.config.effective_step_advantage_step_weight > 0
                else 0.0
            ),
            "timestep_fraction": self.config.timestep_fraction,
            "fixed_eval_enabled": self.fixed_eval_pool is not None,
            "fixed_eval_prompts": self.config.fixed_eval_prompts,
            "fixed_eval_samples_per_prompt": (
                self.config.fixed_eval_samples_per_prompt
            ),
            "fixed_eval_split": self.config.eval_split,
            "fixed_eval_pool_id": (
                self.fixed_eval_pool.pool_id
                if self.fixed_eval_pool is not None
                else ""
            ),
            "early_stop_patience": self.config.early_stop_patience,
            "log_prob_audit_tolerance": (
                self.config.log_prob_audit_tolerance
            ),
            "reward_calibration_id": (
                self.reward_calibration.calibration_id
                if self.reward_calibration is not None
                else ""
            ),
            "anchor_enabled": self.anchor_enabled,
            "anchor_lambda": self.config.anchor_lambda,
            "anchor_auto_grad_ratio": self.config.anchor_auto_grad_ratio,
            "anchor_batch_size": (
                self.config.anchor_batch_size
                or self.config.train_batch_size
            ),
            "step_reward_enabled": self.config.enable_step_reward,
            "step_prompt_ratio": (
                self.config.step_prompts_per_rollout_batch
                / self.config.prompts_per_rollout_batch
            ),
            "step_prompts_per_rollout_batch": (
                self.config.step_prompts_per_rollout_batch
            ),
            "humanml_prompts_per_rollout_batch": (
                self.config.humanml_prompts_per_rollout_batch
            ),
            "step_targets": list(self.config.step_target_values),
            "step_detector": self.config.step_detector_config(),
            "step_reward": self.config.step_reward_config(),
            "step_reward_weight": self.config.step_reward_weight,
            "step_use_m2m_reward": self.config.step_use_m2m_reward,
            "step_balanced_sampling": self.config.step_balanced_sampling,
            "step_advantage_retrieval_weight": (
                self.config.effective_step_advantage_retrieval_weight
            ),
            "step_advantage_m2m_weight": (
                self.config.effective_step_advantage_m2m_weight
            ),
            "step_advantage_step_weight": (
                self.config.effective_step_advantage_step_weight
            ),
            "step_reward_calibration_id": (
                self.step_reward_calibration.calibration_id
                if self.step_reward_calibration is not None
                else ""
            ),
            "fixed_step_eval_pool_id": (
                self.fixed_step_eval_pool.pool_id
                if self.fixed_step_eval_pool is not None
                else ""
            ),
        }

    def _initialize_step_training_data(self) -> None:
        mean, std = load_humanml_stats(self.config.mdm_root)
        records = load_step_manifest(
            self.config.step_data_manifest,
            motion_root=self.config.step_motion_root,
            targets=self.config.step_target_values,
            min_frames=self.config.step_min_frames,
            max_frames=self.config.step_max_frames,
        )
        training, evaluation = stratified_step_split(
            records,
            eval_per_target=self.config.step_eval_samples_per_target,
            split_seed=self.config.step_split_seed,
            prompt_seed=self.config.step_prompt_seed,
        )
        synthetic_length_interval: tuple[int, int] | None = None
        if self.config.step_rollout_source == "synthetic":
            synthetic_records, synthetic_length_interval = (
                create_synthetic_step_records(
                    training,
                    targets=self.config.step_target_values,
                    seed=self.config.step_synthetic_seed,
                )
            )
            dataset = SyntheticStepConditionDataset(
                synthetic_records,
                max_frames=self.config.step_max_frames,
            )
        else:
            dataset = StepMotionDataset(
                training,
                mean=mean,
                std=std,
                max_frames=self.config.step_max_frames,
            )
        self.step_data_loader = build_step_data_loader(
            dataset,
            batch_size=self.config.step_prompts_per_rollout_batch,
            seed=self.config.seed + 7919,
            workers=self.config.data_workers,
            pin_memory=self.config.pin_memory,
            balanced_targets=self.config.step_balanced_sampling,
        )
        self.step_eval_records = list(evaluation)
        self.step_mdm_mean = torch.as_tensor(
            mean,
            device=self.device,
            dtype=torch.float32,
        )
        self.step_mdm_std = torch.as_tensor(
            std,
            device=self.device,
            dtype=torch.float32,
        )
        LOGGER.info(
            "Step mixed data: train=%d, held_out_val=%d, targets=%s, "
            "target_histogram=%s, prompts_per_batch=humanml:%d step:%d, "
            "motions_per_batch=humanml:%d step:%d, balanced_sampling=%s, "
            "rollout_source=%s, shared_length_interval=%s.",
            len(training),
            len(evaluation),
            self.config.step_target_values,
            target_histogram(dataset.records),
            self.config.humanml_prompts_per_rollout_batch,
            self.config.step_prompts_per_rollout_batch,
            self.config.humanml_rollout_samples,
            self.config.step_rollout_samples,
            self.config.step_balanced_sampling,
            self.config.step_rollout_source,
            synthetic_length_interval,
        )

    def _load_or_create_fixed_step_eval_pool(self) -> FixedStepEvalPool:
        if self.step_mdm_mean is None or self.step_mdm_std is None:
            raise RuntimeError("Step training data has not been initialized.")
        output_path = self.output_dir / "fixed_step_eval_pool.pt"
        configured_path = (
            Path(self.config.fixed_step_eval_pool_path).expanduser().resolve()
            if self.config.fixed_step_eval_pool_path
            else None
        )
        source_path: Path | None = None
        if configured_path is not None and configured_path.exists():
            source_path = configured_path
        elif output_path.exists():
            source_path = output_path
        elif self.config.resume:
            candidate = (
                Path(self.config.resume).expanduser().resolve().parent
                / "fixed_step_eval_pool.pt"
            )
            if candidate.exists():
                source_path = candidate
            else:
                raise FileNotFoundError(
                    "Cannot resume step validation because "
                    "fixed_step_eval_pool.pt is missing next to the checkpoint."
                )
        if source_path is None:
            pool = create_fixed_step_eval_pool(
                self.step_eval_records,
                mean=self.step_mdm_mean.detach().cpu().numpy(),
                std=self.step_mdm_std.detach().cpu().numpy(),
                max_frames=self.config.step_max_frames,
                noise_seed=self.config.fixed_eval_seed + 104729,
                detector_backend=self.config.step_detector_backend,
            )
            if configured_path is not None:
                save_fixed_step_eval_pool(pool, configured_path)
        else:
            pool = load_fixed_step_eval_pool(source_path)
        expected_count = (
            len(self.config.step_target_values)
            * self.config.step_eval_samples_per_target
        )
        target_counts = {
            target: int((pool.target_steps == target).sum())
            for target in self.config.step_target_values
        }
        if (
            pool.prompt_count != expected_count
            or pool.detector_backend != self.config.step_detector_backend
            or pool.noise_seed != self.config.fixed_eval_seed + 104729
            or set(pool.target_steps.tolist())
            != set(self.config.step_target_values)
            or any(
                count != self.config.step_eval_samples_per_target
                for count in target_counts.values()
            )
        ):
            raise ValueError(
                "Fixed step-eval pool does not match current step settings."
            )
        if configured_path is not None and not configured_path.exists():
            save_fixed_step_eval_pool(pool, configured_path)
        if output_path != source_path:
            save_fixed_step_eval_pool(pool, output_path)
        LOGGER.info(
            "Fixed step validation pool: prompts=%d, pool_id=%s",
            pool.prompt_count,
            pool.pool_id,
        )
        return pool

    def _create_fixed_eval_pool(self) -> FixedEvalPool:
        """Materialize exact held-out samples without advancing training RNG."""
        from data_loaders.get_data import get_collate_fn

        rng_state = self._rng_state()
        try:
            seed_everything(self.config.fixed_eval_seed)
            dataset = build_dataset(
                self.config,
                split=self.config.eval_split,
            )
            if len(dataset) < self.config.fixed_eval_prompts:
                raise ValueError(
                    "Held-out dataset has fewer samples than "
                    "--fixed-eval-prompts: "
                    f"{len(dataset)} < {self.config.fixed_eval_prompts}."
                )
            generator = torch.Generator().manual_seed(
                self.config.fixed_eval_seed
            )
            dataset_indices = torch.randperm(
                len(dataset),
                generator=generator,
            )[: self.config.fixed_eval_prompts]
            items = [dataset[int(index)] for index in dataset_indices]
            collate_fn = get_collate_fn(
                self.config.dataset,
                hml_mode="train",
                batch_size=self.config.fixed_eval_prompts,
            )
            motion, condition = collate_fn(items)
        finally:
            self._restore_rng_state(rng_state)

        prompt_count = motion.shape[0]
        if (
            prompt_count != self.config.fixed_eval_prompts
            or len(condition["y"]["text"]) != prompt_count
            or len(condition["y"]["lengths"]) != prompt_count
        ):
            raise RuntimeError(
                "Fixed-eval collate returned an unexpected prompt count: "
                f"expected={self.config.fixed_eval_prompts}, "
                f"motion={prompt_count}, "
                f"text={len(condition['y']['text'])}, "
                f"lengths={len(condition['y']['lengths'])}."
            )
        prompt_noise_seeds = (
            torch.arange(prompt_count, dtype=torch.long) * 1_000_003
            + self.config.fixed_eval_seed
        )
        return validate_fixed_eval_pool(
            FixedEvalPool(
                dataset_indices=dataset_indices,
                motion=motion,
                lengths=condition["y"]["lengths"],
                texts=list(condition["y"]["text"]),
                split=self.config.eval_split,
                noise_seed=self.config.fixed_eval_seed,
                prompt_noise_seeds=prompt_noise_seeds,
            )
        )

    def _validate_fixed_eval_pool_config(
        self,
        pool: FixedEvalPool,
    ) -> None:
        expected = {
            "split": self.config.eval_split,
            "prompt_count": self.config.fixed_eval_prompts,
            "noise_seed": self.config.fixed_eval_seed,
        }
        actual = {
            "split": pool.split,
            "prompt_count": pool.prompt_count,
            "noise_seed": pool.noise_seed,
        }
        if actual != expected:
            raise ValueError(
                "Fixed-eval pool does not match current configuration: "
                f"expected={expected}, actual={actual}. Use a fresh output "
                "directory or the matching fixed-eval settings."
            )

    def _load_or_create_fixed_eval_pool(self) -> FixedEvalPool:
        output_path = self.output_dir / "fixed_eval_pool.pt"
        configured_path = (
            Path(self.config.fixed_eval_pool_path).expanduser().resolve()
            if self.config.fixed_eval_pool_path
            else None
        )
        source_path: Path | None = None
        if configured_path is not None and configured_path.exists():
            source_path = configured_path
        elif output_path.exists():
            source_path = output_path
        elif self.config.resume:
            resume_pool_path = (
                Path(self.config.resume).expanduser().resolve().parent
                / "fixed_eval_pool.pt"
            )
            if resume_pool_path.exists():
                source_path = resume_pool_path
            else:
                raise FileNotFoundError(
                    "Cannot resume fixed validation because the original "
                    f"fixed_eval_pool.pt is missing next to {self.config.resume}."
                )

        if source_path is None:
            pool = self._create_fixed_eval_pool()
            if configured_path is not None:
                save_fixed_eval_pool(pool, configured_path)
                LOGGER.info("Created shared fixed-eval pool: %s", configured_path)
        else:
            pool = load_fixed_eval_pool(source_path)
            LOGGER.info("Loaded fixed-eval pool: %s", source_path)

        self._validate_fixed_eval_pool_config(pool)
        if configured_path is not None and not configured_path.exists():
            save_fixed_eval_pool(pool, configured_path)
        if output_path != source_path:
            save_fixed_eval_pool(pool, output_path)
        LOGGER.info(
            "Fixed validation pool: split=%s, prompts=%d, pool_id=%s",
            pool.split,
            pool.prompt_count,
            pool.pool_id,
        )
        return pool

    def _fixed_eval_signature(self) -> dict[str, Any]:
        """Describe every setting that changes the deterministic eval pool."""
        if self.fixed_eval_pool is None:
            raise RuntimeError("Fixed evaluation is disabled.")
        prompt_batch_size = min(
            self.fixed_eval_pool.prompt_count,
            self.config.humanml_fixed_eval_prompts_per_batch,
        )
        precision_code = {"no": 0.0, "fp16": 1.0, "bf16": 2.0}
        signature = {
            "eval_samples": float(
                self.fixed_eval_pool.prompt_count
                * self.config.fixed_eval_samples_per_prompt
            ),
            "eval_prompts": float(self.fixed_eval_pool.prompt_count),
            "eval_samples_per_prompt": float(
                self.config.fixed_eval_samples_per_prompt
            ),
            "eval_seed": float(self.fixed_eval_pool.noise_seed),
            "eval_split": self.fixed_eval_pool.split,
            "eval_pool_id": self.fixed_eval_pool.pool_id,
            "eval_prompt_batch_size": float(prompt_batch_size),
            "eval_batch_size": float(
                prompt_batch_size
                * self.config.fixed_eval_samples_per_prompt
            ),
            "eval_diffusion_steps": float(self.diffusion.num_timesteps),
            "eval_guidance_scale": float(self.config.guidance_scale),
            "eval_ddim_eta": float(self.config.ddim_eta),
            "eval_clip_denoised": float(self.config.clip_denoised),
            "eval_precision_code": precision_code[self.config.precision],
            "eval_allow_tf32": float(self.config.allow_tf32),
            "eval_retrieval_weight": float(self.config.retrieval_weight),
            "eval_m2m_weight": float(self.config.m2m_weight),
        }
        step_pool = getattr(self, "fixed_step_eval_pool", None)
        if step_pool is not None:
            signature.update(
                {
                    "step_eval_samples": float(
                        step_pool.prompt_count
                        * self.config.fixed_step_eval_samples_per_prompt
                    ),
                    "step_eval_prompts": float(
                        step_pool.prompt_count
                    ),
                    "step_eval_samples_per_prompt": float(
                        self.config.fixed_step_eval_samples_per_prompt
                    ),
                    "step_eval_seed": float(
                        step_pool.noise_seed
                    ),
                    "step_eval_pool_id": step_pool.pool_id,
                    "step_eval_detector_backend": (
                        step_pool.detector_backend
                    ),
                    "step_eval_reward_weight": float(
                        self.config.step_reward_weight
                    ),
                    "step_eval_reward_mode": self.config.step_reward_mode,
                    "step_eval_use_m2m_reward": float(
                        self.config.step_use_m2m_reward
                    ),
                    "step_eval_soft_lead_temperature": float(
                        self.config.step_soft_lead_temperature
                    ),
                    "step_eval_soft_length_temperature": float(
                        self.config.step_soft_length_temperature
                    ),
                    "step_eval_soft_progress_temperature": float(
                        self.config.step_soft_progress_temperature
                    ),
                    "step_eval_soft_cluster_gap_seconds": float(
                        self.config.step_soft_cluster_gap_seconds
                    ),
                }
            )
        return signature

    @torch.no_grad()
    def evaluate_fixed_pool(self) -> FixedEvalResult:
        """Evaluate identical prompts and diffusion noise with mean embeddings."""
        if self.fixed_eval_pool is None:
            raise RuntimeError("Fixed evaluation is disabled.")

        prompt_batch_size = self.config.humanml_fixed_eval_prompts_per_batch
        total_prompt_count = self.fixed_eval_pool.prompt_count
        chunk_count = math.ceil(total_prompt_count / prompt_batch_size)
        progress = tqdm(
            total=chunk_count * self.diffusion.num_timesteps,
            desc="fixed evaluation",
            leave=False,
            dynamic_ncols=True,
        )
        reward_totals: list[torch.Tensor] = []
        retrieval_totals: list[torch.Tensor] = []
        m2m_totals: list[torch.Tensor] = []

        for prompt_start in range(0, total_prompt_count, prompt_batch_size):
            prompt_end = min(
                prompt_start + prompt_batch_size,
                total_prompt_count,
            )
            motion, lengths, texts, _ = repeat_prompt_batch(
                self.fixed_eval_pool.motion[prompt_start:prompt_end],
                self.fixed_eval_pool.lengths[prompt_start:prompt_end],
                self.fixed_eval_pool.texts[prompt_start:prompt_end],
                self.config.fixed_eval_samples_per_prompt,
            )
            motion = motion.to(self.device, dtype=torch.float32)
            batch_size, _, _, num_frames = motion.shape
            model_kwargs = build_model_kwargs(
                self.model,
                texts,
                lengths,
                num_frames,
                device=self.device,
                guidance_scale=self.config.guidance_scale,
            )
            noise_parts: list[torch.Tensor] = []
            prompt_generators: list[torch.Generator] = []
            prompt_shape = tuple(
                self.fixed_eval_pool.motion.shape[1:]
            )
            for prompt_index in range(prompt_start, prompt_end):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(
                    int(self.fixed_eval_pool.prompt_noise_seeds[prompt_index])
                )
                prompt_generators.append(generator)
                noise_parts.append(
                    torch.randn(
                        (
                            self.config.fixed_eval_samples_per_prompt,
                            *prompt_shape,
                        ),
                        device=self.device,
                        dtype=motion.dtype,
                        generator=generator,
                    )
                )
            current = torch.cat(noise_parts, dim=0)
            for step in range(self.diffusion.num_timesteps - 1, -1, -1):
                timestep = torch.full(
                    (batch_size,),
                    step,
                    device=self.device,
                    dtype=torch.long,
                )
                transition_noise = torch.cat(
                    [
                        torch.randn(
                            (
                                self.config.fixed_eval_samples_per_prompt,
                                *prompt_shape,
                            ),
                            device=self.device,
                            dtype=motion.dtype,
                            generator=generator,
                        )
                        for generator in prompt_generators
                    ],
                    dim=0,
                )
                with autocast_context(self.device, self.config.precision):
                    current, _, _ = ddim_step_with_logprob(
                        self.diffusion,
                        self.policy_model,
                        current,
                        timestep,
                        model_kwargs=model_kwargs,
                        eta=self.config.ddim_eta,
                        mask=model_kwargs["y"]["mask"],
                        clip_denoised=self.config.clip_denoised,
                        noise=transition_noise,
                    )
                progress.update(1)

            generated_motion = current.squeeze(2).permute(0, 2, 1).contiguous()
            gt_motion = motion.squeeze(2).permute(0, 2, 1).contiguous()
            previous_mode = self.reward_model.embedding_mode
            self.reward_model.embedding_mode = "mean"
            try:
                reward_output = self.reward_model.score(
                    texts=texts,
                    generated_motion=generated_motion,
                    lengths=lengths,
                    gt_motion=gt_motion,
                )
            finally:
                self.reward_model.embedding_mode = previous_mode
            reward_totals.append(reward_output.total.detach().float().cpu())
            retrieval_totals.append(
                reward_output.retrieval.detach().float().cpu()
            )
            m2m_totals.append(reward_output.m2m.detach().float().cpu())

        progress.close()
        rewards = torch.cat(reward_totals)
        retrieval_rewards = torch.cat(retrieval_totals)
        m2m_rewards = torch.cat(m2m_totals)
        expected_samples = (
            self.fixed_eval_pool.prompt_count
            * self.config.fixed_eval_samples_per_prompt
        )
        if len(rewards) != expected_samples:
            raise RuntimeError(
                "Fixed evaluation produced an unexpected sample count: "
                f"expected={expected_samples}, actual={len(rewards)}."
            )

        group_shape = (
            self.fixed_eval_pool.prompt_count,
            self.config.fixed_eval_samples_per_prompt,
        )
        total_by_prompt = rewards.reshape(group_shape)
        retrieval_by_prompt = retrieval_rewards.reshape(group_shape)
        m2m_by_prompt = m2m_rewards.reshape(group_shape)
        return FixedEvalResult(
            metrics={
                "eval_reward_std": rewards.std(unbiased=False).item(),
                **self._fixed_eval_signature(),
            },
            total_per_prompt=total_by_prompt.mean(dim=1),
            retrieval_per_prompt=retrieval_by_prompt.mean(dim=1),
            m2m_per_prompt=m2m_by_prompt.mean(dim=1),
            total_by_prompt=total_by_prompt,
            retrieval_by_prompt=retrieval_by_prompt,
            m2m_by_prompt=m2m_by_prompt,
        )

    @torch.no_grad()
    def evaluate_fixed_step_pool(
        self,
        *,
        calibrating: bool = False,
    ) -> FixedStepEvalResult:
        """Evaluate fixed step prompts with hard and event-level soft counts."""
        if self.fixed_step_eval_pool is None:
            raise RuntimeError("Fixed step evaluation is disabled.")
        if (
            self.step_detector is None
            or self.step_mdm_mean is None
            or self.step_mdm_std is None
        ):
            raise RuntimeError("Step detector is not initialized.")

        prompt_batch_size = self.config.step_fixed_eval_prompts_per_batch
        total_prompt_count = self.fixed_step_eval_pool.prompt_count
        chunk_count = math.ceil(total_prompt_count / prompt_batch_size)
        progress = tqdm(
            total=chunk_count * self.diffusion.num_timesteps,
            desc="fixed step evaluation",
            leave=False,
            dynamic_ncols=True,
        )
        base_total_parts: list[torch.Tensor] = []
        retrieval_parts: list[torch.Tensor] = []
        m2m_parts: list[torch.Tensor] = []
        detected_parts: list[torch.Tensor] = []
        soft_count_parts: list[torch.Tensor] = []
        candidate_count_parts: list[torch.Tensor] = []
        candidate_spacing_parts: list[torch.Tensor] = []
        ankle_high_frequency_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []

        for prompt_start in range(0, total_prompt_count, prompt_batch_size):
            prompt_end = min(prompt_start + prompt_batch_size, total_prompt_count)
            motion, lengths, texts, _ = repeat_prompt_batch(
                self.fixed_step_eval_pool.motion[prompt_start:prompt_end],
                self.fixed_step_eval_pool.lengths[prompt_start:prompt_end],
                self.fixed_step_eval_pool.texts[prompt_start:prompt_end],
                self.config.fixed_step_eval_samples_per_prompt,
            )
            target_steps = self.fixed_step_eval_pool.target_steps[
                prompt_start:prompt_end
            ].repeat_interleave(self.config.fixed_step_eval_samples_per_prompt)
            motion = motion.to(self.device, dtype=torch.float32)
            batch_size, _, _, num_frames = motion.shape
            model_kwargs = build_model_kwargs(
                self.model,
                texts,
                lengths,
                num_frames,
                device=self.device,
                guidance_scale=self.config.guidance_scale,
                target_steps=target_steps,
            )
            prompt_shape = tuple(self.fixed_step_eval_pool.motion.shape[1:])
            prompt_generators: list[torch.Generator] = []
            noise_parts: list[torch.Tensor] = []
            for prompt_index in range(prompt_start, prompt_end):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(
                    int(
                        self.fixed_step_eval_pool.prompt_noise_seeds[
                            prompt_index
                        ]
                    )
                )
                prompt_generators.append(generator)
                noise_parts.append(
                    torch.randn(
                        (
                            self.config.fixed_step_eval_samples_per_prompt,
                            *prompt_shape,
                        ),
                        device=self.device,
                        dtype=motion.dtype,
                        generator=generator,
                    )
                )
            current = torch.cat(noise_parts, dim=0)
            for step in range(self.diffusion.num_timesteps - 1, -1, -1):
                timestep = torch.full(
                    (batch_size,),
                    step,
                    device=self.device,
                    dtype=torch.long,
                )
                transition_noise = torch.cat(
                    [
                        torch.randn(
                            (
                                self.config.fixed_step_eval_samples_per_prompt,
                                *prompt_shape,
                            ),
                            device=self.device,
                            dtype=motion.dtype,
                            generator=generator,
                        )
                        for generator in prompt_generators
                    ],
                    dim=0,
                )
                with autocast_context(self.device, self.config.precision):
                    current, _, _ = ddim_step_with_logprob(
                        self.diffusion,
                        self.policy_model,
                        current,
                        timestep,
                        model_kwargs=model_kwargs,
                        eta=self.config.ddim_eta,
                        mask=model_kwargs["y"]["mask"],
                        clip_denoised=self.config.clip_denoised,
                        noise=transition_noise,
                    )
                progress.update(1)

            generated_motion = current.squeeze(2).permute(0, 2, 1).contiguous()
            gt_motion = motion.squeeze(2).permute(0, 2, 1).contiguous()
            previous_mode = self.reward_model.embedding_mode
            self.reward_model.embedding_mode = "mean"
            try:
                base_reward = self.reward_model.score(
                    texts=texts,
                    generated_motion=generated_motion,
                    lengths=lengths,
                    gt_motion=gt_motion,
                )
            finally:
                self.reward_model.embedding_mode = previous_mode
            base_reward = apply_step_m2m_policy(
                base_reward,
                step_mask=torch.ones_like(target_steps, dtype=torch.bool),
                m2m_weight=self.config.m2m_weight,
                enabled=self.config.step_use_m2m_reward,
            )
            detection = self.step_detector.detect_normalized(
                generated_motion,
                lengths,
                mean=self.step_mdm_mean,
                std=self.step_mdm_std,
            )
            base_total_parts.append(base_reward.total.detach().float().cpu())
            retrieval_parts.append(base_reward.retrieval.detach().float().cpu())
            m2m_parts.append(base_reward.m2m.detach().float().cpu())
            detected_parts.append(detection.hard_count.detach().long().cpu())
            soft_count_parts.append(detection.soft_count.detach().float().cpu())
            candidate_count_parts.append(
                detection.candidate_count.detach().float().cpu()
            )
            candidate_spacing_parts.append(
                detection.candidate_spacing_mean.detach().float().cpu()
            )
            ankle_high_frequency_parts.append(
                detection.ankle_high_frequency_ratio.detach().float().cpu()
            )
            target_parts.append(target_steps.detach().long().cpu())

        progress.close()
        base_total = torch.cat(base_total_parts)
        retrieval = torch.cat(retrieval_parts)
        m2m = torch.cat(m2m_parts)
        detected_steps = torch.cat(detected_parts)
        soft_count = torch.cat(soft_count_parts)
        candidate_count = torch.cat(candidate_count_parts)
        candidate_spacing = torch.cat(candidate_spacing_parts)
        ankle_high_frequency = torch.cat(ankle_high_frequency_parts)
        target_steps = torch.cat(target_parts)
        target_error_scales: dict[str, float] | None = None
        target_scale: torch.Tensor | None = None
        if self.config.step_reward_mode == "soft_huber_exact":
            calibration = getattr(self, "step_reward_calibration", None)
            if calibration is not None and not calibrating:
                target_scale = calibration.target_error_scales(target_steps)
                target_error_scales = {
                    str(target): float(
                        calibration.target_error_scales(
                            torch.tensor([target])
                        )[0]
                    )
                    for target in self.config.step_target_values
                }
            elif calibrating:
                target_error_scales = compute_target_error_scales(
                    soft_count,
                    target_steps,
                    minimum_scale=self.config.step_soft_target_scale_floor,
                )
                target_scale = torch.tensor(
                    [target_error_scales[str(int(value))] for value in target_steps],
                    dtype=torch.float32,
                )
            else:
                raise RuntimeError(
                    "Soft fixed step evaluation requires calibrated target scales."
                )
        step_output = compute_step_count_reward(
            detected_steps,
            target_steps,
            mode=self.config.step_reward_mode,
            temperature=self.config.step_reward_temperature,
            linear_tolerance=self.config.step_reward_linear_tolerance,
            soft_count=soft_count,
            target_scale=target_scale,
            huber_delta=self.config.step_soft_huber_delta,
            exact_bonus=self.config.step_soft_exact_bonus,
        )
        step_reward = step_output.reward.detach().float().cpu()
        total = base_total + self.config.step_reward_weight * step_reward
        group_shape = (
            self.fixed_step_eval_pool.prompt_count,
            self.config.fixed_step_eval_samples_per_prompt,
        )
        total_by_prompt = total.reshape(group_shape)
        retrieval_by_prompt = retrieval.reshape(group_shape)
        m2m_by_prompt = m2m.reshape(group_shape)
        step_reward_by_prompt = step_reward.reshape(group_shape)
        detected_by_prompt = detected_steps.reshape(group_shape)
        soft_count_by_prompt = soft_count.reshape(group_shape)
        candidate_count_by_prompt = candidate_count.reshape(group_shape)
        candidate_spacing_by_prompt = candidate_spacing.reshape(group_shape)
        ankle_high_frequency_by_prompt = ankle_high_frequency.reshape(
            group_shape
        )
        target_by_prompt = target_steps.reshape(group_shape)
        error_by_prompt = (detected_by_prompt - target_by_prompt).abs().float()
        soft_error_by_prompt = soft_count_by_prompt - target_by_prompt.float()
        return FixedStepEvalResult(
            metrics={
                "step_eval_reward_std": step_reward.std(unbiased=False).item(),
                "step_eval_detected_mean": detected_steps.float().mean().item(),
                "step_eval_target_mean": target_steps.float().mean().item(),
                "step_eval_soft_count_mean": soft_count.mean().item(),
                "step_eval_soft_count_error_mean": (
                    soft_count.sub(target_steps.float()).mean().item()
                ),
                "step_eval_soft_count_mae": (
                    soft_count.sub(target_steps.float()).abs().mean().item()
                ),
                "step_eval_soft_hard_count_difference_mean": (
                    soft_count.sub(detected_steps.float()).abs().mean().item()
                ),
                "step_eval_candidate_count_mean": candidate_count.mean().item(),
                "step_eval_candidate_spacing_mean": (
                    candidate_spacing.mean().item()
                ),
                "step_eval_ankle_high_frequency_ratio": (
                    ankle_high_frequency.mean().item()
                ),
                "step_eval_samples": float(len(step_reward)),
                "step_eval_prompts": float(
                    self.fixed_step_eval_pool.prompt_count
                ),
                "step_eval_samples_per_prompt": float(
                    self.config.fixed_step_eval_samples_per_prompt
                ),
                "step_eval_seed": float(
                    self.fixed_step_eval_pool.noise_seed
                ),
                "step_eval_pool_id": self.fixed_step_eval_pool.pool_id,
                "step_eval_detector_backend": (
                    self.fixed_step_eval_pool.detector_backend
                ),
                "step_eval_reward_weight": float(
                    self.config.step_reward_weight
                ),
                "step_eval_reward_mode": self.config.step_reward_mode,
                "step_eval_use_m2m_reward": float(
                    self.config.step_use_m2m_reward
                ),
            },
            total_per_prompt=total_by_prompt.mean(dim=1),
            retrieval_per_prompt=retrieval_by_prompt.mean(dim=1),
            m2m_per_prompt=m2m_by_prompt.mean(dim=1),
            step_reward_per_prompt=step_reward_by_prompt.mean(dim=1),
            exact_per_prompt=(error_by_prompt == 0).float().mean(dim=1),
            within_one_per_prompt=(error_by_prompt <= 1).float().mean(dim=1),
            mae_per_prompt=error_by_prompt.mean(dim=1),
            detected_mean_per_prompt=detected_by_prompt.float().mean(dim=1),
            soft_count_mean_per_prompt=soft_count_by_prompt.mean(dim=1),
            soft_error_mean_per_prompt=soft_error_by_prompt.mean(dim=1),
            soft_mae_per_prompt=soft_error_by_prompt.abs().mean(dim=1),
            candidate_count_mean_per_prompt=(
                candidate_count_by_prompt.mean(dim=1)
            ),
            candidate_spacing_mean_per_prompt=(
                candidate_spacing_by_prompt.mean(dim=1)
            ),
            ankle_high_frequency_ratio_per_prompt=(
                ankle_high_frequency_by_prompt.mean(dim=1)
            ),
            total_by_prompt=total_by_prompt,
            retrieval_by_prompt=retrieval_by_prompt,
            m2m_by_prompt=m2m_by_prompt,
            step_reward_by_prompt=step_reward_by_prompt,
            detected_steps_by_prompt=detected_by_prompt,
            soft_count_by_prompt=soft_count_by_prompt,
            candidate_count_by_prompt=candidate_count_by_prompt,
            candidate_spacing_by_prompt=candidate_spacing_by_prompt,
            ankle_high_frequency_ratio_by_prompt=(
                ankle_high_frequency_by_prompt
            ),
            target_error_scales=target_error_scales,
        )

    def _next_batch(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.data_iterator is None:
            self.data_iterator = iter(self.data_loader)
        try:
            human_motion, human_condition = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            human_motion, human_condition = next(self.data_iterator)
        human_count = human_motion.shape[0]
        expected_human_count = self.config.humanml_prompts_per_rollout_batch
        if human_count != expected_human_count:
            raise RuntimeError(
                "HumanML rollout loader returned an unexpected prompt count: "
                f"expected={expected_human_count}, actual={human_count}."
            )
        human_condition["y"]["target_steps"] = torch.full(
            (human_count,),
            -1,
            dtype=torch.long,
        )
        human_condition["y"]["step_mask"] = torch.zeros(
            human_count,
            dtype=torch.bool,
        )
        if not self.config.enable_step_reward:
            return human_motion, human_condition
        if self.step_data_loader is None:
            raise RuntimeError("Step data loader is missing.")
        if self.step_data_iterator is None:
            self.step_data_iterator = iter(self.step_data_loader)
        try:
            step_motion, step_condition = next(self.step_data_iterator)
        except StopIteration:
            self.step_data_iterator = iter(self.step_data_loader)
            step_motion, step_condition = next(self.step_data_iterator)
        expected_step_count = self.config.step_prompts_per_rollout_batch
        if step_motion.shape[0] != expected_step_count:
            raise RuntimeError(
                "Step rollout loader returned an unexpected prompt count: "
                f"expected={expected_step_count}, actual={step_motion.shape[0]}."
            )
        target_frames = human_motion.shape[-1]
        if step_motion.shape[-1] > target_frames:
            step_motion = step_motion[..., :target_frames]
            step_condition["y"]["lengths"] = step_condition["y"][
                "lengths"
            ].clamp_max(target_frames)
        elif step_motion.shape[-1] < target_frames:
            padding = target_frames - step_motion.shape[-1]
            step_motion = torch.nn.functional.pad(step_motion, (0, padding))
        motion = torch.cat([human_motion, step_motion], dim=0)
        lengths = torch.cat(
            [
                human_condition["y"]["lengths"],
                step_condition["y"]["lengths"],
            ]
        )
        target_steps = torch.cat(
            [
                human_condition["y"]["target_steps"],
                step_condition["y"]["target_steps"],
            ]
        )
        step_mask = torch.cat(
            [
                human_condition["y"]["step_mask"],
                step_condition["y"]["step_mask"],
            ]
        )
        texts = list(human_condition["y"]["text"]) + list(
            step_condition["y"]["text"]
        )
        permutation = torch.randperm(len(motion))
        return motion[permutation], {
            "y": {
                "lengths": lengths[permutation],
                "text": [texts[index] for index in permutation.tolist()],
                "target_steps": target_steps[permutation],
                "step_mask": step_mask[permutation],
            }
        }

    @torch.no_grad()
    def _rollout_batch(self, epoch: int, batch_index: int) -> Trajectory:
        motion, condition = self._next_batch()
        lengths = condition["y"]["lengths"].long()
        texts = list(condition["y"]["text"])
        prompt_target_steps = condition["y"]["target_steps"].long()
        prompt_step_mask = condition["y"]["step_mask"].bool()
        prompt_sample_counts = torch.where(
            prompt_step_mask,
            torch.full_like(
                prompt_target_steps,
                self.config.step_samples_per_prompt,
            ),
            torch.full_like(
                prompt_target_steps,
                self.config.human_samples_per_prompt,
            ),
        )
        motion, lengths, texts, prompt_ids = repeat_prompt_batch(
            motion,
            lengths,
            texts,
            prompt_sample_counts,
        )
        target_steps = prompt_target_steps.repeat_interleave(prompt_sample_counts)
        step_mask = prompt_step_mask.repeat_interleave(prompt_sample_counts)
        if motion.shape[0] != self.config.rollout_batch_size:
            raise RuntimeError(
                "Mixed rollout assembly produced an unexpected motion count: "
                f"expected={self.config.rollout_batch_size}, "
                f"actual={motion.shape[0]}."
            )
        motion = motion.to(
            self.device,
            dtype=torch.float32,
            non_blocking=self.config.pin_memory,
        )
        batch_size, _, _, num_frames = motion.shape
        model_kwargs = build_model_kwargs(
            self.model,
            texts,
            lengths,
            num_frames,
            device=self.device,
            guidance_scale=self.config.guidance_scale,
            target_steps=target_steps,
        )
        motion_mask = model_kwargs["y"]["mask"]
        cached_text_embeddings = split_text_embeddings(
            model_kwargs["y"]["text_embed"]
        )

        current = torch.randn_like(motion)
        latents: list[torch.Tensor] = []
        next_latents: list[torch.Tensor] = []
        timesteps: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []

        step_iterator = tqdm(
            range(self.diffusion.num_timesteps - 1, -1, -1),
            desc=f"epoch {epoch} rollout {batch_index}",
            leave=False,
            dynamic_ncols=True,
        )
        for step in step_iterator:
            timestep = torch.full(
                (batch_size,),
                step,
                device=self.device,
                dtype=torch.long,
            )
            with autocast_context(self.device, self.config.precision):
                previous, log_prob, _ = ddim_step_with_logprob(
                    self.diffusion,
                    self.policy_model,
                    current,
                    timestep,
                    model_kwargs=model_kwargs,
                    eta=self.config.ddim_eta,
                    mask=motion_mask,
                    clip_denoised=self.config.clip_denoised,
                )
            # t=0 is deterministic and is deliberately absent from PPO.
            if step > 0:
                latents.append(current.detach().cpu())
                next_latents.append(previous.detach().cpu())
                timesteps.append(timestep.detach().cpu())
                log_probs.append(log_prob.detach().float().cpu())
            current = previous

        generated_motion = current.squeeze(2).permute(0, 2, 1).contiguous()
        gt_motion = motion.squeeze(2).permute(0, 2, 1).contiguous()
        reward_output: RewardOutput = self.reward_model.score(
            texts=texts,
            generated_motion=generated_motion,
            lengths=lengths,
            gt_motion=gt_motion,
        )
        reward_output = apply_step_m2m_policy(
            reward_output,
            step_mask=step_mask.to(self.device),
            m2m_weight=self.config.m2m_weight,
            enabled=self.config.step_use_m2m_reward,
        )
        step_reward = torch.zeros(batch_size, device=self.device)
        detected_steps = torch.full(
            (batch_size,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        step_absolute_error = torch.full_like(detected_steps, -1)
        soft_step_count = torch.full(
            (batch_size,),
            -1.0,
            device=self.device,
        )
        soft_step_error = torch.full_like(soft_step_count, float("nan"))
        raw_candidate_count = torch.full_like(detected_steps, -1)
        candidate_count = torch.full_like(detected_steps, -1)
        candidate_spacing_mean = torch.zeros_like(soft_step_count)
        candidate_spacing_min = torch.zeros_like(soft_step_count)
        ankle_high_frequency_ratio = torch.zeros_like(soft_step_count)
        if step_mask.any():
            if (
                self.step_detector is None
                or self.step_mdm_mean is None
                or self.step_mdm_std is None
            ):
                raise RuntimeError("Step detector is not initialized.")
            active = step_mask.to(self.device)
            active_detection = self.step_detector.detect_normalized(
                generated_motion[active],
                lengths[step_mask],
                mean=self.step_mdm_mean,
                std=self.step_mdm_std,
            )
            active_targets = target_steps[step_mask].to(self.device)
            target_scale = (
                self.step_reward_calibration.target_error_scales(
                    active_targets
                )
                if self.config.step_reward_mode == "soft_huber_exact"
                and self.step_reward_calibration is not None
                else None
            )
            active_output = compute_step_count_reward(
                active_detection.hard_count,
                active_targets,
                mode=self.config.step_reward_mode,
                temperature=self.config.step_reward_temperature,
                linear_tolerance=self.config.step_reward_linear_tolerance,
                soft_count=active_detection.soft_count,
                target_scale=target_scale,
                huber_delta=self.config.step_soft_huber_delta,
                exact_bonus=self.config.step_soft_exact_bonus,
            )
            step_reward[active] = active_output.reward
            detected_steps[active] = active_output.detected_steps
            step_absolute_error[active] = active_output.absolute_error
            soft_step_count[active] = active_detection.soft_count
            soft_step_error[active] = (
                active_detection.soft_count - active_targets.float()
            )
            raw_candidate_count[active] = (
                active_detection.raw_candidate_count
            )
            candidate_count[active] = active_detection.candidate_count
            candidate_spacing_mean[active] = (
                active_detection.candidate_spacing_mean
            )
            candidate_spacing_min[active] = (
                active_detection.candidate_spacing_min
            )
            ankle_high_frequency_ratio[active] = (
                active_detection.ankle_high_frequency_ratio
            )
        reward_output = add_step_reward(
            reward_output,
            step=step_reward,
            step_mask=step_mask.to(self.device),
            detected_steps=detected_steps,
            target_steps=target_steps.to(self.device),
            absolute_error=step_absolute_error,
            step_weight=self.config.step_reward_weight,
            soft_step_count=soft_step_count,
            soft_step_error=soft_step_error,
            raw_candidate_count=raw_candidate_count,
            candidate_count=candidate_count,
            candidate_spacing_mean=candidate_spacing_mean,
            candidate_spacing_min=candidate_spacing_min,
            ankle_high_frequency_ratio=ankle_high_frequency_ratio,
        )

        return Trajectory(
            latents=torch.stack(latents, dim=1),
            next_latents=torch.stack(next_latents, dim=1),
            timesteps=torch.stack(timesteps, dim=1),
            old_log_probs=torch.stack(log_probs, dim=1),
            rewards=reward_output.total.detach().float().cpu(),
            retrieval_rewards=reward_output.retrieval.detach().float().cpu(),
            m2m_rewards=reward_output.m2m.detach().float().cpu(),
            texts=texts,
            text_embeddings=cached_text_embeddings,
            lengths=lengths.detach().cpu(),
            gt_motion=gt_motion.detach().float().cpu(),
            prompt_ids=prompt_ids,
            step_rewards=reward_output.step.detach().float().cpu(),
            step_mask=reward_output.step_mask.detach().cpu(),
            detected_steps=reward_output.detected_steps.detach().cpu(),
            target_steps=reward_output.target_steps.detach().cpu(),
            step_absolute_error=(
                reward_output.step_absolute_error.detach().cpu()
            ),
            soft_step_count=reward_output.soft_step_count.detach().cpu(),
            soft_step_error=reward_output.soft_step_error.detach().cpu(),
            step_raw_candidate_count=(
                reward_output.step_raw_candidate_count.detach().cpu()
            ),
            step_candidate_count=(
                reward_output.step_candidate_count.detach().cpu()
            ),
            step_candidate_spacing_mean=(
                reward_output.step_candidate_spacing_mean.detach().cpu()
            ),
            step_candidate_spacing_min=(
                reward_output.step_candidate_spacing_min.detach().cpu()
            ),
            step_ankle_high_frequency_ratio=(
                reward_output.step_ankle_high_frequency_ratio.detach().cpu()
            ),
        )

    def collect_rollouts(self, epoch: int) -> Trajectory:
        self.model.eval()
        # A fresh randomized subset per DDPO epoch also makes epoch-boundary
        # resume reproducible from the DataLoader generator state.
        self.data_iterator = iter(self.data_loader)
        if self.step_data_loader is not None:
            self.step_data_iterator = iter(self.step_data_loader)
        batches = [
            self._rollout_batch(epoch, batch_index)
            for batch_index in range(self.config.rollout_batches_per_epoch)
        ]
        trajectory = Trajectory.concatenate(batches)
        if self.config.advantage_mode == "component_shrink":
            trajectory.advantages, component_stats = (
                compute_component_shrink_advantages(
                    trajectory.retrieval_rewards,
                    trajectory.m2m_rewards,
                    trajectory.prompt_ids,
                    epsilon=self.config.advantage_epsilon,
                    retrieval_std_floor=self._advantage_std_floor(
                        "retrieval"
                    ),
                    m2m_std_floor=self._advantage_std_floor("m2m"),
                    retrieval_weight=(
                        self.config.advantage_retrieval_weight
                    ),
                    m2m_weight=self.config.advantage_m2m_weight,
                    step_retrieval_weight=(
                        self.config.effective_step_advantage_retrieval_weight
                    ),
                    step_m2m_weight=(
                        self.config.effective_step_advantage_m2m_weight
                    ),
                    step_rewards=trajectory.step_rewards,
                    step_mask=trajectory.step_mask,
                    step_std_floor=(
                        self._advantage_std_floor("step")
                        if self.config.enable_step_reward
                        and self.config.effective_step_advantage_step_weight > 0
                        else None
                    ),
                    step_weight=(
                        self.config.effective_step_advantage_step_weight
                        if self.config.enable_step_reward
                        else 0.0
                    ),
                    step_use_m2m_reward=(
                        self.config.step_use_m2m_reward
                        if self.config.enable_step_reward
                        else True
                    ),
                )
            )
            _, total_stats = compute_grouped_advantages(
                trajectory.rewards,
                trajectory.prompt_ids,
                self.config.advantage_epsilon,
                mode="group_centered",
            )
            trajectory.group_stats = {**total_stats, **component_stats}
        else:
            trajectory.advantages, trajectory.group_stats = (
                compute_grouped_advantages(
                    trajectory.rewards,
                    trajectory.prompt_ids,
                    self.config.advantage_epsilon,
                    self.config.advantage_mode,
                    std_floor=(
                        self._advantage_std_floor("total")
                        if self.config.advantage_mode == "group_shrink"
                        else None
                    ),
                )
            )
        if trajectory.group_stats["zero_variance_prompt_fraction"] > 0:
            LOGGER.warning(
                "%.1f%% of prompt groups have effectively zero reward variance.",
                100.0 * trajectory.group_stats["zero_variance_prompt_fraction"],
            )
        return trajectory

    def _advantage_std_floor(self, component: str) -> float:
        if component == "step":
            if self.step_reward_calibration is None:
                raise RuntimeError(
                    "A loaded step reward calibration is required for "
                    "step component shrinkage."
                )
            return self.step_reward_calibration.within_group_std_floor(
                self.config.advantage_std_floor_quantile
            )
        if self.reward_calibration is None:
            raise RuntimeError(
                "A loaded reward calibration is required for shrinkage."
            )
        return self.reward_calibration.within_group_std_floor(
            component,
            self.config.advantage_std_floor_quantile,
        )

    def _selected_timesteps(
        self,
        num_samples: int,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, int]:
        per_sample = max(
            1,
            int(math.ceil(num_timesteps * self.config.timestep_fraction)),
        )
        selected = torch.stack(
            [
                torch.randperm(num_timesteps)[:per_sample]
                for _ in range(num_samples)
            ],
            dim=0,
        )
        return selected, per_sample

    @torch.no_grad()
    def _audit_first_update_log_probs(
        self,
        trajectory: Trajectory,
        sample_indices: torch.Tensor,
        selected_timesteps: torch.Tensor,
        model_kwargs: dict[str, Any],
    ) -> dict[str, float]:
        old_parts: list[torch.Tensor] = []
        new_parts: list[torch.Tensor] = []
        for timestep_position in range(selected_timesteps.shape[1]):
            time_indices = selected_timesteps[:, timestep_position]
            current = trajectory.latents[sample_indices, time_indices].to(
                self.device,
                non_blocking=self.config.pin_memory,
            )
            previous = trajectory.next_latents[sample_indices, time_indices].to(
                self.device,
                non_blocking=self.config.pin_memory,
            )
            timesteps = trajectory.timesteps[sample_indices, time_indices].to(
                self.device,
                non_blocking=self.config.pin_memory,
            )
            old_log_probs = trajectory.old_log_probs[
                sample_indices, time_indices
            ].to(self.device, non_blocking=self.config.pin_memory)
            with autocast_context(self.device, self.config.precision):
                _, new_log_probs, _ = ddim_step_with_logprob(
                    self.diffusion,
                    self.policy_model,
                    current,
                    timesteps,
                    model_kwargs=model_kwargs,
                    eta=self.config.ddim_eta,
                    prev_sample=previous,
                    mask=model_kwargs["y"]["mask"],
                    clip_denoised=self.config.clip_denoised,
                )
            old_parts.append(old_log_probs.detach().float().cpu())
            new_parts.append(new_log_probs.detach().float().cpu())
        return log_prob_consistency_metrics(
            torch.cat(old_parts),
            torch.cat(new_parts),
            self.config.log_prob_audit_tolerance,
        )

    def _anchor_sample_indices(
        self,
        trajectory: Trajectory,
        group_sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        target_count = (
            self.config.anchor_batch_size
            or self.config.train_batch_size
        )
        selected: list[int] = []
        seen_prompt_ids: set[int] = set()
        candidate_indices = group_sample_indices.tolist()
        if trajectory.step_mask is not None:
            group_index_set = set(candidate_indices)
            candidate_indices.extend(
                index
                for index in range(len(trajectory.prompt_ids))
                if index not in group_index_set
            )
        for sample_index in candidate_indices:
            if (
                trajectory.step_mask is not None
                and bool(trajectory.step_mask[sample_index])
            ):
                continue
            prompt_id = int(trajectory.prompt_ids[sample_index])
            if prompt_id in seen_prompt_ids:
                continue
            seen_prompt_ids.add(prompt_id)
            selected.append(sample_index)
            if len(selected) >= target_count:
                break
        if not selected:
            raise RuntimeError("Anchor update found no distinct real motions.")
        return torch.tensor(selected, dtype=torch.long)

    def _compute_anchor_loss(
        self,
        trajectory: Trajectory,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.anchor_model is None:
            raise RuntimeError("The MDM diffusion anchor is disabled.")
        gt_motion = trajectory.gt_motion[sample_indices].to(
            self.device,
            dtype=torch.float32,
            non_blocking=self.config.pin_memory,
        )
        motion = gt_motion.permute(0, 2, 1).unsqueeze(2).contiguous()
        lengths = trajectory.lengths[sample_indices]
        index_values = sample_indices.tolist()
        texts = [trajectory.texts[index] for index in index_values]
        cached_text_embeddings = [
            trajectory.text_embeddings[index] for index in index_values
        ]
        model_kwargs = build_model_kwargs(
            self.model,
            texts,
            lengths,
            motion.shape[-1],
            device=self.device,
            guidance_scale=1.0,
            cached_text_embeddings=cached_text_embeddings,
        )
        timesteps = torch.randint(
            0,
            self.base_diffusion.num_timesteps,
            (motion.shape[0],),
            device=self.device,
        )
        self.anchor_model.train()
        try:
            with autocast_context(self.device, self.config.precision):
                terms = self.base_diffusion.training_losses(
                    self.anchor_model,
                    motion,
                    timesteps,
                    model_kwargs=model_kwargs,
                    dataset=self.data_loader.dataset,
                )
                anchor_loss = terms["loss"].mean()
        finally:
            self.anchor_model.eval()
            self.model.eval()
            self.policy_model.eval()
        if not torch.isfinite(anchor_loss):
            raise FloatingPointError("Native MDM anchor loss is non-finite.")
        return anchor_loss

    def _add_anchor_gradients(
        self,
        trajectory: Trajectory,
        group_sample_indices: torch.Tensor,
        trainable_parameters: list[torch.Tensor],
    ) -> dict[str, float]:
        scale = (
            float(self.scaler.get_scale())
            if self.scaler.is_enabled()
            else 1.0
        )
        ppo_grad_norm = gradient_l2_norm(
            [parameter.grad for parameter in trainable_parameters],
            scale=scale,
        )
        anchor_indices = self._anchor_sample_indices(
            trajectory,
            group_sample_indices,
        )
        anchor_loss = self._compute_anchor_loss(
            trajectory,
            anchor_indices,
        )
        anchor_gradients = list(
            torch.autograd.grad(
                anchor_loss,
                trainable_parameters,
                allow_unused=True,
            )
        )
        anchor_grad_norm = gradient_l2_norm(anchor_gradients)
        if not math.isfinite(anchor_grad_norm):
            raise FloatingPointError("Native MDM anchor gradients are non-finite.")
        if (
            self.config.anchor_auto_grad_ratio > 0
            and not self.anchor_lambda_calibrated
        ):
            self.anchor_lambda_effective = calibrate_anchor_lambda(
                ppo_grad_norm,
                anchor_grad_norm,
                self.config.anchor_auto_grad_ratio,
            )
            self.anchor_lambda_calibrated = True
            LOGGER.info(
                "Calibrated anchor lambda=%.8g for target grad ratio=%.4f "
                "(ppo_grad_norm=%.6g, anchor_grad_norm=%.6g).",
                self.anchor_lambda_effective,
                self.config.anchor_auto_grad_ratio,
                ppo_grad_norm,
                anchor_grad_norm,
            )
        weighted_grad_norm = (
            self.anchor_lambda_effective * anchor_grad_norm
        )
        with torch.no_grad():
            for parameter, anchor_gradient in zip(
                trainable_parameters,
                anchor_gradients,
            ):
                if anchor_gradient is None:
                    continue
                contribution = (
                    anchor_gradient.detach()
                    * self.anchor_lambda_effective
                    * scale
                )
                if parameter.grad is None:
                    parameter.grad = contribution.clone()
                else:
                    parameter.grad.add_(contribution)
        return {
            "anchor_loss": anchor_loss.detach().float().item(),
            "anchor_weighted_loss": (
                anchor_loss.detach().float().item()
                * self.anchor_lambda_effective
            ),
            "anchor_grad_norm": anchor_grad_norm,
            "anchor_weighted_grad_norm": weighted_grad_norm,
            "ppo_grad_norm": ppo_grad_norm,
            "anchor_grad_ratio": weighted_grad_norm / max(ppo_grad_norm, 1.0e-12),
            "anchor_lambda": self.anchor_lambda_effective,
            "anchor_batch_samples": float(len(anchor_indices)),
            "anchor_calls": 1.0,
        }

    def optimize(self, trajectory: Trajectory) -> dict[str, float]:
        if trajectory.advantages is None:
            raise ValueError("Advantages must be computed before optimization.")
        self.model.eval()
        num_samples, num_timesteps = trajectory.timesteps.shape
        metric_values: dict[str, list[float]] = {
            "loss": [],
            "grad_norm": [],
            "update_norm": [],
            "lora_update_norm": [],
            "count_update_norm": [],
            "skipped_updates": [],
        }
        if self.anchor_enabled:
            for name in (
                "anchor_loss",
                "anchor_weighted_loss",
                "anchor_grad_norm",
                "anchor_weighted_grad_norm",
                "ppo_grad_norm",
                "anchor_grad_ratio",
                "anchor_lambda",
                "anchor_batch_samples",
                "anchor_calls",
            ):
                metric_values[name] = []
        log_ratio_parts: list[torch.Tensor] = []
        ratio_parts: list[torch.Tensor] = []
        audit_metrics: dict[str, float] | None = None
        audit_records: list[dict[str, float]] = []

        self.optimizer.zero_grad(set_to_none=True)
        named_trainable_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        trainable_parameters = [
            parameter for _, parameter in named_trainable_parameters
        ]
        lora_parameters = [
            parameter
            for name, parameter in named_trainable_parameters
            if "lora_a" in name or "lora_b" in name
        ]
        count_parameters = [
            parameter
            for name, parameter in named_trainable_parameters
            if name.startswith("count_conditioning.")
        ]
        for inner_epoch in range(self.config.inner_epochs):
            selected_timesteps, timesteps_per_sample = self._selected_timesteps(
                num_samples,
                num_timesteps,
            )
            sample_minibatches = shuffled_sample_minibatches(
                num_samples,
                self.config.train_batch_size,
            )
            num_sample_minibatches = len(sample_minibatches)
            progress = tqdm(
                total=num_sample_minibatches * timesteps_per_sample,
                desc=f"DDPO inner epoch {inner_epoch}",
                leave=False,
                dynamic_ncols=True,
            )
            for minibatch_position, sample_indices in enumerate(
                sample_minibatches
            ):
                advantages = trajectory.advantages[sample_indices].to(
                    self.device,
                    non_blocking=self.config.pin_memory,
                )
                lengths = trajectory.lengths[sample_indices]
                texts = [
                    trajectory.texts[index]
                    for index in sample_indices.tolist()
                ]
                cached_text_embeddings = [
                    trajectory.text_embeddings[index]
                    for index in sample_indices.tolist()
                ]
                model_kwargs = build_model_kwargs(
                    self.model,
                    texts,
                    lengths,
                    trajectory.latents.shape[-1],
                    device=self.device,
                    guidance_scale=self.config.guidance_scale,
                    cached_text_embeddings=cached_text_embeddings,
                    target_steps=(
                        trajectory.target_steps[sample_indices]
                        if trajectory.target_steps is not None
                        else None
                    ),
                )

                minibatch_timesteps = selected_timesteps[sample_indices]
                first_update_group_end = min(
                    self.config.gradient_accumulation_steps,
                    num_sample_minibatches,
                )
                if (
                    audit_metrics is None
                    and inner_epoch == 0
                    and minibatch_position < first_update_group_end
                ):
                    audit_records.append(self._audit_first_update_log_probs(
                        trajectory,
                        sample_indices,
                        minibatch_timesteps,
                        model_kwargs,
                    ))
                    if minibatch_position + 1 == first_update_group_end:
                        audit_metrics = merge_log_prob_audit_metrics(
                            audit_records
                        )
                        LOGGER.info(
                            "Initial old/new log-probability audit over the "
                            "full first optimizer update: %s",
                            json.dumps(audit_metrics, sort_keys=True),
                        )

                group_start = (
                    minibatch_position
                    // self.config.gradient_accumulation_steps
                ) * self.config.gradient_accumulation_steps
                group_end = min(
                    group_start + self.config.gradient_accumulation_steps,
                    num_sample_minibatches,
                )
                accumulation_divisor = (
                    group_end - group_start
                ) * timesteps_per_sample

                for timestep_position in range(timesteps_per_sample):
                    time_indices = minibatch_timesteps[:, timestep_position]
                    current = trajectory.latents[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)
                    previous = trajectory.next_latents[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)
                    timesteps = trajectory.timesteps[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)
                    old_log_probs = trajectory.old_log_probs[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)

                    with autocast_context(self.device, self.config.precision):
                        _, new_log_probs, _ = ddim_step_with_logprob(
                            self.diffusion,
                            self.policy_model,
                            current,
                            timesteps,
                            model_kwargs=model_kwargs,
                            eta=self.config.ddim_eta,
                            prev_sample=previous,
                            mask=model_kwargs["y"]["mask"],
                            clip_denoised=self.config.clip_denoised,
                        )
                        clipped_advantages = advantages.clamp(
                            -self.config.adv_clip_max,
                            self.config.adv_clip_max,
                        )
                        raw_log_ratio = new_log_probs - old_log_probs
                        if not torch.isfinite(raw_log_ratio).all():
                            raise FloatingPointError(
                                "Non-finite PPO log ratio encountered."
                            )
                        log_ratio = raw_log_ratio.clamp(-20.0, 20.0)
                        ratio = log_ratio.exp()
                        unclipped_loss = -clipped_advantages * ratio
                        clipped_loss = -clipped_advantages * ratio.clamp(
                            1.0 - self.config.clip_range,
                            1.0 + self.config.clip_range,
                        )
                        loss = torch.maximum(
                            unclipped_loss,
                            clipped_loss,
                        ).mean()

                    self.scaler.scale(
                        loss / accumulation_divisor
                    ).backward()

                    metric_values["loss"].append(
                        loss.detach().float().item()
                    )
                    log_ratio_parts.append(
                        raw_log_ratio.detach().float().cpu().reshape(-1)
                    )
                    ratio_parts.append(
                        ratio.detach().float().cpu().reshape(-1)
                    )
                    progress.update(1)

                    should_step = (
                        timestep_position + 1 == timesteps_per_sample
                        and minibatch_position + 1 == group_end
                    )
                    if should_step:
                        if self.anchor_enabled:
                            group_sample_indices = torch.cat(
                                sample_minibatches[group_start:group_end]
                            )
                            anchor_metrics = self._add_anchor_gradients(
                                trajectory,
                                group_sample_indices,
                                trainable_parameters,
                            )
                            for name, value in anchor_metrics.items():
                                metric_values[name].append(value)
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = clip_grad_norm_(
                            trainable_parameters,
                            self.config.max_grad_norm,
                        )
                        finite_gradients = bool(
                            torch.isfinite(grad_norm).item()
                        )
                        if finite_gradients:
                            policy_before_update = [
                                parameter.detach().clone()
                                for parameter in trainable_parameters
                            ]
                            lora_before_update = [
                                parameter.detach().clone()
                                for parameter in lora_parameters
                            ]
                            count_before_update = [
                                parameter.detach().clone()
                                for parameter in count_parameters
                            ]
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.global_step += 1
                            metric_values["grad_norm"].append(
                                float(grad_norm)
                            )
                            metric_values["update_norm"].append(
                                _tensor_collection_l2_norm(
                                    [
                                        parameter.detach() - before
                                        for parameter, before in zip(
                                            trainable_parameters,
                                            policy_before_update,
                                        )
                                    ]
                                )
                            )
                            metric_values["lora_update_norm"].append(
                                _tensor_collection_l2_norm(
                                    [
                                        parameter.detach() - before
                                        for parameter, before in zip(
                                            lora_parameters,
                                            lora_before_update,
                                        )
                                    ]
                                )
                            )
                            metric_values["count_update_norm"].append(
                                _tensor_collection_l2_norm(
                                    [
                                        parameter.detach() - before
                                        for parameter, before in zip(
                                            count_parameters,
                                            count_before_update,
                                        )
                                    ]
                                )
                            )
                            metric_values["skipped_updates"].append(0.0)
                        else:
                            LOGGER.warning(
                                "Skipping an optimizer update because the "
                                "gradient norm is non-finite. Consider "
                                "--precision bf16 or no."
                            )
                            # After unscale_, GradScaler has enough information
                            # to reduce its scale even when step is skipped.
                            self.scaler.update()
                            metric_values["skipped_updates"].append(1.0)
                        self.optimizer.zero_grad(set_to_none=True)
            progress.close()

        if audit_metrics is None or not log_ratio_parts:
            raise RuntimeError("PPO optimization produced no policy transitions.")
        all_log_ratios = torch.cat(log_ratio_parts)
        all_ratios = torch.cat(ratio_parts)
        result = {
            name: (
                float(np.sum(values))
                if name == "anchor_calls"
                else float(np.mean(values))
            ) if values else 0.0
            for name, values in metric_values.items()
        }
        result.update(
            {
                "approx_kl": (
                    0.5 * all_log_ratios.square().mean()
                ).item(),
                "log_ratio_mean": all_log_ratios.mean().item(),
                "log_ratio_std": all_log_ratios.std(unbiased=False).item(),
                "log_ratio_max": all_log_ratios.abs().max().item(),
                "ratio": all_ratios.mean().item(),
                "ratio_std": all_ratios.std(unbiased=False).item(),
                "clip_fraction": (
                    (all_ratios - 1.0).abs() > self.config.clip_range
                ).float().mean().item(),
                "lora_norm": _tensor_collection_l2_norm(lora_parameters),
                "count_norm": _tensor_collection_l2_norm(count_parameters),
                "trainable_parameter_norm": _tensor_collection_l2_norm(
                    trainable_parameters
                ),
                **audit_metrics,
            }
        )
        return result

    def _rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        advantages = trajectory.advantages
        assert advantages is not None
        group_stats = trajectory.group_stats or {}
        result = {
            "reward": trajectory.rewards.mean().item(),
            "reward_std": trajectory.rewards.std(unbiased=False).item(),
            "reward_retrieval": trajectory.retrieval_rewards.mean().item(),
            "reward_m2m": trajectory.m2m_rewards.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "rollout_samples": float(len(trajectory.rewards)),
            "samples_per_prompt": float(self.config.samples_per_prompt),
            "humanml_samples_per_prompt": float(
                self.config.human_samples_per_prompt
            ),
            "step_samples_per_prompt": float(
                self.config.step_samples_per_prompt
            ),
            "humanml_rollout_samples": float(
                self.config.humanml_rollout_samples
            ),
            "step_motion_ratio": (
                self.config.step_rollout_samples
                / self.config.rollout_batch_size
            ),
            "step_use_m2m_reward": float(
                self.config.step_use_m2m_reward
            ),
            **group_stats,
        }
        if trajectory.step_mask is not None and trajectory.step_mask.any():
            assert trajectory.step_rewards is not None
            assert trajectory.detected_steps is not None
            assert trajectory.target_steps is not None
            assert trajectory.step_absolute_error is not None
            assert trajectory.soft_step_count is not None
            assert trajectory.soft_step_error is not None
            assert trajectory.step_raw_candidate_count is not None
            assert trajectory.step_candidate_count is not None
            assert trajectory.step_candidate_spacing_mean is not None
            assert trajectory.step_candidate_spacing_min is not None
            assert trajectory.step_ankle_high_frequency_ratio is not None
            active = trajectory.step_mask.bool()
            errors = trajectory.step_absolute_error[active].float()
            result.update(
                {
                    "reward_step": trajectory.step_rewards[active].mean().item(),
                    "step_exact_fraction": (errors == 0).float().mean().item(),
                    "step_within_one_fraction": (
                        (errors <= 1).float().mean().item()
                    ),
                    "step_mae": errors.mean().item(),
                    "step_detected_mean": (
                        trajectory.detected_steps[active].float().mean().item()
                    ),
                    "step_target_mean": (
                        trajectory.target_steps[active].float().mean().item()
                    ),
                    "step_soft_count_mean": (
                        trajectory.soft_step_count[active].mean().item()
                    ),
                    "step_soft_count_error_mean": (
                        trajectory.soft_step_error[active].mean().item()
                    ),
                    "step_soft_count_mae": (
                        trajectory.soft_step_error[active].abs().mean().item()
                    ),
                    "step_soft_hard_count_difference_mean": (
                        trajectory.soft_step_count[active]
                        .sub(trajectory.detected_steps[active].float())
                        .abs()
                        .mean()
                        .item()
                    ),
                    "step_raw_candidate_count_mean": (
                        trajectory.step_raw_candidate_count[active]
                        .float()
                        .mean()
                        .item()
                    ),
                    "step_candidate_count_mean": (
                        trajectory.step_candidate_count[active]
                        .float()
                        .mean()
                        .item()
                    ),
                    "step_candidate_spacing_mean": (
                        trajectory.step_candidate_spacing_mean[active]
                        .mean()
                        .item()
                    ),
                    "step_candidate_spacing_min_mean": (
                        trajectory.step_candidate_spacing_min[active]
                        .mean()
                        .item()
                    ),
                    "step_ankle_high_frequency_ratio": (
                        trajectory.step_ankle_high_frequency_ratio[active]
                        .mean()
                        .item()
                    ),
                    "step_reward_candidate_count_correlation": (
                        _pearson_tensor(
                            trajectory.step_rewards[active],
                            trajectory.step_candidate_count[active].float(),
                        )
                    ),
                    "step_reward_ankle_high_frequency_correlation": (
                        _pearson_tensor(
                            trajectory.step_rewards[active],
                            trajectory.step_ankle_high_frequency_ratio[active],
                        )
                    ),
                    "step_rollout_samples": float(active.sum()),
                }
            )
            for target in self.config.step_target_values:
                target_samples = float(
                    (trajectory.target_steps[active] == target).sum()
                )
                result[f"step_target_{target}_samples"] = target_samples
                result[f"step_target_{target}_prompt_groups"] = (
                    target_samples / self.config.step_samples_per_prompt
                )
        else:
            result["step_rollout_samples"] = 0.0
        return result

    def _append_metrics(self, record: dict[str, Any]) -> None:
        with open(self.output_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_fixed_eval(self, record: dict[str, Any]) -> None:
        with open(
            self.output_dir / "fixed_eval.jsonl",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_fixed_eval_per_prompt(
        self,
        *,
        event: str,
        epoch: int,
        evaluation: FixedEvalResult,
    ) -> None:
        if self.fixed_eval_pool is None:
            raise RuntimeError("Fixed evaluation is disabled.")
        if self.fixed_eval_baseline_per_prompt is None:
            raise RuntimeError("Fixed evaluation baseline is missing.")
        entries = []
        for index in range(self.fixed_eval_pool.prompt_count):
            retrieval = evaluation.retrieval_per_prompt[index].item()
            m2m = evaluation.m2m_per_prompt[index].item()
            retrieval_baseline = self.fixed_eval_baseline_per_prompt[
                "retrieval"
            ][index].item()
            m2m_baseline = self.fixed_eval_baseline_per_prompt["m2m"][
                index
            ].item()
            entries.append(
                {
                    "pool_position": index,
                    "dataset_index": int(
                        self.fixed_eval_pool.dataset_indices[index]
                    ),
                    "text": self.fixed_eval_pool.texts[index],
                    "length": int(self.fixed_eval_pool.lengths[index]),
                    "retrieval_baseline": retrieval_baseline,
                    "retrieval_current": retrieval,
                    "retrieval_delta": retrieval - retrieval_baseline,
                    "m2m_baseline": m2m_baseline,
                    "m2m_current": m2m,
                    "m2m_delta": m2m - m2m_baseline,
                }
            )
        record = {
            "event": event,
            "epoch": epoch,
            "global_step": self.global_step,
            "eval_split": self.fixed_eval_pool.split,
            "eval_pool_id": self.fixed_eval_pool.pool_id,
            "prompts": entries,
        }
        with open(
            self.output_dir / "fixed_eval_per_prompt.jsonl",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_fixed_step_eval_per_prompt(
        self,
        *,
        event: str,
        epoch: int,
        evaluation: FixedStepEvalResult,
    ) -> None:
        if self.fixed_step_eval_pool is None:
            raise RuntimeError("Fixed step evaluation is disabled.")
        if self.fixed_step_eval_baseline_per_prompt is None:
            raise RuntimeError("Fixed step evaluation baseline is missing.")
        entries = []
        for index in range(self.fixed_step_eval_pool.prompt_count):
            entry: dict[str, Any] = {
                "pool_position": index,
                "manifest_index": int(
                    self.fixed_step_eval_pool.manifest_indices[index]
                ),
                "sample_id": self.fixed_step_eval_pool.sample_ids[index],
                "text": self.fixed_step_eval_pool.texts[index],
                "length": int(self.fixed_step_eval_pool.lengths[index]),
                "target_steps": int(
                    self.fixed_step_eval_pool.target_steps[index]
                ),
            }
            current_values = {
                "total": evaluation.total_per_prompt[index].item(),
                "retrieval": evaluation.retrieval_per_prompt[index].item(),
                "m2m": evaluation.m2m_per_prompt[index].item(),
                "step_reward": evaluation.step_reward_per_prompt[index].item(),
                "exact_fraction": evaluation.exact_per_prompt[index].item(),
                "within_one_fraction": (
                    evaluation.within_one_per_prompt[index].item()
                ),
                "mae": evaluation.mae_per_prompt[index].item(),
                "detected_mean": (
                    evaluation.detected_mean_per_prompt[index].item()
                ),
                "soft_count_mean": (
                    evaluation.soft_count_mean_per_prompt[index].item()
                ),
                "soft_error_mean": (
                    evaluation.soft_error_mean_per_prompt[index].item()
                ),
                "soft_mae": evaluation.soft_mae_per_prompt[index].item(),
                "candidate_count_mean": (
                    evaluation.candidate_count_mean_per_prompt[index].item()
                ),
                "candidate_spacing_mean": (
                    evaluation.candidate_spacing_mean_per_prompt[index].item()
                ),
                "ankle_high_frequency_ratio": (
                    evaluation.ankle_high_frequency_ratio_per_prompt[index].item()
                ),
            }
            for name, current in current_values.items():
                baseline = self._fixed_step_baseline_component(name)[
                    index
                ].item()
                entry[f"{name}_baseline"] = baseline
                entry[f"{name}_current"] = current
                entry[f"{name}_delta"] = current - baseline
            entries.append(entry)
        record = {
            "event": event,
            "epoch": epoch,
            "global_step": self.global_step,
            "eval_split": self.fixed_step_eval_pool.split,
            "eval_pool_id": self.fixed_step_eval_pool.pool_id,
            "detector_backend": self.fixed_step_eval_pool.detector_backend,
            "prompts": entries,
        }
        with open(
            self.output_dir / "fixed_step_eval_per_prompt.jsonl",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _fixed_step_baseline_component(self, name: str) -> torch.Tensor:
        """Read new diagnostics from old fixed-step checkpoint baselines safely."""

        baseline = self.fixed_step_eval_baseline_per_prompt
        if baseline is None:
            raise RuntimeError(
                "Fixed step evaluation baseline has not been initialized."
            )
        if name in baseline:
            return baseline[name]
        detected = baseline["detected_mean"]
        if name in {"soft_count_mean", "candidate_count_mean"}:
            return detected
        if name == "soft_mae":
            return baseline["mae"]
        if name == "soft_error_mean":
            pool = getattr(self, "fixed_step_eval_pool", None)
            if pool is not None and len(pool.target_steps) == len(detected):
                return detected - pool.target_steps.detach().float().cpu()
            return torch.zeros_like(detected)
        if name in {
            "candidate_spacing_mean",
            "ankle_high_frequency_ratio",
        }:
            return torch.zeros_like(detected)
        raise KeyError(f"Fixed step baseline has no component {name!r}.")

    def _fixed_eval_with_deltas(
        self,
        evaluation: FixedEvalResult,
    ) -> dict[str, Any]:
        if self.fixed_eval_baseline_per_prompt is None:
            raise RuntimeError("Fixed evaluation baseline has not been initialized.")
        result = dict(evaluation.metrics)
        components = (
            (
                "eval_reward",
                evaluation.total_per_prompt,
                self.fixed_eval_baseline_per_prompt["total"],
            ),
            (
                "eval_reward_retrieval",
                evaluation.retrieval_per_prompt,
                self.fixed_eval_baseline_per_prompt["retrieval"],
            ),
            (
                "eval_reward_m2m",
                evaluation.m2m_per_prompt,
                self.fixed_eval_baseline_per_prompt["m2m"],
            ),
        )
        for offset, (name, current, baseline) in enumerate(components):
            result.update(
                summarize_fixed_eval_component(
                    name,
                    current,
                    baseline,
                    bootstrap_samples=(
                        self.config.fixed_eval_bootstrap_samples
                    ),
                    seed=self.config.fixed_eval_seed + 100 * offset,
                )
            )
        if self.reward_calibration is None:
            raise RuntimeError(
                "Balanced fixed validation requires reward calibration."
            )
        result.update(
            compute_balanced_validation_metrics(
                evaluation.retrieval_per_prompt,
                self.fixed_eval_baseline_per_prompt["retrieval"],
                evaluation.m2m_per_prompt,
                self.fixed_eval_baseline_per_prompt["m2m"],
                retrieval_scale=self.reward_calibration.global_scale(
                    "retrieval"
                ),
                m2m_scale=self.reward_calibration.global_scale("m2m"),
                bootstrap_samples=self.config.fixed_eval_bootstrap_samples,
                seed=self.config.fixed_eval_seed + 10_000,
            )
        )
        return result

    def _fixed_step_eval_with_deltas(
        self,
        evaluation: FixedStepEvalResult,
    ) -> dict[str, Any]:
        if self.fixed_step_eval_baseline_per_prompt is None:
            raise RuntimeError(
                "Fixed step evaluation baseline has not been initialized."
            )
        result = dict(evaluation.metrics)
        components = (
            (
                "eval_step_total",
                evaluation.total_per_prompt,
                self.fixed_step_eval_baseline_per_prompt["total"],
            ),
            (
                "eval_step_retrieval",
                evaluation.retrieval_per_prompt,
                self.fixed_step_eval_baseline_per_prompt["retrieval"],
            ),
            (
                "eval_step_m2m",
                evaluation.m2m_per_prompt,
                self.fixed_step_eval_baseline_per_prompt["m2m"],
            ),
            (
                "eval_step_reward",
                evaluation.step_reward_per_prompt,
                self.fixed_step_eval_baseline_per_prompt["step_reward"],
            ),
            (
                "eval_step_exact_fraction",
                evaluation.exact_per_prompt,
                self.fixed_step_eval_baseline_per_prompt["exact_fraction"],
            ),
            (
                "eval_step_within_one_fraction",
                evaluation.within_one_per_prompt,
                self.fixed_step_eval_baseline_per_prompt[
                    "within_one_fraction"
                ],
            ),
            (
                "eval_step_soft_count",
                evaluation.soft_count_mean_per_prompt,
                self._fixed_step_baseline_component("soft_count_mean"),
            ),
            (
                "eval_step_candidate_count",
                evaluation.candidate_count_mean_per_prompt,
                self._fixed_step_baseline_component("candidate_count_mean"),
            ),
            (
                "eval_step_candidate_spacing",
                evaluation.candidate_spacing_mean_per_prompt,
                self._fixed_step_baseline_component(
                    "candidate_spacing_mean"
                ),
            ),
            (
                "eval_step_ankle_high_frequency_ratio",
                evaluation.ankle_high_frequency_ratio_per_prompt,
                self._fixed_step_baseline_component(
                    "ankle_high_frequency_ratio"
                ),
            ),
        )
        for offset, (name, current, baseline) in enumerate(components):
            result.update(
                summarize_fixed_eval_component(
                    name,
                    current,
                    baseline,
                    bootstrap_samples=self.config.fixed_eval_bootstrap_samples,
                    seed=self.config.fixed_eval_seed + 20_000 + 100 * offset,
                )
            )
        result.update(
            summarize_fixed_eval_error(
                "eval_step_mae",
                evaluation.mae_per_prompt,
                self.fixed_step_eval_baseline_per_prompt["mae"],
                bootstrap_samples=self.config.fixed_eval_bootstrap_samples,
                seed=self.config.fixed_eval_seed + 30_000,
            )
        )
        result.update(
            summarize_fixed_eval_error(
                "eval_step_soft_mae",
                evaluation.soft_mae_per_prompt,
                self._fixed_step_baseline_component("soft_mae"),
                bootstrap_samples=self.config.fixed_eval_bootstrap_samples,
                seed=self.config.fixed_eval_seed + 31_000,
            )
        )
        soft_error_baseline = self._fixed_step_baseline_component(
            "soft_error_mean"
        )
        result.update(
            {
                "eval_step_soft_error_mean": (
                    evaluation.soft_error_mean_per_prompt.mean().item()
                ),
                "eval_step_soft_error_mean_baseline": (
                    soft_error_baseline.mean().item()
                ),
                "eval_step_soft_error_mean_delta": (
                    evaluation.soft_error_mean_per_prompt
                    - soft_error_baseline
                ).mean().item(),
            }
        )
        detected_baseline = self.fixed_step_eval_baseline_per_prompt[
            "detected_mean"
        ]
        result.update(
            {
                "eval_step_detected_mean": (
                    evaluation.detected_mean_per_prompt.mean().item()
                ),
                "eval_step_detected_mean_baseline": (
                    detected_baseline.mean().item()
                ),
                "eval_step_detected_mean_delta": (
                    evaluation.detected_mean_per_prompt
                    - detected_baseline
                ).mean().item(),
            }
        )
        if self.step_reward_calibration is None:
            result["eval_normalized_step_delta"] = 0.0
        else:
            result["eval_normalized_step_delta"] = (
                result["eval_step_reward_delta"]
                / self.step_reward_calibration.global_scale()
            )
        return result

    def _initialize_fixed_eval(self) -> dict[str, Any] | None:
        if self.fixed_eval_pool is None:
            return None
        expected_signature = self._fixed_eval_signature()
        if (
            self.fixed_eval_baseline is not None
            and (
                self.fixed_eval_baseline_per_prompt is None
                or (
                    getattr(self, "fixed_step_eval_pool", None) is not None
                    and getattr(
                        self,
                        "fixed_step_eval_baseline_per_prompt",
                        None,
                    )
                    is None
                )
                or any(
                    self.fixed_eval_baseline.get(name) != value
                    for name, value in expected_signature.items()
                )
            )
        ):
            LOGGER.warning(
                "Fixed-eval signature changed; using the current policy as "
                "a new baseline. previous=%s current=%s",
                {
                    name: self.fixed_eval_baseline.get(name)
                    for name in expected_signature
                },
                expected_signature,
            )
            self.fixed_eval_baseline = None
            self.fixed_eval_baseline_per_prompt = None
            self.best_balanced_score = None
            self.best_balanced_epoch = None
            self.best_retrieval_delta = None
            self.best_retrieval_epoch = None
            self.best_m2m_delta = None
            self.best_m2m_epoch = None
            self.fixed_step_eval_baseline_per_prompt = None
            self.best_step_reward_delta = None
            self.best_step_epoch = None
            self.best_step_acceptance_score = None
            self.best_step_acceptance_epoch = None
            self.evals_without_improvement = 0
        if self.fixed_eval_baseline is None:
            if self.config.resume:
                LOGGER.warning(
                    "Using the resumed policy as the new fixed-eval baseline."
                )
            baseline_evaluation = self.evaluate_fixed_pool()
            self.fixed_eval_baseline_per_prompt = {
                "total": baseline_evaluation.total_per_prompt.clone(),
                "retrieval": (
                    baseline_evaluation.retrieval_per_prompt.clone()
                ),
                "m2m": baseline_evaluation.m2m_per_prompt.clone(),
            }
            self.fixed_eval_baseline = self._fixed_eval_with_deltas(
                baseline_evaluation
            )
        else:
            assert self.fixed_eval_baseline_per_prompt is not None
            baseline_evaluation = FixedEvalResult(
                metrics=dict(self.fixed_eval_baseline),
                total_per_prompt=(
                    self.fixed_eval_baseline_per_prompt["total"]
                ),
                retrieval_per_prompt=(
                    self.fixed_eval_baseline_per_prompt["retrieval"]
                ),
                m2m_per_prompt=self.fixed_eval_baseline_per_prompt["m2m"],
            )
        step_baseline_evaluation: FixedStepEvalResult | None = None
        if getattr(self, "fixed_step_eval_pool", None) is not None:
            if self.fixed_step_eval_baseline_per_prompt is None:
                step_baseline_evaluation = self.evaluate_fixed_step_pool()
                self.fixed_step_eval_baseline_per_prompt = {
                    "total": step_baseline_evaluation.total_per_prompt.clone(),
                    "retrieval": (
                        step_baseline_evaluation.retrieval_per_prompt.clone()
                    ),
                    "m2m": step_baseline_evaluation.m2m_per_prompt.clone(),
                    "step_reward": (
                        step_baseline_evaluation.step_reward_per_prompt.clone()
                    ),
                    "exact_fraction": (
                        step_baseline_evaluation.exact_per_prompt.clone()
                    ),
                    "within_one_fraction": (
                        step_baseline_evaluation.within_one_per_prompt.clone()
                    ),
                    "mae": step_baseline_evaluation.mae_per_prompt.clone(),
                    "detected_mean": (
                        step_baseline_evaluation.detected_mean_per_prompt.clone()
                    ),
                    "soft_count_mean": (
                        step_baseline_evaluation.soft_count_mean_per_prompt.clone()
                    ),
                    "soft_error_mean": (
                        step_baseline_evaluation.soft_error_mean_per_prompt.clone()
                    ),
                    "soft_mae": (
                        step_baseline_evaluation.soft_mae_per_prompt.clone()
                    ),
                    "candidate_count_mean": (
                        step_baseline_evaluation.candidate_count_mean_per_prompt.clone()
                    ),
                    "candidate_spacing_mean": (
                        step_baseline_evaluation.candidate_spacing_mean_per_prompt.clone()
                    ),
                    "ankle_high_frequency_ratio": (
                        step_baseline_evaluation.ankle_high_frequency_ratio_per_prompt.clone()
                    ),
                }
                self.fixed_eval_baseline.update(
                    self._fixed_step_eval_with_deltas(
                        step_baseline_evaluation
                    )
                )
            else:
                step_baseline_evaluation = FixedStepEvalResult(
                    metrics=dict(self.fixed_eval_baseline),
                    total_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt["total"]
                    ),
                    retrieval_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt["retrieval"]
                    ),
                    m2m_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt["m2m"]
                    ),
                    step_reward_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt["step_reward"]
                    ),
                    exact_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt[
                            "exact_fraction"
                        ]
                    ),
                    within_one_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt[
                            "within_one_fraction"
                        ]
                    ),
                    mae_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt["mae"]
                    ),
                    detected_mean_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt[
                            "detected_mean"
                        ]
                    ),
                    soft_count_mean_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt.get(
                            "soft_count_mean",
                            self.fixed_step_eval_baseline_per_prompt[
                                "detected_mean"
                            ],
                        )
                    ),
                    soft_error_mean_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt.get(
                            "soft_error_mean",
                            torch.zeros_like(
                                self.fixed_step_eval_baseline_per_prompt[
                                    "detected_mean"
                                ]
                            ),
                        )
                    ),
                    soft_mae_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt.get(
                            "soft_mae",
                            self.fixed_step_eval_baseline_per_prompt["mae"],
                        )
                    ),
                    candidate_count_mean_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt.get(
                            "candidate_count_mean",
                            self.fixed_step_eval_baseline_per_prompt[
                                "detected_mean"
                            ],
                        )
                    ),
                    candidate_spacing_mean_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt.get(
                            "candidate_spacing_mean",
                            torch.zeros_like(
                                self.fixed_step_eval_baseline_per_prompt[
                                    "detected_mean"
                                ]
                            ),
                        )
                    ),
                    ankle_high_frequency_ratio_per_prompt=(
                        self.fixed_step_eval_baseline_per_prompt.get(
                            "ankle_high_frequency_ratio",
                            torch.zeros_like(
                                self.fixed_step_eval_baseline_per_prompt[
                                    "detected_mean"
                                ]
                            ),
                        )
                    ),
                )
        initialized_best = self.best_balanced_score is None
        initialized_step_best = bool(
            getattr(self, "fixed_step_eval_pool", None) is not None
            and getattr(self, "best_step_reward_delta", None) is None
        )
        baseline_epoch = self.start_epoch - 1
        if self.best_balanced_score is None:
            self.best_balanced_score = 0.0
            self.best_balanced_epoch = baseline_epoch
            self.best_retrieval_delta = 0.0
            self.best_retrieval_epoch = baseline_epoch
            self.best_m2m_delta = 0.0
            self.best_m2m_epoch = baseline_epoch
        if initialized_step_best:
            self.best_step_reward_delta = 0.0
            self.best_step_epoch = baseline_epoch
        if initialized_best:
            for name in (
                "best_balanced.pt",
                "best_retrieval.pt",
                "best_m2m.pt",
            ):
                self._save_named_snapshot(name, baseline_epoch)
        if initialized_step_best:
            self._save_named_snapshot("best_step.pt", baseline_epoch)
        baseline_metrics = self._fixed_eval_with_deltas(baseline_evaluation)
        if step_baseline_evaluation is not None:
            baseline_metrics.update(
                self._fixed_step_eval_with_deltas(step_baseline_evaluation)
            )
        record: dict[str, Any] = {
            "event": "baseline",
            "epoch": self.start_epoch - 1,
            "global_step": self.global_step,
            **baseline_metrics,
        }
        self._append_fixed_eval(record)
        self._append_fixed_eval_per_prompt(
            event="baseline",
            epoch=self.start_epoch - 1,
            evaluation=baseline_evaluation,
        )
        if step_baseline_evaluation is not None:
            self._append_fixed_step_eval_per_prompt(
                event="baseline",
                epoch=self.start_epoch - 1,
                evaluation=step_baseline_evaluation,
            )
        LOGGER.info("fixed evaluation baseline: %s", json.dumps(record, sort_keys=True))
        return record

    def _run_fixed_eval(self, epoch: int) -> dict[str, Any]:
        evaluation = self.evaluate_fixed_pool()
        metrics = self._fixed_eval_with_deltas(evaluation)
        step_evaluation: FixedStepEvalResult | None = None
        is_best_step = False
        is_best_step_acceptance = False
        step_acceptance = False
        step_acceptance_score = 0.0
        if getattr(self, "fixed_step_eval_pool", None) is not None:
            step_evaluation = self.evaluate_fixed_step_pool()
            metrics.update(self._fixed_step_eval_with_deltas(step_evaluation))
            is_best_step = bool(
                self.best_step_reward_delta is None
                or metrics["eval_step_reward_delta"]
                > self.best_step_reward_delta
            )
            if is_best_step:
                self.best_step_reward_delta = metrics[
                    "eval_step_reward_delta"
                ]
                self.best_step_epoch = epoch
        retrieval_tolerance = -(
            self.config.checkpoint_feasible_se_multiplier
            * metrics["eval_reward_retrieval_delta_bootstrap_se"]
        )
        m2m_tolerance = -(
            self.config.checkpoint_feasible_se_multiplier
            * metrics["eval_reward_m2m_delta_bootstrap_se"]
        )
        feasible = bool(
            metrics["eval_reward_retrieval_delta"] >= retrieval_tolerance
            and metrics["eval_reward_m2m_delta"] >= m2m_tolerance
        )
        if step_evaluation is not None:
            step_acceptance = bool(
                feasible
                and metrics["eval_step_mae_delta"] < 0
                and metrics["eval_step_exact_fraction_delta"] > 0
                and metrics["eval_step_within_one_fraction_delta"] > 0
            )
            sample_resolution = 1.0 / max(
                1.0,
                float(metrics["step_eval_samples"]),
            )

            def standardized_improvement(name: str, *, lower: bool) -> float:
                delta = float(metrics[f"{name}_delta"])
                standard_error = max(
                    sample_resolution,
                    float(metrics[f"{name}_delta_bootstrap_se"]),
                )
                return (-delta if lower else delta) / standard_error

            step_acceptance_score = (
                standardized_improvement("eval_step_mae", lower=True)
                + standardized_improvement(
                    "eval_step_exact_fraction",
                    lower=False,
                )
                + standardized_improvement(
                    "eval_step_within_one_fraction",
                    lower=False,
                )
            )
            is_best_step_acceptance = bool(
                step_acceptance
                and (
                    self.best_step_acceptance_score is None
                    or step_acceptance_score
                    > self.best_step_acceptance_score
                )
            )
            if is_best_step_acceptance:
                self.best_step_acceptance_score = step_acceptance_score
                self.best_step_acceptance_epoch = epoch
        automatic_min_delta = (
            self.config.early_stop_se_multiplier
            * metrics["eval_balanced_score_bootstrap_se"]
            if self.config.early_stop_min_delta_mode == "auto"
            else 0.0
        )
        effective_min_delta = max(
            self.config.early_stop_min_delta,
            automatic_min_delta,
        )
        is_best_balanced = bool(
            feasible
            and (
                self.best_balanced_score is None
                or metrics["eval_balanced_score"]
                > self.best_balanced_score + effective_min_delta
            )
        )
        is_best_retrieval = bool(
            self.best_retrieval_delta is None
            or metrics["eval_reward_retrieval_delta"]
            > self.best_retrieval_delta
        )
        is_best_m2m = bool(
            self.best_m2m_delta is None
            or metrics["eval_reward_m2m_delta"] > self.best_m2m_delta
        )
        if is_best_balanced:
            self.best_balanced_score = metrics["eval_balanced_score"]
            self.best_balanced_epoch = epoch
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
        if is_best_retrieval:
            self.best_retrieval_delta = metrics[
                "eval_reward_retrieval_delta"
            ]
            self.best_retrieval_epoch = epoch
        if is_best_m2m:
            self.best_m2m_delta = metrics["eval_reward_m2m_delta"]
            self.best_m2m_epoch = epoch
        assert self.best_balanced_score is not None
        assert self.best_balanced_epoch is not None
        assert self.best_retrieval_delta is not None
        assert self.best_retrieval_epoch is not None
        assert self.best_m2m_delta is not None
        assert self.best_m2m_epoch is not None
        metrics.update(
            {
                "eval_feasible": float(feasible),
                "eval_retrieval_tolerance": retrieval_tolerance,
                "eval_m2m_tolerance": m2m_tolerance,
                "eval_effective_min_delta": effective_min_delta,
                "eval_is_best": float(is_best_balanced),
                "eval_is_best_balanced": float(is_best_balanced),
                "eval_is_best_retrieval": float(is_best_retrieval),
                "eval_is_best_m2m": float(is_best_m2m),
                "eval_is_best_step": float(is_best_step),
                "eval_step_acceptance": float(step_acceptance),
                "eval_step_acceptance_score": step_acceptance_score,
                "eval_is_best_step_acceptance": float(
                    is_best_step_acceptance
                ),
                "eval_best_balanced_score": self.best_balanced_score,
                "eval_best_balanced_epoch": float(
                    self.best_balanced_epoch
                ),
                "eval_best_retrieval_delta": self.best_retrieval_delta,
                "eval_best_retrieval_epoch": float(
                    self.best_retrieval_epoch
                ),
                "eval_best_m2m_delta": self.best_m2m_delta,
                "eval_best_m2m_epoch": float(self.best_m2m_epoch),
                "eval_best_step_delta": float(
                    getattr(self, "best_step_reward_delta", None) or 0.0
                ),
                "eval_best_step_epoch": float(
                    getattr(self, "best_step_epoch", None)
                    if getattr(self, "best_step_epoch", None) is not None
                    else -1
                ),
                "eval_best_step_acceptance_score": float(
                    getattr(self, "best_step_acceptance_score", None) or 0.0
                ),
                "eval_best_step_acceptance_epoch": float(
                    getattr(self, "best_step_acceptance_epoch", None)
                    if getattr(
                        self,
                        "best_step_acceptance_epoch",
                        None,
                    ) is not None
                    else -1
                ),
                "eval_evals_without_improvement": float(
                    self.evals_without_improvement
                ),
            }
        )
        self._append_fixed_eval(
            {
                "event": "evaluation",
                "epoch": epoch,
                "global_step": self.global_step,
                **metrics,
            }
        )
        self._append_fixed_eval_per_prompt(
            event="evaluation",
            epoch=epoch,
            evaluation=evaluation,
        )
        if step_evaluation is not None:
            self._append_fixed_step_eval_per_prompt(
                event="evaluation",
                epoch=epoch,
                evaluation=step_evaluation,
            )
        return metrics

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if self.data_loader.generator is not None:
            state["data_loader"] = self.data_loader.generator.get_state()
        if (
            self.step_data_loader is not None
            and self.step_data_loader.generator is not None
        ):
            state["step_data_loader"] = (
                self.step_data_loader.generator.get_state()
            )
        return state

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            for device_index, device_state in enumerate(
                state["cuda"][: torch.cuda.device_count()]
            ):
                torch.cuda.set_rng_state(
                    device_state.cpu(),
                    device=device_index,
                )
        if (
            "data_loader" in state
            and self.data_loader.generator is not None
        ):
            self.data_loader.generator.set_state(
                state["data_loader"].cpu()
            )
        if (
            "step_data_loader" in state
            and self.step_data_loader is not None
            and self.step_data_loader.generator is not None
        ):
            self.step_data_loader.generator.set_state(
                state["step_data_loader"].cpu()
            )

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "mdm_diffusion": dict(
                getattr(self, "diffusion_metadata", {})
            ),
            "count_conditioning": count_conditioning_signature(self.model),
            "initial_policy_id": getattr(self, "initial_policy_id", ""),
            "train_mode": self.config.train_mode,
            "policy": trainable_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": self._rng_state(),
            "fixed_eval_baseline": self.fixed_eval_baseline,
            "fixed_eval_baseline_per_prompt": (
                self.fixed_eval_baseline_per_prompt
            ),
            "fixed_step_eval_baseline_per_prompt": (
                self.fixed_step_eval_baseline_per_prompt
            ),
            "fixed_eval_pool_id": (
                self.fixed_eval_pool.pool_id
                if self.fixed_eval_pool is not None
                else None
            ),
            "anchor_lambda_effective": self.anchor_lambda_effective,
            "anchor_lambda_calibrated": self.anchor_lambda_calibrated,
            "reward_calibration_id": (
                self.reward_calibration.calibration_id
                if self.reward_calibration is not None
                else None
            ),
            "step_reward_calibration_id": (
                self.step_reward_calibration.calibration_id
                if self.step_reward_calibration is not None
                else None
            ),
            "fixed_step_eval_pool_id": (
                self.fixed_step_eval_pool.pool_id
                if self.fixed_step_eval_pool is not None
                else None
            ),
            "best_balanced_score": self.best_balanced_score,
            "best_balanced_epoch": self.best_balanced_epoch,
            "best_retrieval_delta": self.best_retrieval_delta,
            "best_retrieval_epoch": self.best_retrieval_epoch,
            "best_m2m_delta": self.best_m2m_delta,
            "best_m2m_epoch": self.best_m2m_epoch,
            "best_step_reward_delta": self.best_step_reward_delta,
            "best_step_epoch": self.best_step_epoch,
            "best_step_acceptance_score": self.best_step_acceptance_score,
            "best_step_acceptance_epoch": self.best_step_acceptance_epoch,
            "evals_without_improvement": self.evals_without_improvement,
        }

    def _save_named_snapshot(self, name: str, epoch: int) -> Path:
        if name not in {
            "best_balanced.pt",
            "best_retrieval.pt",
            "best_m2m.pt",
            "best_step.pt",
            "best_step_acceptance.pt",
            "latest.pt",
        }:
            raise ValueError(f"Unsupported named checkpoint: {name!r}.")
        path = self.output_dir / name
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch), temporary_path)
        os.replace(temporary_path, path)
        LOGGER.info("Saved named checkpoint: %s", path)
        return path

    def _save_checkpoint(
        self,
        epoch: int,
        *,
        best_names: tuple[str, ...] = (),
    ) -> Path:
        checkpoint_path = self.output_dir / f"checkpoint_{epoch:06d}.pt"
        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(self._checkpoint_payload(epoch), temporary_path)
        os.replace(temporary_path, checkpoint_path)
        shutil.copy2(checkpoint_path, self.output_dir / "latest.pt")
        for name in best_names:
            if name not in {
                "best_balanced.pt",
                "best_retrieval.pt",
                "best_m2m.pt",
                "best_step.pt",
                "best_step_acceptance.pt",
            }:
                raise ValueError(f"Unsupported best checkpoint: {name!r}.")
            shutil.copy2(checkpoint_path, self.output_dir / name)
            LOGGER.info("Saved best checkpoint: %s", self.output_dir / name)
        LOGGER.info("Saved checkpoint: %s", checkpoint_path)
        return checkpoint_path

    def _validate_policy_structure(
        self,
        payload: dict[str, Any],
        *,
        source: str,
    ) -> None:
        checkpoint_mode = payload.get("train_mode")
        if checkpoint_mode != self.config.train_mode:
            raise ValueError(
                f"{source} train_mode={checkpoint_mode!r} does not match "
                f"current mode={self.config.train_mode!r}."
            )
        checkpoint_config = payload.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError(f"{source} has no valid config mapping.")
        expected_lora = {
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_target_regex": self.config.lora_target_regex,
        }
        actual_lora = {
            name: checkpoint_config.get(name)
            for name in expected_lora
        }
        if actual_lora != expected_lora:
            raise ValueError(
                f"{source} LoRA configuration does not match the current "
                f"policy: expected={expected_lora}, actual={actual_lora}."
            )
        current_diffusion = getattr(self, "diffusion_metadata", None)
        if current_diffusion is not None:
            checkpoint_diffusion = payload.get("mdm_diffusion")
            if not isinstance(checkpoint_diffusion, dict):
                raise ValueError(
                    f"{source} has no audited MDM diffusion metadata."
                )
            validate_diffusion_runtime_metadata(
                checkpoint_diffusion,
                current_diffusion,
                source=source,
            )
        validate_count_conditioning_signature(
            self.model,
            payload.get("count_conditioning"),
            source=source,
        )

    def _load_initial_policy(self, path: str) -> None:
        """Load SFT policy tensors without inheriting optimizer/train state."""

        checkpoint_path = Path(path).expanduser().resolve()
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise TypeError("Initial policy checkpoint must contain a mapping.")
        if payload.get("format") != "count_conditioning_sft_v1":
            raise ValueError(
                "--initial-policy-path must point to a "
                "count_conditioning_sft_v1 checkpoint."
            )
        self._validate_policy_structure(payload, source="Initial SFT policy")
        checkpoint_model_path = payload["config"].get("model_path")
        if checkpoint_model_path is not None:
            expected_model = Path(self.config.model_path).expanduser().resolve()
            actual_model = Path(checkpoint_model_path).expanduser().resolve()
            if actual_model != expected_model:
                raise ValueError(
                    "Initial SFT policy was trained from a different base MDM: "
                    f"expected={expected_model}, actual={actual_model}."
                )
        policy = payload.get("policy")
        if not isinstance(policy, dict):
            raise ValueError("Initial SFT policy has no policy state mapping.")
        load_trainable_state_dict(self.model, policy)
        self.initial_policy_id = policy_checkpoint_id(payload)
        LOGGER.info(
            "Initialized LoRA/count policy from native SFT checkpoint: %s",
            checkpoint_path,
        )

    def _load_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self._validate_policy_structure(checkpoint, source="Checkpoint")
        checkpoint_calibration_id = checkpoint.get("reward_calibration_id")
        current_calibration_id = (
            self.reward_calibration.calibration_id
            if self.reward_calibration is not None
            else None
        )
        if (
            checkpoint_calibration_id is not None
            and checkpoint_calibration_id != current_calibration_id
        ):
            raise ValueError(
                "Checkpoint reward calibration does not match the current "
                "--reward-calibration-path."
            )
        checkpoint_step_calibration_id = checkpoint.get(
            "step_reward_calibration_id"
        )
        current_step_calibration_id = (
            self.step_reward_calibration.calibration_id
            if self.step_reward_calibration is not None
            else None
        )
        if (
            checkpoint_step_calibration_id is not None
            and checkpoint_step_calibration_id
            != current_step_calibration_id
        ):
            raise ValueError(
                "Checkpoint step reward calibration does not match the "
                "current --step-reward-calibration-path."
            )
        load_trainable_state_dict(self.model, checkpoint["policy"])
        restored_optimizer = restore_optimizer_state(
            self.optimizer,
            self.scaler,
            checkpoint,
            self.config,
        )
        if not restored_optimizer:
            LOGGER.info(
                "Reset AdamW and GradScaler state while resuming policy weights."
            )
        if checkpoint.get("rng"):
            self._restore_rng_state(checkpoint["rng"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.initial_policy_id = str(
            checkpoint.get("initial_policy_id", "")
        )
        checkpoint_anchor_target = float(
            checkpoint.get("config", {}).get("anchor_auto_grad_ratio", 0.0)
        )
        if (
            self.config.anchor_auto_grad_ratio > 0
            and math.isclose(
                checkpoint_anchor_target,
                self.config.anchor_auto_grad_ratio,
            )
        ):
            self.anchor_lambda_effective = float(
                checkpoint.get("anchor_lambda_effective", 0.0)
            )
            self.anchor_lambda_calibrated = bool(
                checkpoint.get("anchor_lambda_calibrated", False)
            )
        self.fixed_eval_baseline = checkpoint.get("fixed_eval_baseline")
        baseline_per_prompt = checkpoint.get("fixed_eval_baseline_per_prompt")
        self.fixed_eval_baseline_per_prompt = (
            {
                name: values.detach().float().cpu()
                for name, values in baseline_per_prompt.items()
            }
            if baseline_per_prompt is not None
            else None
        )
        step_baseline_per_prompt = checkpoint.get(
            "fixed_step_eval_baseline_per_prompt"
        )
        self.fixed_step_eval_baseline_per_prompt = (
            {
                name: values.detach().float().cpu()
                for name, values in step_baseline_per_prompt.items()
            }
            if step_baseline_per_prompt is not None
            else None
        )
        self.checkpoint_fixed_eval_pool_id = checkpoint.get(
            "fixed_eval_pool_id"
        )
        self.checkpoint_fixed_step_eval_pool_id = checkpoint.get(
            "fixed_step_eval_pool_id"
        )
        self.best_balanced_score = checkpoint.get("best_balanced_score")
        self.best_balanced_epoch = checkpoint.get("best_balanced_epoch")
        self.best_retrieval_delta = checkpoint.get("best_retrieval_delta")
        self.best_retrieval_epoch = checkpoint.get("best_retrieval_epoch")
        self.best_m2m_delta = checkpoint.get("best_m2m_delta")
        self.best_m2m_epoch = checkpoint.get("best_m2m_epoch")
        self.best_step_reward_delta = checkpoint.get(
            "best_step_reward_delta"
        )
        self.best_step_epoch = checkpoint.get("best_step_epoch")
        self.best_step_acceptance_score = checkpoint.get(
            "best_step_acceptance_score"
        )
        self.best_step_acceptance_epoch = checkpoint.get(
            "best_step_acceptance_epoch"
        )
        if (
            "best_balanced_score" not in checkpoint
            and self.fixed_eval_baseline is not None
        ):
            LOGGER.warning(
                "Legacy checkpoint has no balanced-selection state; using "
                "the resumed policy as a new fixed-validation baseline."
            )
            self.fixed_eval_baseline = None
            self.fixed_eval_baseline_per_prompt = None
        self.evals_without_improvement = int(
            checkpoint.get("evals_without_improvement", 0)
        )
        LOGGER.info(
            "Resumed from %s at epoch=%d, global_step=%d",
            checkpoint_path,
            self.start_epoch,
            self.global_step,
        )

    def train(self) -> None:
        if self.start_epoch >= self.config.epochs:
            LOGGER.info(
                "Nothing to do: resume epoch %d >= configured epochs %d.",
                self.start_epoch,
                self.config.epochs,
            )
            return

        with SwanLabTracker(self.config, self.output_dir) as tracker:
            baseline_record = self._initialize_fixed_eval()
            if baseline_record is not None:
                tracker.log(
                    format_training_metrics(
                        baseline_record,
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    ),
                    step=0,
                )
            last_saved_epoch = -1
            completed_epoch = self.start_epoch - 1
            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_started = time.time()
                trajectory = self.collect_rollouts(epoch)
                rollout_metrics = self._rollout_metrics(trajectory)
                optimization_metrics = self.optimize(trajectory)
                fixed_eval_metrics: dict[str, Any] = {}
                if (
                    self.fixed_eval_pool is not None
                    and (epoch + 1) % self.config.fixed_eval_every == 0
                ):
                    fixed_eval_metrics = self._run_fixed_eval(epoch)
                best_names = tuple(
                    name
                    for metric_name, name in (
                        ("eval_is_best_balanced", "best_balanced.pt"),
                        ("eval_is_best_retrieval", "best_retrieval.pt"),
                        ("eval_is_best_m2m", "best_m2m.pt"),
                        ("eval_is_best_step", "best_step.pt"),
                        (
                            "eval_is_best_step_acceptance",
                            "best_step_acceptance.pt",
                        ),
                    )
                    if bool(fixed_eval_metrics.get(metric_name, 0.0))
                )
                should_early_stop = bool(
                    fixed_eval_metrics
                    and self.config.early_stop_patience > 0
                    and self.evals_without_improvement
                    >= self.config.early_stop_patience
                )
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "elapsed_seconds": time.time() - epoch_started,
                    **rollout_metrics,
                    **optimization_metrics,
                    **fixed_eval_metrics,
                }
                self._append_metrics(record)
                tracker.log(
                    format_training_metrics(
                        record,
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    ),
                    step=epoch + 1,
                )
                if epoch % self.config.log_every == 0:
                    LOGGER.info(
                        "epoch metrics: %s",
                        json.dumps(record, sort_keys=True),
                    )
                if (
                    (epoch + 1) % self.config.save_every == 0
                    or best_names
                    or should_early_stop
                ):
                    self._save_checkpoint(epoch, best_names=best_names)
                    last_saved_epoch = epoch
                else:
                    self._save_named_snapshot("latest.pt", epoch)
                completed_epoch = epoch
                if should_early_stop:
                    LOGGER.info(
                        "Early stopping at epoch=%d after %d fixed "
                        "evaluations without balanced improvement; best "
                        "epoch=%d, best balanced score=%.6f.",
                        epoch,
                        self.evals_without_improvement,
                        self.best_balanced_epoch,
                        self.best_balanced_score,
                    )
                    break

            final_epoch = completed_epoch
            if last_saved_epoch != final_epoch:
                self._save_checkpoint(final_epoch)
