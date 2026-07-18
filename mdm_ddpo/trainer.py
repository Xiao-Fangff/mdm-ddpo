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
from .diffusion import ddim_step_with_logprob
from .lora import (
    LoRAReport,
    configure_trainable_policy,
    load_trainable_state_dict,
    parameter_counts,
    trainable_state_dict,
)
from .rewards import MotionReward, RewardOutput
from .runtime import (
    autocast_context,
    bootstrap_external_repositories,
    build_data_loader,
    build_dataset,
    build_mdm,
    build_model_kwargs,
    build_policy_model,
    CachedTextEmbedding,
    load_model_args,
    resolve_device,
    resolve_reward_device,
    seed_everything,
    split_text_embeddings,
)
from .tracking import SwanLabTracker, format_training_metrics


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


def _tensor_collection_l2_norm(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    squared_norm = sum(
        tensor.detach().float().square().sum()
        for tensor in tensors
    )
    return squared_norm.sqrt().item()


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
    samples_per_prompt: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Repeat each conditioning item contiguously for grouped DDPO rollouts."""
    prompt_count = motion.shape[0]
    if lengths.shape[0] != prompt_count or len(texts) != prompt_count:
        raise ValueError("Motion, length, and text prompt counts must match.")
    if samples_per_prompt < 2:
        raise ValueError("Grouped DDPO requires at least two samples per prompt.")
    repeated_motion = motion.repeat_interleave(samples_per_prompt, dim=0)
    repeated_lengths = lengths.repeat_interleave(samples_per_prompt, dim=0)
    repeated_texts = [
        text
        for text in texts
        for _ in range(samples_per_prompt)
    ]
    prompt_ids = torch.arange(prompt_count).repeat_interleave(samples_per_prompt)
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Shrink reward components independently before fixed-weight combining."""
    if retrieval_rewards.shape != m2m_rewards.shape:
        raise ValueError("Retrieval and M2M reward tensors must match.")
    if retrieval_rewards.ndim != 1 or prompt_ids.shape != retrieval_rewards.shape:
        raise ValueError("Component rewards and prompt ids must be matching 1-D tensors.")
    if retrieval_std_floor <= 0 or m2m_std_floor <= 0:
        raise ValueError("Component shrinkage floors must be positive.")
    if retrieval_weight < 0 or m2m_weight < 0:
        raise ValueError("Component advantage weights must be non-negative.")
    if retrieval_weight == 0 and m2m_weight == 0:
        raise ValueError("At least one component advantage weight must be non-zero.")

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
        self.data_loader = build_data_loader(config)
        self.data_iterator: Any | None = None
        self.model, self.diffusion, self.sample_steps = build_mdm(
            config,
            self.model_args,
            self.data_loader,
            self.device,
        )

        self.lora_report: LoRAReport | None = configure_trainable_policy(
            self.model,
            mode=config.train_mode,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_target_regex=config.lora_target_regex,
        )
        self.model.eval()
        self.policy_model = build_policy_model(
            self.model,
            config.guidance_scale,
        )
        self.policy_model.eval()

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
        self.evals_without_improvement = 0
        if config.resume:
            self._load_checkpoint(config.resume)
        self.fixed_eval_pool = (
            self._load_or_create_fixed_eval_pool()
            if config.fixed_eval_every > 0
            else None
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
            "Diffusion steps=%d, policy transitions per sample=%d, "
            "policy device=%s, reward device=%s",
            self.sample_steps,
            self.sample_steps - 1,
            self.device,
            self.reward_device,
        )
        if (
            config.precision != "no"
            and config.train_batch_size != config.rollout_batch_size
        ):
            LOGGER.warning(
                "Low-precision old/new log-probability agreement is best when "
                "--train-batch-size equals --rollout-batch-size."
            )

    def preflight_summary(self) -> dict[str, Any]:
        total, trainable = parameter_counts(self.model)
        return {
            "dataset_samples": len(self.data_loader.dataset),
            "diffusion_steps": self.sample_steps,
            "policy_transitions": self.sample_steps - 1,
            "policy_parameters": total,
            "trainable_parameters": trainable,
            "lora_adapters": (
                self.lora_report.adapters if self.lora_report is not None else 0
            ),
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
        }

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
            self.config.prompts_per_rollout_batch,
        )
        precision_code = {"no": 0.0, "fp16": 1.0, "bf16": 2.0}
        return {
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

    @torch.no_grad()
    def evaluate_fixed_pool(self) -> FixedEvalResult:
        """Evaluate identical prompts and diffusion noise with mean embeddings."""
        if self.fixed_eval_pool is None:
            raise RuntimeError("Fixed evaluation is disabled.")

        prompt_batch_size = self.config.prompts_per_rollout_batch
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

    def _next_batch(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.data_iterator is None:
            self.data_iterator = iter(self.data_loader)
        try:
            return next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            return next(self.data_iterator)

    @torch.no_grad()
    def _rollout_batch(self, epoch: int, batch_index: int) -> Trajectory:
        motion, condition = self._next_batch()
        lengths = condition["y"]["lengths"].long()
        texts = list(condition["y"]["text"])
        motion, lengths, texts, prompt_ids = repeat_prompt_batch(
            motion,
            lengths,
            texts,
            self.config.samples_per_prompt,
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
        )

    def collect_rollouts(self, epoch: int) -> Trajectory:
        self.model.eval()
        # A fresh randomized subset per DDPO epoch also makes epoch-boundary
        # resume reproducible from the DataLoader generator state.
        self.data_iterator = iter(self.data_loader)
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

    def optimize(self, trajectory: Trajectory) -> dict[str, float]:
        if trajectory.advantages is None:
            raise ValueError("Advantages must be computed before optimization.")
        self.model.eval()
        num_samples, num_timesteps = trajectory.timesteps.shape
        metric_values: dict[str, list[float]] = {
            "loss": [],
            "grad_norm": [],
            "update_norm": [],
            "skipped_updates": [],
        }
        log_ratio_parts: list[torch.Tensor] = []
        ratio_parts: list[torch.Tensor] = []
        audit_metrics: dict[str, float] | None = None

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
                )

                minibatch_timesteps = selected_timesteps[sample_indices]
                if audit_metrics is None:
                    audit_metrics = self._audit_first_update_log_probs(
                        trajectory,
                        sample_indices,
                        minibatch_timesteps,
                        model_kwargs,
                    )
                    LOGGER.info(
                        "Initial old/new log-probability audit: %s",
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
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = clip_grad_norm_(
                            trainable_parameters,
                            self.config.max_grad_norm,
                        )
                        finite_gradients = bool(
                            torch.isfinite(grad_norm).item()
                        )
                        if finite_gradients:
                            lora_before_update = [
                                parameter.detach().clone()
                                for parameter in lora_parameters
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
                                            lora_parameters,
                                            lora_before_update,
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
            name: float(np.mean(values)) if values else 0.0
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
                **audit_metrics,
            }
        )
        return result

    def _rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        advantages = trajectory.advantages
        assert advantages is not None
        group_stats = trajectory.group_stats or {}
        return {
            "reward": trajectory.rewards.mean().item(),
            "reward_std": trajectory.rewards.std(unbiased=False).item(),
            "reward_retrieval": trajectory.retrieval_rewards.mean().item(),
            "reward_m2m": trajectory.m2m_rewards.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "rollout_samples": float(len(trajectory.rewards)),
            "samples_per_prompt": float(self.config.samples_per_prompt),
            **group_stats,
        }

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

    def _initialize_fixed_eval(self) -> dict[str, Any] | None:
        if self.fixed_eval_pool is None:
            return None
        expected_signature = self._fixed_eval_signature()
        if (
            self.fixed_eval_baseline is not None
            and (
                self.fixed_eval_baseline_per_prompt is None
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
        initialized_best = self.best_balanced_score is None
        baseline_epoch = self.start_epoch - 1
        if self.best_balanced_score is None:
            self.best_balanced_score = 0.0
            self.best_balanced_epoch = baseline_epoch
            self.best_retrieval_delta = 0.0
            self.best_retrieval_epoch = baseline_epoch
            self.best_m2m_delta = 0.0
            self.best_m2m_epoch = baseline_epoch
        if initialized_best:
            for name in (
                "best_balanced.pt",
                "best_retrieval.pt",
                "best_m2m.pt",
            ):
                self._save_named_snapshot(name, baseline_epoch)
        record: dict[str, Any] = {
            "event": "baseline",
            "epoch": self.start_epoch - 1,
            "global_step": self.global_step,
            **self._fixed_eval_with_deltas(baseline_evaluation),
        }
        self._append_fixed_eval(record)
        self._append_fixed_eval_per_prompt(
            event="baseline",
            epoch=self.start_epoch - 1,
            evaluation=baseline_evaluation,
        )
        LOGGER.info("fixed evaluation baseline: %s", json.dumps(record, sort_keys=True))
        return record

    def _run_fixed_eval(self, epoch: int) -> dict[str, Any]:
        evaluation = self.evaluate_fixed_pool()
        metrics = self._fixed_eval_with_deltas(evaluation)
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

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "train_mode": self.config.train_mode,
            "policy": trainable_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": self._rng_state(),
            "fixed_eval_baseline": self.fixed_eval_baseline,
            "fixed_eval_baseline_per_prompt": (
                self.fixed_eval_baseline_per_prompt
            ),
            "fixed_eval_pool_id": (
                self.fixed_eval_pool.pool_id
                if self.fixed_eval_pool is not None
                else None
            ),
            "reward_calibration_id": (
                self.reward_calibration.calibration_id
                if self.reward_calibration is not None
                else None
            ),
            "best_balanced_score": self.best_balanced_score,
            "best_balanced_epoch": self.best_balanced_epoch,
            "best_retrieval_delta": self.best_retrieval_delta,
            "best_retrieval_epoch": self.best_retrieval_epoch,
            "best_m2m_delta": self.best_m2m_delta,
            "best_m2m_epoch": self.best_m2m_epoch,
            "evals_without_improvement": self.evals_without_improvement,
        }

    def _save_named_snapshot(self, name: str, epoch: int) -> Path:
        if name not in {
            "best_balanced.pt",
            "best_retrieval.pt",
            "best_m2m.pt",
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
            }:
                raise ValueError(f"Unsupported best checkpoint: {name!r}.")
            shutil.copy2(checkpoint_path, self.output_dir / name)
            LOGGER.info("Saved best checkpoint: %s", self.output_dir / name)
        LOGGER.info("Saved checkpoint: %s", checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint_mode = checkpoint.get("train_mode")
        if checkpoint_mode != self.config.train_mode:
            raise ValueError(
                f"Checkpoint train_mode={checkpoint_mode!r} does not match "
                f"current mode={self.config.train_mode!r}."
            )
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
        self.checkpoint_fixed_eval_pool_id = checkpoint.get(
            "fixed_eval_pool_id"
        )
        self.best_balanced_score = checkpoint.get("best_balanced_score")
        self.best_balanced_epoch = checkpoint.get("best_balanced_epoch")
        self.best_retrieval_delta = checkpoint.get("best_retrieval_delta")
        self.best_retrieval_epoch = checkpoint.get("best_retrieval_epoch")
        self.best_m2m_delta = checkpoint.get("best_m2m_delta")
        self.best_m2m_epoch = checkpoint.get("best_m2m_epoch")
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
