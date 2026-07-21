from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .lora import LoRAWeight


@dataclass(frozen=True)
class TimestepBucket:
    label: str
    lower: int
    upper: int

    def contains(self, values: torch.Tensor) -> torch.Tensor:
        return (values >= self.lower) & (values <= self.upper)


def parse_timestep_buckets(
    value: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[TimestepBucket, ...]:
    """Parse inclusive timestep ranges such as ``1-2,3-5,6-15``."""

    if minimum > maximum:
        raise ValueError("Timestep bucket bounds are invalid.")
    buckets: list[TimestepBucket] = []
    occupied: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_lower, raw_upper = part.split("-", 1)
            lower, upper = int(raw_lower), int(raw_upper)
        else:
            lower = upper = int(part)
        if lower > upper or lower < minimum or upper > maximum:
            raise ValueError(
                f"Timestep bucket {part!r} is outside [{minimum}, {maximum}]."
            )
        values = set(range(lower, upper + 1))
        if occupied & values:
            raise ValueError("Timestep buckets must not overlap.")
        occupied.update(values)
        buckets.append(TimestepBucket(part, lower, upper))
    if not buckets:
        raise ValueError("At least one timestep bucket is required.")
    return tuple(buckets)


def summarize_timestep_log_ratios(
    timesteps: torch.Tensor,
    log_ratios: torch.Tensor,
    *,
    clip_range: float,
    buckets: Iterable[TimestepBucket],
) -> dict[str, dict[str, float]]:
    """Summarize policy movement separately for configured timestep ranges."""

    timestep_values = timesteps.detach().long().reshape(-1).cpu()
    log_ratio_values = log_ratios.detach().float().reshape(-1).cpu()
    if timestep_values.shape != log_ratio_values.shape:
        raise ValueError("Timesteps and log ratios must have matching shapes.")
    if clip_range <= 0 or not math.isfinite(clip_range):
        raise ValueError("PPO clip range must be finite and positive.")
    output: dict[str, dict[str, float]] = {}
    for bucket in buckets:
        active = bucket.contains(timestep_values)
        if not active.any():
            continue
        values = log_ratio_values[active]
        ratios = values.clamp(-20.0, 20.0).exp()
        output[bucket.label] = {
            "samples": float(values.numel()),
            "log_ratio_mean": values.mean().item(),
            "log_ratio_std": values.std(unbiased=False).item(),
            "log_ratio_abs_mean": values.abs().mean().item(),
            "log_ratio_abs_max": values.abs().max().item(),
            "ratio_mean": ratios.mean().item(),
            "ratio_std": ratios.std(unbiased=False).item(),
            "clip_fraction": (
                (ratios - 1.0).abs() > clip_range
            ).float().mean().item(),
        }
    return output


def xstart_ddim_score_sensitivity(
    diffusion: Any,
    *,
    eta: float,
) -> dict[int, dict[str, float]]:
    """Measure d transition-mean / d x0, normalized by transition std.

    For a sampled action ``x_(t-1) = mean + std * noise``, the Gaussian
    score with respect to an x-start prediction is proportional to
    ``abs(d mean / d x0) / std``.  This exposes parameterization-specific
    timestep imbalance without running the policy network.
    """

    if eta <= 0 or not math.isfinite(eta):
        raise ValueError("DDIM eta must be finite and positive.")
    alpha = torch.as_tensor(diffusion.alphas_cumprod, dtype=torch.float64)
    alpha_prev = torch.as_tensor(
        diffusion.alphas_cumprod_prev,
        dtype=torch.float64,
    )
    sigma_sq = (
        eta**2
        * (1.0 - alpha_prev)
        / (1.0 - alpha).clamp_min(1.0e-30)
        * (1.0 - alpha / alpha_prev.clamp_min(1.0e-30))
    ).clamp_min(0.0)
    direction = (1.0 - alpha_prev - sigma_sq).clamp_min(0.0).sqrt()
    coefficient = alpha_prev.sqrt() - (
        direction * alpha.sqrt() / (1.0 - alpha).clamp_min(1.0e-30).sqrt()
    )
    sigma = sigma_sq.sqrt()
    output: dict[int, dict[str, float]] = {}
    for timestep in range(1, len(alpha)):
        output[timestep] = {
            "mean_xstart_coefficient": coefficient[timestep].item(),
            "transition_std": sigma[timestep].item(),
            "score_sensitivity": (
                coefficient[timestep].abs()
                / sigma[timestep].clamp_min(1.0e-30)
            ).item(),
        }
    return output


def epsilon_ddim_score_sensitivity(
    diffusion: Any,
    *,
    eta: float,
) -> dict[int, dict[str, float]]:
    """Hypothetical score sensitivity when the network predicts epsilon."""

    if eta <= 0 or not math.isfinite(eta):
        raise ValueError("DDIM eta must be finite and positive.")
    alpha = torch.as_tensor(diffusion.alphas_cumprod, dtype=torch.float64)
    alpha_prev = torch.as_tensor(
        diffusion.alphas_cumprod_prev,
        dtype=torch.float64,
    )
    sigma_sq = (
        eta**2
        * (1.0 - alpha_prev)
        / (1.0 - alpha).clamp_min(1.0e-30)
        * (1.0 - alpha / alpha_prev.clamp_min(1.0e-30))
    ).clamp_min(0.0)
    direction = (1.0 - alpha_prev - sigma_sq).clamp_min(0.0).sqrt()
    coefficient = direction - (
        alpha_prev.sqrt()
        * (1.0 - alpha).clamp_min(0.0).sqrt()
        / alpha.clamp_min(1.0e-30).sqrt()
    )
    sigma = sigma_sq.sqrt()
    output: dict[int, dict[str, float]] = {}
    for timestep in range(1, len(alpha)):
        output[timestep] = {
            "mean_epsilon_coefficient": coefficient[timestep].item(),
            "transition_std": sigma[timestep].item(),
            "score_sensitivity": (
                coefficient[timestep].abs()
                / sigma[timestep].clamp_min(1.0e-30)
            ).item(),
        }
    return output


def summarize_sensitivity_buckets(
    sensitivity: dict[int, dict[str, float]],
    buckets: Iterable[TimestepBucket],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for bucket in buckets:
        values = [
            record["score_sensitivity"]
            for timestep, record in sensitivity.items()
            if bucket.lower <= timestep <= bucket.upper
        ]
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float64)
        output[bucket.label] = {
            "timesteps": float(len(values)),
            "score_sensitivity_mean": tensor.mean().item(),
            "score_sensitivity_min": tensor.min().item(),
            "score_sensitivity_max": tensor.max().item(),
        }
    return output


def advantage_logprob_alignment(
    advantages: torch.Tensor,
    cumulative_logprob_delta: torch.Tensor,
) -> dict[str, float]:
    """Check whether fixed actions move in the direction of their advantage."""

    advantage = advantages.detach().float().reshape(-1).cpu()
    delta = cumulative_logprob_delta.detach().float().reshape(-1).cpu()
    if advantage.shape != delta.shape:
        raise ValueError("Advantages and log-probability deltas must match.")
    positive = advantage > 0
    negative = advantage < 0
    centered_advantage = advantage - advantage.mean()
    centered_delta = delta - delta.mean()
    denominator = (
        centered_advantage.square().sum().sqrt()
        * centered_delta.square().sum().sqrt()
    )
    correlation = (
        (centered_advantage * centered_delta).sum().div(denominator).item()
        if denominator > 0
        else 0.0
    )
    positive_mean = delta[positive].mean().item() if positive.any() else 0.0
    negative_mean = delta[negative].mean().item() if negative.any() else 0.0
    return {
        "positive_advantage_logprob_delta_mean": positive_mean,
        "negative_advantage_logprob_delta_mean": negative_mean,
        "positive_negative_gap": positive_mean - negative_mean,
        "advantage_logprob_delta_correlation": correlation,
        "advantage_weighted_logprob_delta_mean": (
            advantage * delta
        ).mean().item(),
    }


def effective_lora_delta_norm(model: torch.nn.Module) -> float:
    """Return the norm of effective LoRA weight deltas, not raw A/B factors."""

    squared = torch.zeros((), dtype=torch.float64)
    found = False
    for module in model.modules():
        if not isinstance(module, LoRAWeight):
            continue
        found = True
        delta = (
            module.lora_b.detach().float()
            @ module.lora_a.detach().float()
        ) * module.scale
        squared += delta.double().square().sum().cpu()
    if not found:
        return 0.0
    return squared.sqrt().item()


__all__ = [
    "TimestepBucket",
    "advantage_logprob_alignment",
    "epsilon_ddim_score_sensitivity",
    "effective_lora_delta_norm",
    "parse_timestep_buckets",
    "summarize_sensitivity_buckets",
    "summarize_timestep_log_ratios",
    "xstart_ddim_score_sensitivity",
]
