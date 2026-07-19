from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


STEP_REWARD_CALIBRATION_VERSION = 1
MIN_STEP_CALIBRATION_PROMPTS = 256
MIN_STEP_CALIBRATION_SAMPLES_PER_PROMPT = 4


def compute_target_error_scales(
    soft_counts: torch.Tensor,
    target_steps: torch.Tensor,
    *,
    minimum_scale: float,
) -> dict[str, float]:
    """Compute an immutable per-target RMSE scale for soft-count shaping."""

    if minimum_scale <= 0 or not math.isfinite(minimum_scale):
        raise ValueError("Soft-count minimum target scale must be positive.")
    soft = torch.as_tensor(soft_counts).detach().float().cpu()
    targets = torch.as_tensor(target_steps).detach().long().cpu()
    if soft.shape != targets.shape or soft.numel() == 0:
        raise ValueError("Soft counts and targets must be non-empty and match.")
    if not torch.isfinite(soft).all():
        raise FloatingPointError("Soft-count calibration values must be finite.")
    scales: dict[str, float] = {}
    for target in torch.unique(targets, sorted=True).tolist():
        active = targets == int(target)
        errors = soft[active] - float(target)
        rmse = errors.square().mean().sqrt().item()
        scales[str(int(target))] = max(float(minimum_scale), float(rmse))
    return scales


def _statistics(values: torch.Tensor) -> dict[str, float]:
    flattened = values.reshape(-1)
    within_std = values.std(dim=1, unbiased=False)
    within_range = values.max(dim=1).values - values.min(dim=1).values
    global_std = flattened.std(unbiased=False).item()
    if global_std <= 0:
        raise ValueError("Step reward calibration requires non-zero variance.")
    positive_std = within_std[within_std > 0]
    if positive_std.numel() == 0:
        raise ValueError(
            "Step reward calibration has no prompt with non-zero group variance."
        )
    raw_p25 = torch.quantile(within_std, 0.25).item()
    raw_p50 = torch.quantile(within_std, 0.50).item()
    tie_tolerance = 1.0e-6
    unique_levels = torch.tensor(
        [
            torch.unique(torch.round(group / tie_tolerance)).numel()
            for group in values
        ],
        dtype=torch.float32,
    )
    centered = values - values.mean(dim=1, keepdim=True)
    positive = centered.clamp_min(0.0)
    positive_sum = positive.sum(dim=1)
    top1_concentration = torch.where(
        positive_sum > tie_tolerance,
        positive.max(dim=1).values / positive_sum.clamp_min(tie_tolerance),
        torch.zeros_like(positive_sum),
    )
    tied_pairs = 0
    total_pairs = 0
    for group in values:
        difference = (group[:, None] - group[None, :]).abs()
        upper = torch.triu(
            torch.ones_like(difference, dtype=torch.bool),
            diagonal=1,
        )
        tied_pairs += int((difference[upper] <= tie_tolerance).sum())
        total_pairs += int(upper.sum())
    return {
        "global_mean": flattened.mean().item(),
        "global_std": global_std,
        "global_scale": global_std,
        "global_min": flattened.min().item(),
        "global_max": flattened.max().item(),
        # Hard count rewards are discrete and often produce exactly constant
        # prompt groups. Floors therefore use the positive-variance
        # distribution, while raw quantiles and zero fraction stay visible.
        "within_group_std_p25": torch.quantile(positive_std, 0.25).item(),
        "within_group_std_p50": torch.quantile(positive_std, 0.50).item(),
        "within_group_std_p25_raw": raw_p25,
        "within_group_std_p50_raw": raw_p50,
        "within_group_zero_std_fraction": (
            (within_std == 0).float().mean().item()
        ),
        "within_group_std_mean": within_std.mean().item(),
        "within_group_std_min": within_std.min().item(),
        "within_group_std_max": within_std.max().item(),
        "within_group_range_p25": torch.quantile(within_range, 0.25).item(),
        "within_group_range_p50": torch.quantile(within_range, 0.50).item(),
        "within_group_range_mean": within_range.mean().item(),
        "unique_reward_levels_mean": unique_levels.mean().item(),
        "unique_reward_levels_median": unique_levels.median().item(),
        "unique_reward_levels_min": unique_levels.min().item(),
        "pairwise_reward_tie_fraction": (
            tied_pairs / total_pairs if total_pairs else 0.0
        ),
        "nonzero_centered_sample_fraction": (
            (centered.abs() > tie_tolerance).float().mean().item()
        ),
        "top1_centered_concentration": top1_concentration.mean().item(),
    }


def step_calibration_payload_id(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("calibration_id", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_step_reward_calibration(
    rewards: torch.Tensor,
    detected_steps: torch.Tensor,
    target_steps: torch.Tensor,
    *,
    detector_config: dict[str, Any],
    reward_config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    soft_counts: torch.Tensor | None = None,
    target_error_scales: dict[str, float] | None = None,
    detection_diagnostics: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    rewards = rewards.detach().float().cpu()
    detected_steps = detected_steps.detach().long().cpu()
    target_steps = target_steps.detach().long().cpu()
    if rewards.ndim != 2:
        raise ValueError("Step calibration rewards must have shape [P,K].")
    if detected_steps.shape != rewards.shape:
        raise ValueError("Detected step calibration values must match rewards.")
    if target_steps.shape == (rewards.shape[0],):
        target_steps = target_steps[:, None].expand_as(detected_steps)
    if target_steps.shape != rewards.shape:
        raise ValueError("Target step calibration values must match rewards.")
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("Step calibration rewards must be finite.")
    prompt_count, samples_per_prompt = rewards.shape
    absolute_error = (detected_steps - target_steps).abs().float()
    resolved_soft: torch.Tensor | None = None
    if soft_counts is not None:
        resolved_soft = soft_counts.detach().float().cpu()
        if resolved_soft.shape != rewards.shape:
            raise ValueError("Soft-count calibration values must match rewards.")
        if not torch.isfinite(resolved_soft).all():
            raise FloatingPointError("Soft-count calibration values must be finite.")
        if target_error_scales is None:
            target_error_scales = compute_target_error_scales(
                resolved_soft,
                target_steps,
                minimum_scale=float(
                    reward_config.get("target_scale_floor", 0.25)
                ),
            )

    per_target: dict[str, dict[str, float]] = {}
    for target in torch.unique(target_steps, sorted=True).tolist():
        active = target_steps == int(target)
        target_hard = detected_steps[active].float()
        target_error = absolute_error[active]
        entry = {
            "samples": float(active.sum()),
            "hard_count_mean": target_hard.mean().item(),
            "hard_mae": target_error.mean().item(),
            "hard_exact_fraction": (target_error == 0).float().mean().item(),
            "hard_within_one_fraction": (
                (target_error <= 1).float().mean().item()
            ),
        }
        if resolved_soft is not None:
            target_soft = resolved_soft[active]
            soft_error = target_soft - float(target)
            entry.update(
                {
                    "soft_count_mean": target_soft.mean().item(),
                    "soft_error_mean": soft_error.mean().item(),
                    "soft_mae": soft_error.abs().mean().item(),
                    "soft_hard_abs_difference_mean": (
                        target_soft.sub(target_hard).abs().mean().item()
                    ),
                    "target_error_scale": float(
                        target_error_scales[str(int(target))]
                    ),
                }
            )
        per_target[str(int(target))] = entry
    payload: dict[str, Any] = {
        "schema_version": STEP_REWARD_CALIBRATION_VERSION,
        "num_prompts": prompt_count,
        "samples_per_prompt": samples_per_prompt,
        "full_calibration": bool(
            prompt_count >= MIN_STEP_CALIBRATION_PROMPTS
            and samples_per_prompt >= MIN_STEP_CALIBRATION_SAMPLES_PER_PROMPT
        ),
        "component": _statistics(rewards),
        "detector": dict(detector_config),
        "reward": dict(reward_config),
        "metrics": {
            "mean_absolute_error": absolute_error.mean().item(),
            "exact_fraction": (absolute_error == 0).float().mean().item(),
            "within_one_fraction": (absolute_error <= 1).float().mean().item(),
            "detected_mean": detected_steps.float().mean().item(),
            "target_mean": target_steps.float().mean().item(),
        },
        "per_target": per_target,
        "metadata": dict(metadata or {}),
    }
    if resolved_soft is not None:
        payload["target_error_scales"] = dict(target_error_scales or {})
        payload["metrics"].update(
            {
                "soft_count_mean": resolved_soft.mean().item(),
                "soft_mean_absolute_error": (
                    resolved_soft.sub(target_steps.float()).abs().mean().item()
                ),
                "soft_hard_abs_difference_mean": (
                    resolved_soft.sub(detected_steps.float()).abs().mean().item()
                ),
            }
        )
    if detection_diagnostics:
        payload["detection_diagnostics"] = {
            name: {
                "mean": values.detach().float().cpu().mean().item(),
                "max": values.detach().float().cpu().max().item(),
            }
            for name, values in sorted(detection_diagnostics.items())
        }
    payload["calibration_id"] = step_calibration_payload_id(payload)
    return payload


@dataclass(frozen=True)
class StepRewardCalibration:
    payload: dict[str, Any]
    path: Path

    @property
    def calibration_id(self) -> str:
        return str(self.payload["calibration_id"])

    def global_scale(self) -> float:
        return float(self.payload["component"]["global_scale"])

    def within_group_std_floor(self, quantile: str) -> float:
        if quantile not in {"p25", "p50"}:
            raise ValueError("Step calibration quantile must be p25 or p50.")
        return float(
            self.payload["component"][f"within_group_std_{quantile}"]
        )

    def target_error_scales(self, target_steps: torch.Tensor) -> torch.Tensor:
        values = self.payload.get("target_error_scales")
        if not isinstance(values, dict):
            raise KeyError(
                "Soft step calibration is missing per-target error scales."
            )
        targets = torch.as_tensor(target_steps).reshape(-1)
        scales: list[float] = []
        for target in targets.detach().cpu().tolist():
            key = str(int(target))
            if key not in values:
                raise KeyError(
                    f"Soft step calibration has no target scale for {key}."
                )
            scales.append(float(values[key]))
        return torch.tensor(
            scales,
            device=targets.device,
            dtype=torch.float32,
        ).reshape(target_steps.shape)

    def validate_settings(
        self,
        *,
        detector_config: dict[str, Any],
        reward_config: dict[str, Any],
        samples_per_prompt: int | None = None,
    ) -> None:
        if self.payload["detector"] != detector_config:
            raise ValueError(
                "Step reward calibration detector settings do not match training."
            )
        if self.payload["reward"] != reward_config:
            raise ValueError(
                "Step reward calibration reward settings do not match training."
            )
        if (
            samples_per_prompt is not None
            and int(self.payload["samples_per_prompt"])
            != int(samples_per_prompt)
        ):
            raise ValueError(
                "Step reward calibration samples_per_prompt does not match "
                "training: calibration was generated with "
                f"K={int(self.payload['samples_per_prompt'])}, but training "
                f"uses --step-samples-per-prompt={int(samples_per_prompt)}. "
                "Regenerate the calibration with the training step K."
            )
        if reward_config.get("mode") == "soft_huber_exact":
            values = self.payload.get("target_error_scales")
            if not isinstance(values, dict) or not values:
                raise ValueError(
                    "soft_huber_exact calibration is missing target scales."
                )


def validate_step_reward_calibration(
    payload: dict[str, Any],
    *,
    require_full: bool = True,
) -> None:
    if int(payload.get("schema_version", -1)) != STEP_REWARD_CALIBRATION_VERSION:
        raise ValueError("Unsupported step reward calibration schema version.")
    if payload.get("calibration_id") != step_calibration_payload_id(payload):
        raise ValueError("Step reward calibration checksum mismatch.")
    component = payload.get("component")
    if not isinstance(component, dict):
        raise KeyError("Step reward calibration is missing component statistics.")
    for name in (
        "global_scale",
        "within_group_std_p25",
        "within_group_std_p50",
    ):
        value = float(component[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Step reward calibration {name} must be finite and positive."
            )
    if not isinstance(payload.get("detector"), dict):
        raise KeyError("Step reward calibration is missing detector settings.")
    if not isinstance(payload.get("reward"), dict):
        raise KeyError("Step reward calibration is missing reward settings.")
    if payload["reward"].get("mode") == "soft_huber_exact":
        scales = payload.get("target_error_scales")
        if not isinstance(scales, dict) or not scales:
            raise KeyError(
                "Soft step calibration is missing target_error_scales."
            )
        for target, value in scales.items():
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"Soft target scale for {target} must be positive."
                )
    if require_full and not bool(payload.get("full_calibration", False)):
        raise ValueError(
            "Training requires step calibration with at least "
            f"{MIN_STEP_CALIBRATION_PROMPTS} prompts and "
            f"{MIN_STEP_CALIBRATION_SAMPLES_PER_PROMPT} samples per prompt."
        )


def load_step_reward_calibration(
    path: str | Path,
    *,
    require_full: bool = True,
) -> StepRewardCalibration:
    resolved = Path(path).expanduser().resolve()
    with open(resolved, encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_step_reward_calibration(payload, require_full=require_full)
    return StepRewardCalibration(payload=payload, path=resolved)


def save_step_reward_calibration(
    payload: dict[str, Any],
    path: str | Path,
) -> Path:
    validate_step_reward_calibration(payload, require_full=False)
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(resolved)
    return resolved


__all__ = [
    "MIN_STEP_CALIBRATION_PROMPTS",
    "MIN_STEP_CALIBRATION_SAMPLES_PER_PROMPT",
    "StepRewardCalibration",
    "compute_target_error_scales",
    "compute_step_reward_calibration",
    "load_step_reward_calibration",
    "save_step_reward_calibration",
    "step_calibration_payload_id",
    "validate_step_reward_calibration",
]
