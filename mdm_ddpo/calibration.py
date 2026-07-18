from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REWARD_CALIBRATION_VERSION = 1
MIN_CALIBRATION_PROMPTS = 1024
MIN_CALIBRATION_SAMPLES_PER_PROMPT = 4


def _quantile(values: torch.Tensor, probability: float) -> float:
    return torch.quantile(values, probability).item()


def _component_statistics(values: torch.Tensor) -> dict[str, float]:
    flattened = values.reshape(-1)
    within_std = values.std(dim=1, unbiased=False)
    within_range = values.max(dim=1).values - values.min(dim=1).values
    global_std = flattened.std(unbiased=False).item()
    if global_std <= 0:
        raise ValueError("Reward calibration requires non-zero global variance.")
    return {
        "global_mean": flattened.mean().item(),
        "global_std": global_std,
        "global_scale": global_std,
        "global_min": flattened.min().item(),
        "global_max": flattened.max().item(),
        "within_group_std_p25": _quantile(within_std, 0.25),
        "within_group_std_p50": _quantile(within_std, 0.50),
        "within_group_std_mean": within_std.mean().item(),
        "within_group_std_min": within_std.min().item(),
        "within_group_std_max": within_std.max().item(),
        "within_group_range_p25": _quantile(within_range, 0.25),
        "within_group_range_p50": _quantile(within_range, 0.50),
        "within_group_range_mean": within_range.mean().item(),
        "within_group_range_min": within_range.min().item(),
        "within_group_range_max": within_range.max().item(),
    }


def _pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.reshape(-1).float()
    second = second.reshape(-1).float()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if denominator <= 0:
        return 0.0
    return (first * second).sum().div(denominator).item()


def _ranking_relationships(
    retrieval: torch.Tensor,
    m2m: torch.Tensor,
    *,
    tie_epsilon: float = 1.0e-12,
) -> dict[str, float]:
    samples_per_prompt = retrieval.shape[1]
    first_indices, second_indices = torch.triu_indices(
        samples_per_prompt,
        samples_per_prompt,
        offset=1,
    )
    retrieval_differences = (
        retrieval[:, first_indices] - retrieval[:, second_indices]
    )
    m2m_differences = m2m[:, first_indices] - m2m[:, second_indices]
    comparable = (
        retrieval_differences.abs() > tie_epsilon
    ) & (m2m_differences.abs() > tie_epsilon)
    conflicts = comparable & (
        retrieval_differences * m2m_differences < 0
    )
    comparable_count = int(comparable.sum())
    total_count = comparable.numel()
    return {
        "ranking_conflict_fraction": (
            float(conflicts.sum()) / comparable_count
            if comparable_count
            else 0.0
        ),
        "ranking_conflict_fraction_all_pairs": (
            float(conflicts.sum()) / total_count if total_count else 0.0
        ),
        "ranking_comparable_pair_count": float(comparable_count),
        "ranking_total_pair_count": float(total_count),
        "ranking_tie_fraction": (
            1.0 - float(comparable_count) / total_count
            if total_count
            else 0.0
        ),
    }


def calibration_payload_id(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("calibration_id", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_reward_calibration(
    retrieval: torch.Tensor,
    m2m: torch.Tensor,
    *,
    retrieval_weight: float,
    m2m_weight: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute immutable reward scales from [prompt, sample] measurements."""
    retrieval = retrieval.detach().float().cpu()
    m2m = m2m.detach().float().cpu()
    if retrieval.ndim != 2 or retrieval.shape != m2m.shape:
        raise ValueError("Retrieval and M2M calibration tensors must match [P, K].")
    if retrieval.shape[0] <= 0 or retrieval.shape[1] < 2:
        raise ValueError("Reward calibration needs prompts with multiple samples.")
    if not torch.isfinite(retrieval).all() or not torch.isfinite(m2m).all():
        raise FloatingPointError("Reward calibration inputs must be finite.")
    total = retrieval_weight * retrieval + m2m_weight * m2m
    retrieval_centered = retrieval - retrieval.mean(dim=1, keepdim=True)
    m2m_centered = m2m - m2m.mean(dim=1, keepdim=True)
    prompt_count, samples_per_prompt = retrieval.shape
    payload: dict[str, Any] = {
        "schema_version": REWARD_CALIBRATION_VERSION,
        "num_prompts": prompt_count,
        "samples_per_prompt": samples_per_prompt,
        "full_calibration": bool(
            prompt_count >= MIN_CALIBRATION_PROMPTS
            and samples_per_prompt >= MIN_CALIBRATION_SAMPLES_PER_PROMPT
        ),
        "reward_weights": {
            "retrieval": float(retrieval_weight),
            "m2m": float(m2m_weight),
        },
        "components": {
            "retrieval": _component_statistics(retrieval),
            "m2m": _component_statistics(m2m),
            "total": _component_statistics(total),
        },
        "relationships": {
            "global_pearson_correlation": _pearson_correlation(
                retrieval,
                m2m,
            ),
            "within_group_centered_pearson_correlation": (
                _pearson_correlation(retrieval_centered, m2m_centered)
            ),
            **_ranking_relationships(retrieval, m2m),
        },
        "metadata": dict(metadata or {}),
    }
    payload["calibration_id"] = calibration_payload_id(payload)
    return payload


@dataclass(frozen=True)
class RewardCalibration:
    payload: dict[str, Any]
    path: Path

    @property
    def calibration_id(self) -> str:
        return str(self.payload["calibration_id"])

    def global_scale(self, component: str) -> float:
        return float(self.payload["components"][component]["global_scale"])

    def reward_weight(self, component: str) -> float:
        return float(self.payload["reward_weights"][component])

    def within_group_std_floor(self, component: str, quantile: str) -> float:
        if quantile not in {"p25", "p50"}:
            raise ValueError("Calibration floor quantile must be 'p25' or 'p50'.")
        return float(
            self.payload["components"][component][
                f"within_group_std_{quantile}"
            ]
        )


def validate_reward_calibration(
    payload: dict[str, Any],
    *,
    require_full: bool = True,
) -> None:
    if int(payload.get("schema_version", -1)) != REWARD_CALIBRATION_VERSION:
        raise ValueError("Unsupported reward calibration schema version.")
    if payload.get("calibration_id") != calibration_payload_id(payload):
        raise ValueError("Reward calibration checksum mismatch.")
    for component in ("retrieval", "m2m", "total"):
        statistics = payload.get("components", {}).get(component)
        if not isinstance(statistics, dict):
            raise KeyError(f"Reward calibration is missing component {component!r}.")
        for name in (
            "global_scale",
            "within_group_std_p25",
            "within_group_std_p50",
        ):
            value = float(statistics[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Reward calibration {component}.{name} must be finite "
                    "and positive."
                )
    if require_full and not bool(payload.get("full_calibration", False)):
        raise ValueError(
            "Training requires a full reward calibration generated with at "
            f"least {MIN_CALIBRATION_PROMPTS} prompts and "
            f"{MIN_CALIBRATION_SAMPLES_PER_PROMPT} samples per prompt."
        )


def load_reward_calibration(
    path: str | Path,
    *,
    require_full: bool = True,
) -> RewardCalibration:
    resolved = Path(path).expanduser().resolve()
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_reward_calibration(payload, require_full=require_full)
    return RewardCalibration(payload=payload, path=resolved)


def save_reward_calibration(payload: dict[str, Any], path: str | Path) -> Path:
    validate_reward_calibration(payload, require_full=False)
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(resolved)
    return resolved
