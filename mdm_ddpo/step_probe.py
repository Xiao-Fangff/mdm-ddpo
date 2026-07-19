from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .step_data import (
    NUMBER_WORDS,
    SHARED_STEP_PROMPT_TEMPLATES,
    StepSampleRecord,
    parse_step_targets,
    shared_step_length_support,
)


COUNTERFACTUAL_POOL_VERSION = 1


@dataclass(frozen=True)
class CounterfactualNumberPool:
    lengths: torch.Tensor
    template_slots: torch.Tensor
    targets: torch.Tensor
    noise_seeds: torch.Tensor
    max_frames: int
    prompt_seed: int
    number_style: str
    pool_id: str = ""

    @property
    def condition_count(self) -> int:
        return int(self.lengths.numel())

    @property
    def samples_per_condition(self) -> int:
        return int(self.noise_seeds.shape[1])


def render_counterfactual_step_prompt(
    target: int,
    template_slot: int,
    *,
    seed: int,
    number_style: str,
) -> str:
    if target < 1 or target > 6:
        raise ValueError("Counterfactual targets must be in 1..6.")
    if number_style not in {"digits", "words"}:
        raise ValueError("Counterfactual number style must be digits or words.")
    order = list(range(len(SHARED_STEP_PROMPT_TEMPLATES)))
    random.Random(seed * 31).shuffle(order)
    template = SHARED_STEP_PROMPT_TEMPLATES[
        order[template_slot % len(order)]
    ]
    number = str(target) if number_style == "digits" else NUMBER_WORDS[target]
    return template.format(
        n=number,
        steps="step" if target == 1 else "steps",
    )


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def counterfactual_pool_id(pool: CounterfactualNumberPool) -> str:
    digest = hashlib.sha256()
    digest.update(str(COUNTERFACTUAL_POOL_VERSION).encode())
    digest.update(str(pool.max_frames).encode())
    digest.update(str(pool.prompt_seed).encode())
    digest.update(pool.number_style.encode())
    for tensor in (
        pool.lengths,
        pool.template_slots,
        pool.targets,
        pool.noise_seeds,
    ):
        _hash_tensor(digest, tensor)
    return digest.hexdigest()


def validate_counterfactual_pool(
    pool: CounterfactualNumberPool,
) -> CounterfactualNumberPool:
    condition_count = int(pool.lengths.numel())
    if condition_count <= 0:
        raise ValueError("Counterfactual pool cannot be empty.")
    if pool.template_slots.shape != (condition_count,):
        raise ValueError("Counterfactual template slots do not match lengths.")
    if pool.targets.ndim != 1 or pool.targets.numel() < 2:
        raise ValueError("Counterfactual pool needs at least two targets.")
    if pool.noise_seeds.ndim != 2 or pool.noise_seeds.shape[0] != condition_count:
        raise ValueError("Counterfactual noise seed shape is invalid.")
    if pool.noise_seeds.shape[1] <= 0:
        raise ValueError("Counterfactual pool needs at least one noise sample.")
    if pool.max_frames <= 0:
        raise ValueError("Counterfactual max_frames must be positive.")
    if (pool.lengths <= 0).any() or (pool.lengths > pool.max_frames).any():
        raise ValueError("Counterfactual lengths are outside model support.")
    if pool.number_style not in {"digits", "words"}:
        raise ValueError("Counterfactual number style is invalid.")
    normalized = CounterfactualNumberPool(
        lengths=pool.lengths.detach().cpu().long(),
        template_slots=pool.template_slots.detach().cpu().long(),
        targets=pool.targets.detach().cpu().long(),
        noise_seeds=pool.noise_seeds.detach().cpu().long(),
        max_frames=int(pool.max_frames),
        prompt_seed=int(pool.prompt_seed),
        number_style=str(pool.number_style),
    )
    calculated = counterfactual_pool_id(normalized)
    if pool.pool_id and pool.pool_id != calculated:
        raise ValueError("Counterfactual pool checksum mismatch.")
    return replace(normalized, pool_id=calculated)


def create_counterfactual_pool(
    records: Sequence[StepSampleRecord],
    *,
    targets: Sequence[int],
    condition_count: int,
    samples_per_condition: int,
    max_frames: int,
    seed: int,
    prompt_seed: int,
    number_style: str = "words",
) -> CounterfactualNumberPool:
    target_values = parse_step_targets(targets)
    template_count = len(SHARED_STEP_PROMPT_TEMPLATES)
    if condition_count <= 0 or condition_count % template_count != 0:
        raise ValueError(
            "Counterfactual condition_count must be divisible by the shared "
            f"template count ({template_count})."
        )
    if samples_per_condition <= 0:
        raise ValueError("Counterfactual samples_per_condition must be positive.")
    support, _ = shared_step_length_support(records, targets=target_values)
    length_count = condition_count // template_count
    # Deterministic spread over the entire common interval, rather than a
    # target-conditioned empirical draw.
    selected_lengths = [
        support[min(len(support) - 1, int((index + 0.5) * len(support) / length_count))]
        for index in range(length_count)
    ]
    lengths: list[int] = []
    template_slots: list[int] = []
    for length in selected_lengths:
        for template_slot in range(template_count):
            lengths.append(int(length))
            template_slots.append(template_slot)
    noise_seeds = torch.empty(
        condition_count,
        samples_per_condition,
        dtype=torch.long,
    )
    for condition in range(condition_count):
        for sample in range(samples_per_condition):
            noise_seeds[condition, sample] = (
                seed + condition * 1_000_003 + sample * 10_007
            )
    return validate_counterfactual_pool(
        CounterfactualNumberPool(
            lengths=torch.tensor(lengths, dtype=torch.long),
            template_slots=torch.tensor(template_slots, dtype=torch.long),
            targets=torch.tensor(target_values, dtype=torch.long),
            noise_seeds=noise_seeds,
            max_frames=max_frames,
            prompt_seed=prompt_seed,
            number_style=number_style,
        )
    )


def save_counterfactual_pool(
    pool: CounterfactualNumberPool,
    path: str | Path,
) -> Path:
    pool = validate_counterfactual_pool(pool)
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    torch.save(
        {
            "version": COUNTERFACTUAL_POOL_VERSION,
            "pool_id": pool.pool_id,
            "lengths": pool.lengths,
            "template_slots": pool.template_slots,
            "targets": pool.targets,
            "noise_seeds": pool.noise_seeds,
            "max_frames": pool.max_frames,
            "prompt_seed": pool.prompt_seed,
            "number_style": pool.number_style,
        },
        temporary,
    )
    os.replace(temporary, resolved)
    return resolved


def load_counterfactual_pool(path: str | Path) -> CounterfactualNumberPool:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != COUNTERFACTUAL_POOL_VERSION:
        raise ValueError("Unsupported counterfactual pool version.")
    return validate_counterfactual_pool(
        CounterfactualNumberPool(
            lengths=payload["lengths"],
            template_slots=payload["template_slots"],
            targets=payload["targets"],
            noise_seeds=payload["noise_seeds"],
            max_frames=int(payload["max_frames"]),
            prompt_seed=int(payload["prompt_seed"]),
            number_style=str(payload["number_style"]),
            pool_id=str(payload["pool_id"]),
        )
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first).reshape(-1)
    second = np.asarray(second).reshape(-1)
    if first.shape != second.shape or first.size < 2:
        raise ValueError("Spearman inputs must be matching non-trivial vectors.")
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if first_rank.std() == 0 or second_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def regression_effects(
    counts: np.ndarray,
    pool: CounterfactualNumberPool,
) -> dict[str, float]:
    """OLS count ~ target + length + template fixed effects."""

    values = np.asarray(counts, dtype=np.float64)
    expected = (
        pool.condition_count,
        pool.samples_per_condition,
        len(pool.targets),
    )
    if values.shape != expected:
        raise ValueError(
            f"Counterfactual count shape must be {expected}, got {values.shape}."
        )
    target = np.broadcast_to(
        pool.targets.numpy()[None, None, :],
        expected,
    ).reshape(-1)
    length = np.broadcast_to(
        pool.lengths.numpy()[:, None, None],
        expected,
    ).reshape(-1)
    template = np.broadcast_to(
        pool.template_slots.numpy()[:, None, None],
        expected,
    ).reshape(-1)
    template_values = sorted(set(template.tolist()))
    columns = [np.ones_like(target), target, length]
    for template_value in template_values[1:]:
        columns.append((template == template_value).astype(np.float64))
    design = np.stack(columns, axis=1)
    response = values.reshape(-1)
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    prediction = design @ coefficients
    residual = np.square(response - prediction).sum()
    total = np.square(response - response.mean()).sum()
    target_std = target.std()
    length_std = length.std()
    response_std = response.std()
    return {
        "target_regression_coefficient": float(coefficients[1]),
        "length_regression_coefficient": float(coefficients[2]),
        "target_standardized_coefficient": float(
            coefficients[1] * target_std / max(response_std, 1.0e-12)
        ),
        "length_standardized_coefficient": float(
            coefficients[2] * length_std / max(response_std, 1.0e-12)
        ),
        "regression_r2": float(1.0 - residual / max(total, 1.0e-12)),
        "target_count_spearman": spearman_correlation(target, response),
        "length_count_spearman": spearman_correlation(length, response),
    }


def summarize_counterfactual_counts(
    hard_counts: torch.Tensor,
    soft_counts: torch.Tensor,
    pool: CounterfactualNumberPool,
) -> dict[str, Any]:
    hard = hard_counts.detach().float().cpu().numpy()
    soft = soft_counts.detach().float().cpu().numpy()
    expected = (
        pool.condition_count,
        pool.samples_per_condition,
        len(pool.targets),
    )
    if hard.shape != expected or soft.shape != expected:
        raise ValueError("Counterfactual hard/soft count shapes are invalid.")
    target_grid = np.broadcast_to(
        pool.targets.numpy()[None, None, :],
        expected,
    )
    hard_error = np.abs(hard - target_grid)
    soft_error = np.abs(soft - target_grid)
    return {
        "hard": regression_effects(hard, pool),
        "soft": regression_effects(soft, pool),
        "hard_target_means": {
            str(int(target)): float(hard[..., index].mean())
            for index, target in enumerate(pool.targets.tolist())
        },
        "soft_target_means": {
            str(int(target)): float(soft[..., index].mean())
            for index, target in enumerate(pool.targets.tolist())
        },
        "hard_mae": float(hard_error.mean()),
        "hard_exact_fraction": float((hard_error == 0).mean()),
        "hard_within_one_fraction": float((hard_error <= 1).mean()),
        "soft_mae": float(soft_error.mean()),
        "hard_per_target": {
            str(int(target)): {
                "detected_mean": float(hard[..., index].mean()),
                "mae": float(hard_error[..., index].mean()),
                "exact_fraction": float(
                    (hard_error[..., index] == 0).mean()
                ),
                "within_one_fraction": float(
                    (hard_error[..., index] <= 1).mean()
                ),
            }
            for index, target in enumerate(pool.targets.tolist())
        },
        "hard_within_condition_target_range_mean": float(
            (hard.max(axis=-1) - hard.min(axis=-1)).mean()
        ),
        "soft_within_condition_target_range_mean": float(
            (soft.max(axis=-1) - soft.min(axis=-1)).mean()
        ),
        "soft_hard_abs_difference_mean": float(np.abs(soft - hard).mean()),
    }


__all__ = [
    "CounterfactualNumberPool",
    "counterfactual_pool_id",
    "create_counterfactual_pool",
    "load_counterfactual_pool",
    "regression_effects",
    "render_counterfactual_step_prompt",
    "save_counterfactual_pool",
    "spearman_correlation",
    "summarize_counterfactual_counts",
    "validate_counterfactual_pool",
]
