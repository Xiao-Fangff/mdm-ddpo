from __future__ import annotations

import re
from typing import Any, Sequence

import torch


_NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_NUMBER_PATTERN = re.compile(
    r"\b(?:\d+|" + "|".join(_NUMBER_WORD_VALUES) + r")\b"
)
_STEP_UNIT_PATTERN = re.compile(r"\b(?:steps?|times?)\b")


def parse_requested_step_count(text: str) -> int | None:
    """Parse the intended count from an original step-control caption."""
    normalized = str(text).strip().lower()
    if not normalized or _STEP_UNIT_PATTERN.search(normalized) is None:
        return None
    values: list[int] = []
    for token in _NUMBER_PATTERN.findall(normalized):
        values.append(
            int(token) if token.isdigit() else _NUMBER_WORD_VALUES[token]
        )
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else None


def summarize_step_detection(
    target_steps: Sequence[int] | torch.Tensor,
    detected_steps: Sequence[int] | torch.Tensor,
    *,
    requested_targets: Sequence[int],
) -> dict[str, Any]:
    targets = torch.as_tensor(target_steps, dtype=torch.long).reshape(-1)
    detected = torch.as_tensor(detected_steps, dtype=torch.long).reshape(-1)
    if targets.shape != detected.shape or targets.numel() == 0:
        raise ValueError("Target and detected step vectors must be non-empty and match.")
    target_values = tuple(sorted(set(int(value) for value in requested_targets)))
    if not target_values:
        raise ValueError("At least one requested target is required.")
    observed_detected = tuple(sorted(set(int(value) for value in detected.tolist())))
    confusion_counts: dict[str, dict[str, int]] = {}
    per_target: dict[str, dict[str, float | int]] = {}
    for target in target_values:
        active = targets == target
        target_detected = detected[active]
        if target_detected.numel() == 0:
            raise ValueError(f"No GT/reference motions found for target {target}.")
        errors = (target_detected - target).abs().float()
        confusion_counts[str(target)] = {
            str(value): int((target_detected == value).sum())
            for value in observed_detected
        }
        per_target[str(target)] = {
            "samples": int(target_detected.numel()),
            "detected_mean": target_detected.float().mean().item(),
            "mae": errors.mean().item(),
            "exact_accuracy": (errors == 0).float().mean().item(),
            "within_one_accuracy": (errors <= 1).float().mean().item(),
        }
    errors = (detected - targets).abs().float()
    return {
        "samples": int(targets.numel()),
        "target_values": list(target_values),
        "detected_values": list(observed_detected),
        "overall": {
            "detected_mean": detected.float().mean().item(),
            "target_mean": targets.float().mean().item(),
            "mae": errors.mean().item(),
            "exact_accuracy": (errors == 0).float().mean().item(),
            "within_one_accuracy": (errors <= 1).float().mean().item(),
        },
        "per_target": per_target,
        "confusion_counts": confusion_counts,
    }


__all__ = [
    "parse_requested_step_count",
    "summarize_step_detection",
]
