from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ALLOWED_LEARNING_RATES = (3.0e-5, 1.0e-4, 3.0e-4)
ALLOWED_CLIP_RANGES = (1.0e-4, 3.0e-4, 1.0e-3)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _mean(records: list[dict[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if name in record]
    return float(np.mean(values)) if values else None


def _maximum(records: list[dict[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if name in record]
    return max(values) if values else None


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    with open(run_path / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    training = _read_jsonl(run_path / "metrics.jsonl")
    evaluations = [
        record
        for record in _read_jsonl(run_path / "fixed_eval.jsonl")
        if record.get("event") == "evaluation"
    ]
    balanced_updates = [
        record
        for record in evaluations
        if float(record.get("eval_is_best_balanced", 0.0)) >= 0.5
    ]
    best = (
        max(
            balanced_updates,
            key=lambda record: float(record["eval_balanced_score"]),
        )
        if balanced_updates
        else None
    )
    final = evaluations[-1] if evaluations else None
    step_updates = [
        record
        for record in evaluations
        if float(record.get("eval_is_best_step", 0.0)) >= 0.5
    ]
    best_step = (
        max(
            step_updates,
            key=lambda record: float(record["eval_step_reward_delta"]),
        )
        if step_updates
        else None
    )
    step_retrieval_weight = config.get("step_advantage_retrieval_weight")
    step_m2m_weight = config.get("step_advantage_m2m_weight")
    step_weight = config.get("step_advantage_step_weight")
    return {
        "run": run_path.name,
        "run_dir": str(run_path),
        "seed": config.get("seed"),
        "advantage_mode": config.get("advantage_mode"),
        "floor_quantile": config.get("advantage_std_floor_quantile"),
        "learning_rate": config.get("learning_rate"),
        "clip_range": config.get("clip_range"),
        "anchor_grad_ratio_target": config.get("anchor_auto_grad_ratio", 0.0),
        "step_reward_enabled": config.get("enable_step_reward", False),
        "step_data_ratio": config.get("step_data_ratio", 0.0),
        "step_samples_per_prompt": config.get("step_samples_per_prompt"),
        "step_balanced_sampling": config.get("step_balanced_sampling", False),
        "step_retrieval_advantage_weight": (
            config.get("advantage_retrieval_weight", 0.0)
            if step_retrieval_weight is None
            else step_retrieval_weight
        ),
        "step_m2m_advantage_weight": (
            config.get("advantage_m2m_weight", 0.0)
            if step_m2m_weight is None
            else step_m2m_weight
        ),
        "step_advantage_weight": (
            config.get("advantage_step_weight", 0.0)
            if step_weight is None
            else step_weight
        ),
        "epochs_completed": len(training),
        "global_step": training[-1].get("global_step") if training else None,
        "clip_fraction_mean": _mean(training, "clip_fraction"),
        "ratio_std_mean": _mean(training, "ratio_std"),
        "log_ratio_abs_max": _maximum(training, "log_ratio_max"),
        "skipped_updates": sum(
            float(record.get("skipped_updates", 0.0)) for record in training
        ),
        "has_balanced_improvement": best is not None,
        "best_epoch": best.get("epoch") if best else -1,
        "best_balanced_score": (
            best.get("eval_balanced_score") if best else 0.0
        ),
        "best_retrieval_delta": (
            best.get("eval_reward_retrieval_delta") if best else 0.0
        ),
        "best_m2m_delta": best.get("eval_reward_m2m_delta") if best else 0.0,
        "best_balanced_se": (
            best.get("eval_balanced_score_bootstrap_se") if best else 0.0
        ),
        "final_balanced_score": (
            final.get("eval_balanced_score") if final else None
        ),
        "final_retrieval_delta": (
            final.get("eval_reward_retrieval_delta") if final else None
        ),
        "final_m2m_delta": (
            final.get("eval_reward_m2m_delta") if final else None
        ),
        "best_step_epoch": best_step.get("epoch") if best_step else -1,
        "best_step_reward_delta": (
            best_step.get("eval_step_reward_delta") if best_step else None
        ),
        "best_step_mae_delta": (
            best_step.get("eval_step_mae_delta") if best_step else None
        ),
        "best_step_exact_delta": (
            best_step.get("eval_step_exact_fraction_delta")
            if best_step
            else None
        ),
        "best_step_within_one_delta": (
            best_step.get("eval_step_within_one_fraction_delta")
            if best_step
            else None
        ),
        "final_step_reward_delta": (
            final.get("eval_step_reward_delta") if final else None
        ),
        "final_step_mae_delta": (
            final.get("eval_step_mae_delta") if final else None
        ),
        "final_step_exact_delta": (
            final.get("eval_step_exact_fraction_delta") if final else None
        ),
        "final_step_within_one_delta": (
            final.get("eval_step_within_one_fraction_delta") if final else None
        ),
        "best_balanced_checkpoint": str(run_path / "best_balanced.pt"),
        "best_step_checkpoint": str(run_path / "best_step.pt"),
    }


def summarize_runs(run_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [summarize_run(path) for path in run_dirs]


def write_comparison_tables(
    rows: list[dict[str, Any]],
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    columns = [
        "run",
        "seed",
        "advantage_mode",
        "floor_quantile",
        "learning_rate",
        "clip_range",
        "anchor_grad_ratio_target",
        "step_reward_enabled",
        "step_data_ratio",
        "step_samples_per_prompt",
        "step_balanced_sampling",
        "step_retrieval_advantage_weight",
        "step_m2m_advantage_weight",
        "step_advantage_weight",
        "epochs_completed",
        "best_epoch",
        "has_balanced_improvement",
        "best_balanced_score",
        "best_retrieval_delta",
        "best_m2m_delta",
        "best_balanced_se",
        "best_step_epoch",
        "best_step_reward_delta",
        "best_step_mae_delta",
        "best_step_exact_delta",
        "best_step_within_one_delta",
        "clip_fraction_mean",
        "ratio_std_mean",
        "log_ratio_abs_max",
        "skipped_updates",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    "" if row.get(column) is None else str(row.get(column))
                    for column in columns
                )
                + " |\n"
            )
    return csv_path, markdown_path


def aggregate_seed_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_names = (
        "advantage_mode",
        "floor_quantile",
        "learning_rate",
        "clip_range",
        "anchor_grad_ratio_target",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_names)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        valid = [
            row
            for row in group
            if row.get("best_balanced_score") is not None
            and int(row.get("epochs_completed") or 0) >= 30
        ]
        record = dict(zip(group_names, key))
        record["requested_seed_count"] = len(group)
        record["feasible_seed_count"] = len(valid)
        for source, target in (
            ("best_retrieval_delta", "retrieval_delta"),
            ("best_m2m_delta", "m2m_delta"),
            ("best_balanced_score", "balanced_score"),
        ):
            values = np.asarray(
                [float(row[source]) for row in valid],
                dtype=np.float64,
            )
            record[f"{target}_mean"] = (
                float(values.mean()) if len(values) else None
            )
            record[f"{target}_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0 if len(values) == 1 else None
            )
        record["both_components_nonnegative_fraction"] = (
            float(
                np.mean(
                    [
                        float(row["best_retrieval_delta"]) >= 0
                        and float(row["best_m2m_delta"]) >= 0
                        for row in valid
                    ]
                )
            )
            if valid
            else None
        )
        record["three_seed_acceptance"] = bool(
            len(valid) == 3
            and record["retrieval_delta_mean"] >= 0
            and record["m2m_delta_mean"] >= 0
            and record["balanced_score_mean"] > 0
        )
        output.append(record)
    return output


def write_seed_summary(
    rows: list[dict[str, Any]],
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    columns = [
        "advantage_mode",
        "floor_quantile",
        "learning_rate",
        "clip_range",
        "anchor_grad_ratio_target",
        "requested_seed_count",
        "feasible_seed_count",
        "retrieval_delta_mean",
        "retrieval_delta_se",
        "m2m_delta_mean",
        "m2m_delta_se",
        "balanced_score_mean",
        "balanced_score_se",
        "both_components_nonnegative_fraction",
        "three_seed_acceptance",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    "" if row.get(column) is None else str(row.get(column))
                    for column in columns
                )
                + " |\n"
            )
    return csv_path, markdown_path


def top_balanced_runs(
    rows: list[dict[str, Any]],
    count: int = 2,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if bool(row.get("has_balanced_improvement", False))
        and int(row.get("epochs_completed") or 0) >= 30
        and float(row["best_balanced_score"]) > 0
    ]
    return sorted(
        eligible,
        key=lambda row: float(row["best_balanced_score"]),
        reverse=True,
    )[:count]


def _neighbor(
    values: tuple[float, ...],
    current: float,
    offset: int,
) -> float | None:
    index = min(range(len(values)), key=lambda item: abs(values[item] - current))
    target = index + offset
    return values[target] if 0 <= target < len(values) else None


def narrow_followup_pairs(summary: dict[str, Any]) -> list[tuple[float, float]]:
    """Choose a diagnostic subset of the requested LR/clip candidates."""
    learning_rate = float(summary["learning_rate"])
    clip_range = float(summary["clip_range"])
    clip_fraction = float(summary.get("clip_fraction_mean") or 0.0)
    ratio_std = float(summary.get("ratio_std_mean") or 0.0)
    lower_lr = _neighbor(ALLOWED_LEARNING_RATES, learning_rate, -1)
    higher_lr = _neighbor(ALLOWED_LEARNING_RATES, learning_rate, 1)
    narrower_clip = _neighbor(ALLOWED_CLIP_RANGES, clip_range, -1)
    wider_clip = _neighbor(ALLOWED_CLIP_RANGES, clip_range, 1)
    candidates: list[tuple[float | None, float | None]]
    if clip_fraction >= 0.25 or ratio_std >= 2.0 * clip_range:
        candidates = [
            (lower_lr, clip_range),
            (learning_rate, wider_clip),
            (lower_lr, wider_clip),
        ]
    elif clip_fraction <= 0.02 and ratio_std <= 0.5 * clip_range:
        candidates = [
            (higher_lr, clip_range),
            (learning_rate, wider_clip),
            (higher_lr, wider_clip),
        ]
    else:
        candidates = [
            (lower_lr, clip_range),
            (higher_lr, clip_range),
            (learning_rate, narrower_clip),
            (learning_rate, wider_clip),
        ]
    result: list[tuple[float, float]] = []
    for candidate_lr, candidate_clip in candidates:
        if candidate_lr is None or candidate_clip is None:
            continue
        pair = (candidate_lr, candidate_clip)
        if pair == (learning_rate, clip_range) or pair in result:
            continue
        result.append(pair)
    return result
