#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


FIXED_METRICS = (
    "eval_step_reward_delta",
    "eval_step_exact_fraction_delta",
    "eval_step_within_one_fraction_delta",
    "eval_step_mae_delta",
)
ADVANTAGE_METRICS = (
    "component_advantage_step_zero_variance_prompt_fraction",
    "component_advantage_step_group_std_median",
    "component_advantage_step_contribution_mean_abs",
    "component_advantage_retrieval_contribution_mean_abs",
    "component_advantage_m2m_contribution_mean_abs",
    "component_advantage_retrieval_step_conflict_fraction",
    "component_advantage_m2m_step_conflict_fraction",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _target_table(event: dict[str, Any], *, baseline: bool) -> dict[str, Any]:
    suffix = "baseline" if baseline else "current"
    output: dict[str, Any] = {}
    for target in range(1, 7):
        rows = [
            row for row in event["prompts"]
            if int(row["target_steps"]) == target
        ]
        if not rows:
            continue
        output[str(target)] = {
            "detected_mean": statistics.mean(
                float(row[f"detected_mean_{suffix}"]) for row in rows
            ),
            "mae": statistics.mean(
                float(row[f"mae_{suffix}"]) for row in rows
            ),
            "exact_accuracy": statistics.mean(
                float(row[f"exact_fraction_{suffix}"]) for row in rows
            ),
            "within_one_accuracy": statistics.mean(
                float(row[f"within_one_fraction_{suffix}"]) for row in rows
            ),
        }
    return output


def analyze_run(run_dir: Path) -> dict[str, Any]:
    training = _read_jsonl(run_dir / "metrics.jsonl")
    fixed = [
        record
        for record in _read_jsonl(run_dir / "fixed_eval.jsonl")
        if record.get("event") == "evaluation"
    ]
    per_prompt = _read_jsonl(run_dir / "fixed_step_eval_per_prompt.jsonl")
    if not training or not fixed or not per_prompt:
        raise ValueError(f"Incomplete step experiment: {run_dir}")
    best = max(fixed, key=lambda record: float(record["eval_step_reward_delta"]))
    final = fixed[-1]
    event_by_epoch = {int(record["epoch"]): record for record in per_prompt}
    advantage = {}
    for name in ADVANTAGE_METRICS:
        values = [float(record[name]) for record in training]
        advantage[name] = {
            "mean": statistics.mean(values),
            "last_10_mean": statistics.mean(values[-10:]),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    acceptance_points = [
        record
        for record in fixed
        if float(record["eval_step_mae_delta"]) < 0
        and float(record["eval_step_exact_fraction_delta"]) > 0
        and float(record["eval_step_within_one_fraction_delta"]) > 0
    ]
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "epochs": len(training),
        "validation_points": len(fixed),
        "acceptance_points": len(acceptance_points),
        "best_step_epoch": int(best["epoch"]),
        "fixed_best_step": {name: float(best[name]) for name in FIXED_METRICS},
        "fixed_final": {name: float(final[name]) for name in FIXED_METRICS},
        "advantage": advantage,
        "targets_baseline": _target_table(event_by_epoch[-1], baseline=True),
        "targets_best_step": _target_table(
            event_by_epoch[int(best["epoch"])],
            baseline=False,
        ),
        "targets_final": _target_table(
            event_by_epoch[int(final["epoch"])],
            baseline=False,
        ),
    }


def _format(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(runs: list[dict[str, Any]]) -> str:
    lines = [
        "# K16 step M2M ablation analysis",
        "",
        "## Fixed step validation",
        "",
        "| run | state | epoch | reward delta | exact delta | within-one delta | MAE delta | acceptance points |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        for state, values, epoch in (
            ("best_step", run["fixed_best_step"], run["best_step_epoch"]),
            ("final", run["fixed_final"], run["epochs"] - 1),
        ):
            lines.append(
                f"| {run['run']} | {state} | {epoch} | "
                f"{_format(values['eval_step_reward_delta'])} | "
                f"{_format(values['eval_step_exact_fraction_delta'])} | "
                f"{_format(values['eval_step_within_one_fraction_delta'])} | "
                f"{_format(values['eval_step_mae_delta'])} | "
                f"{run['acceptance_points']}/{run['validation_points']} |"
            )
    lines.extend(
        [
            "",
            "## Training advantage statistics (mean over all epochs)",
            "",
            "| run | zero variance | step group std median | step contribution | retrieval contribution | M2M contribution | retrieval-step conflict | M2M-step conflict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        advantage = run["advantage"]
        get = lambda name: advantage[name]["mean"]
        lines.append(
            f"| {run['run']} | "
            f"{_format(get('component_advantage_step_zero_variance_prompt_fraction'))} | "
            f"{_format(get('component_advantage_step_group_std_median'))} | "
            f"{_format(get('component_advantage_step_contribution_mean_abs'))} | "
            f"{_format(get('component_advantage_retrieval_contribution_mean_abs'))} | "
            f"{_format(get('component_advantage_m2m_contribution_mean_abs'))} | "
            f"{_format(get('component_advantage_retrieval_step_conflict_fraction'))} | "
            f"{_format(get('component_advantage_m2m_step_conflict_fraction'))} |"
        )
    for run in runs:
        lines.extend(
            [
                "",
                f"## Per-target fixed step eval: {run['run']}",
                "",
                "| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for target in range(1, 7):
            baseline = run["targets_baseline"][str(target)]
            final = run["targets_final"][str(target)]
            lines.append(
                f"| {target} | {_format(baseline['detected_mean'])} | "
                f"{_format(final['detected_mean'])} | {_format(baseline['mae'])} | "
                f"{_format(final['mae'])} | {_format(baseline['exact_accuracy'])} | "
                f"{_format(final['exact_accuracy'])} | "
                f"{_format(baseline['within_one_accuracy'])} | "
                f"{_format(final['within_one_accuracy'])} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument(
        "--output",
        default="reports/k16_step_m2m_ablation_analysis.json",
    )
    args = parser.parse_args()
    runs = [analyze_run(Path(value).expanduser()) for value in args.run_dirs]
    output = Path(args.output).expanduser().resolve()
    markdown = output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(runs), encoding="utf-8")
    print(f"JSON: {output}")
    print(f"Markdown: {markdown}")


if __name__ == "__main__":
    main()
