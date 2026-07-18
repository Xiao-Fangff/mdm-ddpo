#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm_ddpo.experiments import (  # noqa: E402
    narrow_followup_pairs,
    summarize_runs,
    top_balanced_runs,
)


def _token(value: float) -> str:
    return f"{value:.0e}".replace("-0", "-")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select two best ablations and create a non-Cartesian LR/clip plan."
        )
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument(
        "--output-script",
        default="outputs/followup_plan/run_followups.sh",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/followup_plan/followup_plan.json",
    )
    parser.add_argument(
        "--run-output-root",
        default="outputs/followup_sweeps",
    )
    args = parser.parse_args()
    summaries = summarize_runs(args.run_dirs)
    selected = top_balanced_runs(summaries, count=2)
    if len(selected) < 2:
        raise SystemExit(
            "Need at least two runs with a feasible balanced checkpoint."
        )

    plan = []
    commands = []
    for rank, summary in enumerate(selected, start=1):
        pairs = narrow_followup_pairs(summary)
        entry = {
            "rank": rank,
            "source_run": summary["run_dir"],
            "source_best_balanced_score": summary["best_balanced_score"],
            "clip_fraction_mean": summary["clip_fraction_mean"],
            "ratio_std_mean": summary["ratio_std_mean"],
            "candidates": [],
        }
        for learning_rate, clip_range in pairs:
            run_name = (
                f"H{rank}_{summary['advantage_mode']}_"
                f"lr{_token(learning_rate)}_clip{_token(clip_range)}"
            )
            candidate = {
                "run_name": run_name,
                "advantage_mode": summary["advantage_mode"],
                "floor_quantile": summary["floor_quantile"],
                "learning_rate": learning_rate,
                "clip_range": clip_range,
                "seed": int(summary["seed"] or 42),
            }
            entry["candidates"].append(candidate)
            command_args = [
                "bash",
                str(PROJECT_ROOT / "scripts" / "run_single_experiment.sh"),
                run_name,
                str(summary["advantage_mode"]),
                str(summary["floor_quantile"] or "p25"),
                str(learning_rate),
                str(clip_range),
                str(candidate["seed"]),
                "0",
                "30",
            ]
            commands.append(" ".join(shlex.quote(value) for value in command_args))
        plan.append(entry)

    output_json = Path(args.output_json).expanduser().resolve()
    output_script = Path(args.output_script).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_script.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_script, "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        handle.write(
            "export OUTPUT_ROOT="
            + shlex.quote(
                str(Path(args.run_output_root).expanduser().resolve())
            )
            + "\n\n"
        )
        for command in commands:
            handle.write(command + "\n")
    output_script.chmod(0o755)
    print(json.dumps(plan, indent=2, sort_keys=True))
    print(f"Review, then run: {output_script}")


if __name__ == "__main__":
    main()
