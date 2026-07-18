#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.runtime import bootstrap_external_repositories, resolve_device
from mdm_ddpo.step_diagnostics import (
    parse_requested_step_count,
    summarize_step_detection,
)
from mdm_ddpo.step_reward import HardStepDetector


LOGGER = logging.getLogger("validate_step_detector_gt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the hard step detector on stored reference XYZ motions "
            "against the step count requested by each original caption."
        )
    )
    parser.add_argument("--output", default="artifacts/step_gt_detector_validation.json")
    parser.add_argument("--markdown-output", default="")
    parser.add_argument("--step-data-manifest", default=TrainConfig.step_data_manifest)
    parser.add_argument("--step-motion-root", default=TrainConfig.step_motion_root)
    parser.add_argument("--step-detector-root", default=TrainConfig.step_detector_root)
    parser.add_argument("--step-targets", default="1,2,3,4,5,6")
    parser.add_argument("--max-samples-per-target", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--step-detector-backend",
        choices=["progressive", "rgdno"],
        default="progressive",
    )
    parser.add_argument("--step-detector-fps", type=int, default=20)
    parser.add_argument("--step-detector-lead-threshold", type=float, default=0.138)
    parser.add_argument("--step-detector-rgdno-threshold", type=float, default=0.005)
    parser.add_argument("--motionrft-root", default=TrainConfig.motionrft_root)
    parser.add_argument("--mdm-root", default=TrainConfig.mdm_root)
    return parser


def _resolve_asset(raw_value: str, manifest_path: Path, root: Path) -> Path:
    raw = Path(raw_value).expanduser()
    candidates = [raw] if raw.is_absolute() else []
    candidates.extend([root / raw, manifest_path.parent / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve reference motion {raw_value!r}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = Path(args.step_data_manifest).expanduser().resolve()
    motion_root = Path(args.step_motion_root).expanduser().resolve()
    requested_targets = {
        int(value.strip())
        for value in args.step_targets.split(",")
        if value.strip()
    }
    grouped: dict[int, list[dict[str, Any]]] = {
        target: [] for target in sorted(requested_targets)
    }
    with open(manifest_path, encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            requested = parse_requested_step_count(str(row.get("prompt", "")))
            if requested not in grouped:
                continue
            grouped[requested].append(
                {
                    "line_index": line_index,
                    "sample_id": str(row["sample_id"]),
                    "target_steps": requested,
                    "manifest_detected_steps": int(row["detected_steps"]),
                    "motion_path": _resolve_asset(
                        str(row["motion_path"]),
                        manifest_path,
                        motion_root,
                    ),
                }
            )
    selected: list[dict[str, Any]] = []
    for target, rows in grouped.items():
        if not rows:
            LOGGER.warning(
                "Manifest has no parseable original caption for target %d; "
                "the confusion matrix will mark this target as missing.",
                target,
            )
            continue
        rows.sort(key=lambda row: row["sample_id"])
        random.Random(args.seed + target * 1_000_003).shuffle(rows)
        if args.max_samples_per_target > 0:
            rows = rows[: args.max_samples_per_target]
        selected.extend(rows)
    return selected


def _detect_rows(
    rows: list[dict[str, Any]],
    detector: HardStepDetector,
    *,
    batch_size: int,
    device: torch.device,
) -> list[int]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    detected: list[int] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        motions = [np.load(row["motion_path"]).astype(np.float32) for row in batch_rows]
        for row, motion in zip(batch_rows, motions):
            if motion.ndim != 3 or motion.shape[1:] != (22, 3):
                raise ValueError(
                    f"Reference XYZ motion must be [T,22,3], got {motion.shape} "
                    f"for {row['motion_path']}."
                )
        lengths = [len(motion) for motion in motions]
        padded = np.zeros((len(motions), max(lengths), 22, 3), dtype=np.float32)
        for index, motion in enumerate(motions):
            padded[index, : len(motion)] = motion
        counts = detector.count_xyz(
            torch.from_numpy(padded).to(device),
            lengths=lengths,
        )
        detected.extend(int(value) for value in counts.detach().cpu().tolist())
        LOGGER.info("Validated %d/%d reference motions", min(start + batch_size, len(rows)), len(rows))
    return detected


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Step detector GT/reference validation",
        "",
        f"Samples: {payload['samples']}",
        f"Missing requested targets: {payload.get('missing_target_values', [])}",
        "",
        "| target | samples | detected mean | MAE | exact | within-one |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in payload["target_values"]:
        row = payload["per_target"][str(target)]
        lines.append(
            f"| {target} | {row['samples']} | {row['detected_mean']:.4f} | "
            f"{row['mae']:.4f} | {row['exact_accuracy']:.4f} | "
            f"{row['within_one_accuracy']:.4f} |"
        )
    detected_values = payload["detected_values"]
    lines.extend(
        [
            "",
            "## Target–detected confusion counts",
            "",
            "| target \\ detected | "
            + " | ".join(str(value) for value in detected_values)
            + " |",
            "| ---: | " + " | ".join("---:" for _ in detected_values) + " |",
        ]
    )
    for target in payload["target_values"]:
        counts = payload["confusion_counts"][str(target)]
        lines.append(
            f"| {target} | "
            + " | ".join(str(counts[str(value)]) for value in detected_values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        mdm_root=args.mdm_root,
        motionrft_root=args.motionrft_root,
    )
    bootstrap_external_repositories(config)
    device = resolve_device(args.device)
    rows = _load_rows(args)
    detector = HardStepDetector(
        backend=args.step_detector_backend,
        fps=args.step_detector_fps,
        motion_rule_root=args.step_detector_root,
        lead_threshold=args.step_detector_lead_threshold,
        rgdno_threshold=args.step_detector_rgdno_threshold,
    )
    detected = _detect_rows(
        rows,
        detector,
        batch_size=args.batch_size,
        device=device,
    )
    targets = [int(row["target_steps"]) for row in rows]
    requested_targets = sorted(
        {
            int(value.strip())
            for value in args.step_targets.split(",")
            if value.strip()
        }
    )
    payload = summarize_step_detection(
        targets,
        detected,
        requested_targets=sorted(set(targets)),
    )
    manifest_detected = [int(row["manifest_detected_steps"]) for row in rows]
    payload.update(
        {
            "detector": {
                "backend": args.step_detector_backend,
                "fps": args.step_detector_fps,
                "lead_threshold": args.step_detector_lead_threshold,
                "rgdno_threshold": args.step_detector_rgdno_threshold,
            },
            "manifest": str(Path(args.step_data_manifest).expanduser().resolve()),
            "requested_target_values": requested_targets,
            "missing_target_values": sorted(set(requested_targets) - set(targets)),
            "manifest_reproduction_accuracy": sum(
                first == second
                for first, second in zip(manifest_detected, detected)
            )
            / len(rows),
        }
    )
    output = Path(args.output).expanduser().resolve()
    markdown_output = (
        Path(args.markdown_output).expanduser().resolve()
        if args.markdown_output
        else output.with_suffix(".md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output)
    temporary_markdown = markdown_output.with_suffix(markdown_output.suffix + ".tmp")
    temporary_markdown.write_text(_render_markdown(payload), encoding="utf-8")
    temporary_markdown.replace(markdown_output)
    LOGGER.info("Saved detector validation: %s", output)
    LOGGER.info("Saved detector validation table: %s", markdown_output)
    print(json.dumps(payload["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
