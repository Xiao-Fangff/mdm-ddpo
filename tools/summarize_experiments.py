#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm_ddpo.experiments import (  # noqa: E402
    aggregate_seed_groups,
    summarize_runs,
    write_comparison_tables,
    write_seed_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument(
        "--output-prefix",
        default="outputs/experiment_comparison",
    )
    parser.add_argument("--aggregate-seeds", action="store_true")
    args = parser.parse_args()
    rows = summarize_runs(args.run_dirs)
    csv_path, markdown_path = write_comparison_tables(
        rows,
        args.output_prefix,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    if args.aggregate_seeds:
        seed_rows = aggregate_seed_groups(rows)
        seed_csv, seed_markdown = write_seed_summary(
            seed_rows,
            args.output_prefix + "_seed_summary",
        )
        print(json.dumps(seed_rows, indent=2, sort_keys=True))
        print(f"Seed CSV: {seed_csv}")
        print(f"Seed Markdown: {seed_markdown}")


if __name__ == "__main__":
    main()
