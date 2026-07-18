from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdm_ddpo.experiments import (
    ALLOWED_CLIP_RANGES,
    ALLOWED_LEARNING_RATES,
    aggregate_seed_groups,
    narrow_followup_pairs,
    summarize_run,
    top_balanced_runs,
    write_comparison_tables,
)


class ExperimentUtilitiesTest(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, records) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def test_summary_selects_best_feasible_balanced_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "A4"
            run.mkdir()
            (run / "config.json").write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "advantage_mode": "component_shrink",
                        "advantage_std_floor_quantile": "p25",
                        "learning_rate": 1.0e-4,
                        "clip_range": 1.0e-4,
                        "anchor_auto_grad_ratio": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                run / "metrics.jsonl",
                [
                    {
                        "global_step": 1,
                        "clip_fraction": 0.1,
                        "ratio_std": 2.0e-4,
                        "log_ratio_max": 4.0e-4,
                    },
                    {
                        "global_step": 2,
                        "clip_fraction": 0.2,
                        "ratio_std": 3.0e-4,
                        "log_ratio_max": 5.0e-4,
                    },
                ],
            )
            self._write_jsonl(
                run / "fixed_eval.jsonl",
                [
                    {
                        "event": "evaluation",
                        "epoch": 4,
                        "eval_feasible": 1.0,
                        "eval_is_best_balanced": 1.0,
                        "eval_balanced_score": 0.2,
                        "eval_reward_retrieval_delta": 0.01,
                        "eval_reward_m2m_delta": 0.02,
                        "eval_balanced_score_bootstrap_se": 0.03,
                        "eval_is_best_step": 1.0,
                        "eval_step_reward_delta": 0.08,
                        "eval_step_mae_delta": -0.3,
                        "eval_step_exact_fraction_delta": 0.1,
                    },
                    {
                        "event": "evaluation",
                        "epoch": 9,
                        "eval_feasible": 0.0,
                        "eval_is_best_balanced": 0.0,
                        "eval_balanced_score": 0.9,
                        "eval_reward_retrieval_delta": -0.2,
                        "eval_reward_m2m_delta": 0.5,
                    },
                ],
            )

            summary = summarize_run(run)
            csv_path, markdown_path = write_comparison_tables(
                [summary],
                Path(directory) / "comparison",
            )
            tables_exist = csv_path.exists() and markdown_path.exists()

        self.assertEqual(summary["best_epoch"], 4)
        self.assertEqual(summary["best_balanced_score"], 0.2)
        self.assertEqual(summary["best_step_reward_delta"], 0.08)
        self.assertEqual(summary["best_step_mae_delta"], -0.3)
        self.assertAlmostEqual(summary["clip_fraction_mean"], 0.15)
        self.assertTrue(tables_exist)

    def test_followup_plan_is_diagnostic_subset_not_cartesian_product(self):
        summary = {
            "learning_rate": 1.0e-4,
            "clip_range": 1.0e-4,
            "clip_fraction_mean": 0.4,
            "ratio_std_mean": 4.0e-4,
        }

        pairs = narrow_followup_pairs(summary)

        self.assertGreater(len(pairs), 0)
        self.assertLess(len(pairs), 9)
        self.assertTrue(
            all(pair[0] in ALLOWED_LEARNING_RATES for pair in pairs)
        )
        self.assertTrue(all(pair[1] in ALLOWED_CLIP_RANGES for pair in pairs))
        self.assertIn((3.0e-5, 1.0e-4), pairs)
        self.assertIn((1.0e-4, 3.0e-4), pairs)

    def test_top_balanced_runs_ignores_runs_without_feasible_checkpoint(self):
        rows = [
            {
                "run": "bad",
                "best_balanced_score": 0.0,
                "has_balanced_improvement": False,
                "epochs_completed": 30,
            },
            {
                "run": "second",
                "best_balanced_score": 0.1,
                "has_balanced_improvement": True,
                "epochs_completed": 30,
            },
            {
                "run": "first",
                "best_balanced_score": 0.2,
                "has_balanced_improvement": True,
                "epochs_completed": 30,
            },
        ]

        selected = top_balanced_runs(rows, count=2)

        self.assertEqual([row["run"] for row in selected], ["first", "second"])

    def test_three_seed_aggregation_checks_both_component_means(self):
        rows = [
            {
                "advantage_mode": "component_shrink",
                "floor_quantile": "p25",
                "learning_rate": 1.0e-4,
                "clip_range": 1.0e-4,
                "anchor_grad_ratio_target": 0.1,
                "best_retrieval_delta": retrieval,
                "best_m2m_delta": m2m,
                "best_balanced_score": balanced,
                "has_balanced_improvement": True,
                "epochs_completed": 30,
            }
            for retrieval, m2m, balanced in (
                (0.01, 0.02, 0.2),
                (0.02, -0.01, 0.1),
                (0.00, 0.01, 0.05),
            )
        ]

        summary = aggregate_seed_groups(rows)[0]

        self.assertEqual(summary["feasible_seed_count"], 3)
        self.assertGreater(summary["retrieval_delta_mean"], 0.0)
        self.assertGreater(summary["m2m_delta_mean"], 0.0)
        self.assertTrue(summary["three_seed_acceptance"])


if __name__ == "__main__":
    unittest.main()
