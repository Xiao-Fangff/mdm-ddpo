from __future__ import annotations

import unittest

import torch

from mdm_ddpo.step_diagnostics import (
    parse_requested_step_count,
    summarize_step_detection,
)


class StepDiagnosticsTest(unittest.TestCase):
    def test_requested_count_parser_handles_digits_and_words(self):
        self.assertEqual(parse_requested_step_count("walk forward 4 steps"), 4)
        self.assertEqual(parse_requested_step_count("take six steps ahead"), 6)
        self.assertEqual(parse_requested_step_count("step forward 3 times"), 3)
        self.assertIsNone(parse_requested_step_count("walk forward"))

    def test_detection_summary_builds_target_confusion_and_metrics(self):
        summary = summarize_step_detection(
            torch.tensor([1, 1, 2, 2]),
            torch.tensor([1, 2, 1, 2]),
            requested_targets=(1, 2),
        )

        self.assertEqual(summary["confusion_counts"]["1"], {"1": 1, "2": 1})
        self.assertEqual(summary["confusion_counts"]["2"], {"1": 1, "2": 1})
        self.assertEqual(summary["overall"]["exact_accuracy"], 0.5)
        self.assertEqual(summary["per_target"]["1"]["within_one_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
