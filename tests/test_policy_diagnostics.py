from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from mdm_ddpo.lora import LoRAWeight
from mdm_ddpo.policy_diagnostics import (
    advantage_logprob_alignment,
    epsilon_ddim_score_sensitivity,
    effective_lora_delta_norm,
    parse_timestep_buckets,
    summarize_sensitivity_buckets,
    summarize_timestep_log_ratios,
    xstart_ddim_score_sensitivity,
)


class ToyDiffusion:
    alphas_cumprod = np.asarray([0.9, 0.7, 0.4, 0.1], dtype=np.float64)
    alphas_cumprod_prev = np.asarray([1.0, 0.9, 0.7, 0.4], dtype=np.float64)


class PolicyDiagnosticsTest(unittest.TestCase):
    def test_timestep_ratio_summary_keeps_clip_local_to_each_bucket(self):
        buckets = parse_timestep_buckets("1-2,3", minimum=1, maximum=3)
        summary = summarize_timestep_log_ratios(
            torch.tensor([1, 1, 2, 2, 3, 3]),
            torch.tensor([0.0, 0.2, 0.0, 0.0, -0.3, 0.0]),
            clip_range=0.1,
            buckets=buckets,
        )

        self.assertEqual(summary["1-2"]["samples"], 4.0)
        self.assertAlmostEqual(summary["1-2"]["clip_fraction"], 0.25)
        self.assertAlmostEqual(summary["3"]["clip_fraction"], 0.5)

    def test_timestep_bucket_parser_rejects_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            parse_timestep_buckets("1-3,3-5", minimum=1, maximum=5)

    def test_xstart_sensitivity_is_finite_and_bucketed(self):
        sensitivity = xstart_ddim_score_sensitivity(
            ToyDiffusion(),
            eta=1.0,
        )
        buckets = parse_timestep_buckets("1,2-3", minimum=1, maximum=3)
        summary = summarize_sensitivity_buckets(sensitivity, buckets)

        self.assertEqual(set(sensitivity), {1, 2, 3})
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(record["score_sensitivity"]))
                for record in sensitivity.values()
            )
        )
        self.assertEqual(summary["2-3"]["timesteps"], 2.0)

    def test_epsilon_sensitivity_uses_the_same_transition_std(self):
        xstart = xstart_ddim_score_sensitivity(ToyDiffusion(), eta=1.0)
        epsilon = epsilon_ddim_score_sensitivity(ToyDiffusion(), eta=1.0)

        self.assertEqual(set(epsilon), set(xstart))
        for timestep in epsilon:
            self.assertAlmostEqual(
                epsilon[timestep]["transition_std"],
                xstart[timestep]["transition_std"],
            )
            self.assertTrue(
                math.isfinite(epsilon[timestep]["score_sensitivity"])
            )

    def test_advantage_alignment_separates_positive_and_negative_actions(self):
        metrics = advantage_logprob_alignment(
            torch.tensor([1.0, 0.5, -0.5, -1.0]),
            torch.tensor([0.4, 0.2, -0.1, -0.3]),
        )

        self.assertGreater(metrics["positive_negative_gap"], 0.0)
        self.assertGreater(
            metrics["advantage_logprob_delta_correlation"],
            0.9,
        )
        self.assertGreater(
            metrics["advantage_weighted_logprob_delta_mean"],
            0.0,
        )

    def test_effective_lora_norm_ignores_random_a_until_b_changes(self):
        layer = nn.Linear(4, 3, bias=False)
        adapter = LoRAWeight(
            out_features=3,
            in_features=4,
            rank=2,
            alpha=2.0,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        parametrize.register_parametrization(layer, "weight", adapter)

        self.assertEqual(effective_lora_delta_norm(layer), 0.0)
        with torch.no_grad():
            adapter.lora_b.fill_(0.5)
        self.assertGreater(effective_lora_delta_norm(layer), 0.0)


if __name__ == "__main__":
    unittest.main()
