from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mdm_ddpo.step_data import StepSampleRecord
from mdm_ddpo.step_probe import (
    create_counterfactual_pool,
    load_counterfactual_pool,
    regression_effects,
    save_counterfactual_pool,
    summarize_counterfactual_counts,
)


class CounterfactualStepProbeTest(unittest.TestCase):
    @staticmethod
    def _records():
        return [
            StepSampleRecord(
                manifest_index=target * 10 + slot,
                sample_id=f"t{target}-{slot}",
                target_steps=target,
                feature_path=Path("unused.npy"),
                length=length,
            )
            for target in (1, 2, 3)
            for slot, length in enumerate((40, 50, 60, 70))
        ]

    def test_pool_round_trip_and_factorial_design(self):
        pool = create_counterfactual_pool(
            self._records(),
            targets=(1, 2, 3),
            condition_count=12,
            samples_per_condition=2,
            max_frames=80,
            seed=5,
            prompt_seed=7,
        )

        self.assertEqual(len(set(pool.template_slots.tolist())), 6)
        for template in set(pool.template_slots.tolist()):
            active = pool.template_slots == template
            self.assertEqual(len(set(pool.lengths[active].tolist())), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.pt"
            save_counterfactual_pool(pool, path)
            loaded = load_counterfactual_pool(path)
        self.assertEqual(pool.pool_id, loaded.pool_id)

    def test_regression_recovers_target_and_length_effects(self):
        pool = create_counterfactual_pool(
            self._records(),
            targets=(1, 2, 3),
            condition_count=12,
            samples_per_condition=2,
            max_frames=80,
            seed=5,
            prompt_seed=7,
        )
        target = pool.targets.float()[None, None, :]
        length = pool.lengths.float()[:, None, None]
        counts = (0.5 * target + 0.02 * length).expand(
            pool.condition_count,
            pool.samples_per_condition,
            len(pool.targets),
        )

        effects = regression_effects(counts.numpy(), pool)
        summary = summarize_counterfactual_counts(
            counts.round().long(),
            counts,
            pool,
        )

        self.assertAlmostEqual(effects["target_regression_coefficient"], 0.5)
        self.assertAlmostEqual(effects["length_regression_coefficient"], 0.02)
        self.assertGreater(summary["soft"]["target_count_spearman"], 0.0)
        self.assertIn("detected_mean", summary["hard_per_target"]["1"])


if __name__ == "__main__":
    unittest.main()
