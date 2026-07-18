from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mdm_ddpo.calibration import (
    calibration_payload_id,
    compute_reward_calibration,
    load_reward_calibration,
    save_reward_calibration,
    validate_reward_calibration,
)


class RewardCalibrationTest(unittest.TestCase):
    @staticmethod
    def _full_payload():
        retrieval = torch.tensor([[0.0, 1.0, 2.0, 3.0]]).repeat(1024, 1)
        m2m = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).repeat(1024, 1)
        return compute_reward_calibration(
            retrieval,
            m2m,
            retrieval_weight=1.0,
            m2m_weight=0.5,
            metadata={"policy": "original_mdm_without_lora"},
        )

    def test_statistics_capture_scale_correlation_and_ranking_conflict(self):
        payload = self._full_payload()

        self.assertTrue(payload["full_calibration"])
        self.assertGreater(
            payload["components"]["retrieval"]["global_scale"],
            0.0,
        )
        self.assertGreater(
            payload["components"]["retrieval"]["within_group_std_p25"],
            0.0,
        )
        self.assertGreater(
            payload["components"]["retrieval"]["within_group_range_p50"],
            0.0,
        )
        self.assertAlmostEqual(
            payload["relationships"]["global_pearson_correlation"],
            -1.0,
            places=6,
        )
        self.assertAlmostEqual(
            payload["relationships"]["ranking_conflict_fraction"],
            1.0,
        )

    def test_calibration_round_trip_and_fixed_floor_lookup(self):
        payload = self._full_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward_calibration.json"
            save_reward_calibration(payload, path)
            calibration = load_reward_calibration(path)

        self.assertEqual(
            calibration.calibration_id,
            payload["calibration_id"],
        )
        self.assertEqual(
            calibration.global_scale("m2m"),
            payload["components"]["m2m"]["global_scale"],
        )
        self.assertEqual(
            calibration.within_group_std_floor("retrieval", "p25"),
            payload["components"]["retrieval"]["within_group_std_p25"],
        )

    def test_checksum_detects_modified_statistics(self):
        payload = self._full_payload()
        payload["components"]["retrieval"]["global_scale"] += 1.0

        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_reward_calibration(payload)

    def test_incomplete_calibration_is_rejected_for_training(self):
        payload = compute_reward_calibration(
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
            retrieval_weight=1.0,
            m2m_weight=0.5,
        )
        self.assertEqual(payload["calibration_id"], calibration_payload_id(payload))

        with self.assertRaisesRegex(ValueError, "full reward calibration"):
            validate_reward_calibration(payload, require_full=True)


if __name__ == "__main__":
    unittest.main()
