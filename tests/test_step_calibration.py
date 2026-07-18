from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mdm_ddpo.step_calibration import (
    StepRewardCalibration,
    compute_step_reward_calibration,
    load_step_reward_calibration,
    save_step_reward_calibration,
    validate_step_reward_calibration,
)


class StepCalibrationTest(unittest.TestCase):
    @staticmethod
    def _payload():
        rewards = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.5, 0.5, 0.2],
                [0.2, 0.5, 1.0, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        detected = torch.tensor(
            [
                [2, 2, 2, 2],
                [3, 2, 2, 1],
                [1, 2, 3, 2],
                [4, 4, 4, 4],
            ]
        )
        targets = torch.tensor([2, 3, 2, 4])
        return compute_step_reward_calibration(
            rewards,
            detected,
            targets,
            detector_config={"backend": "progressive"},
            reward_config={"mode": "exp"},
        )

    def test_positive_group_quantiles_remain_usable_when_raw_p25_is_zero(self):
        payload = self._payload()

        self.assertEqual(
            payload["component"]["within_group_std_p25_raw"],
            0.0,
        )
        self.assertGreater(
            payload["component"]["within_group_std_p25"],
            0.0,
        )
        self.assertEqual(
            payload["component"]["within_group_zero_std_fraction"],
            0.5,
        )

    def test_round_trip_and_setting_audit(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_reward_calibration.json"
            save_step_reward_calibration(payload, path)
            calibration = load_step_reward_calibration(
                path,
                require_full=False,
            )

        calibration.validate_settings(
            detector_config={"backend": "progressive"},
            reward_config={"mode": "exp"},
        )
        with self.assertRaisesRegex(ValueError, "detector settings"):
            calibration.validate_settings(
                detector_config={"backend": "rgdno"},
                reward_config={"mode": "exp"},
            )

    def test_checksum_detects_mutation(self):
        payload = self._payload()
        payload["component"]["global_scale"] += 0.1
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_step_reward_calibration(payload, require_full=False)


if __name__ == "__main__":
    unittest.main()
