from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mdm_ddpo.step_reward import (
    HUMANML22_JOINT_NAMES,
    HardStepDetector,
    compute_step_count_reward,
    rgdno_hard_step_count,
)


def _motion_with_ankle_intervals(
    frame_count: int,
    intervals_by_joint: dict[int, list[tuple[int, int]]],
) -> torch.Tensor:
    motion = torch.zeros(frame_count, 22, 3)
    for joint_index, intervals in intervals_by_joint.items():
        delta = torch.zeros(frame_count - 1)
        for start, end in intervals:
            delta[start:end] = 0.1
        motion[1:, joint_index, 0] = torch.cumsum(delta, dim=0)
    return motion


class HardStepRewardTest(unittest.TestCase):
    def test_rgdno_hard_count_matches_reference_transition_convention(self):
        motion = _motion_with_ankle_intervals(
            28,
            {7: [(5, 11)], 8: [(4, 8), (15, 21)]},
        )

        torch.testing.assert_close(
            rgdno_hard_step_count(motion),
            torch.tensor([3]),
        )

    def test_rgdno_uses_true_lengths_not_padded_tail(self):
        motion = _motion_with_ankle_intervals(24, {7: [(5, 11)]})
        padded = motion.clone()
        padded[18:, 8, 0] = torch.arange(6, dtype=torch.float32) * 10

        count = rgdno_hard_step_count(padded, lengths=[18])
        expected = rgdno_hard_step_count(motion[:18])

        torch.testing.assert_close(count, expected)

    def test_progressive_backend_uses_rft_mld_detector_configuration(self):
        calls = []

        def factory(xyz, *, fps):
            calls.append({"xyz": xyz, "fps": fps})
            return SimpleNamespace(xyz=xyz)

        def detector(batch, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(instances=[[object(), object()]])

        hard_detector = HardStepDetector(
            backend="progressive",
            progressive_detector=detector,
            motion_batch_factory=factory,
        )
        joints = torch.zeros(1, 20, len(HUMANML22_JOINT_NAMES), 3)

        counts = hard_detector.count_xyz(joints, lengths=[16])

        self.assertEqual(counts.tolist(), [2])
        self.assertEqual(calls[0]["xyz"].shape, (1, 16, 22, 3))
        self.assertEqual(calls[0]["fps"], 20)
        self.assertEqual(
            calls[1],
            {
                "direction": "forward",
                "frame": "body",
                "foot": "any",
                "step_candidate_source": "lead_offsets",
                "lead_threshold": 0.138,
            },
        )

    def test_exponential_reward_is_shaped_and_masked(self):
        output = compute_step_count_reward(
            torch.tensor([3, 4, 8]),
            torch.tensor([3, 2, -1]),
            mode="exp",
            temperature=1.0,
        )

        torch.testing.assert_close(
            output.reward,
            torch.tensor([1.0, np.exp(-2.0), 0.0], dtype=torch.float32),
        )
        self.assertEqual(output.mask.tolist(), [True, True, False])
        self.assertEqual(output.exact.tolist(), [True, False, False])
        self.assertEqual(output.within_one.tolist(), [True, False, False])

    def test_exact_reward_stays_bounded(self):
        output = compute_step_count_reward(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 4, 2]),
            mode="exact",
        )
        self.assertEqual(output.reward.tolist(), [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
