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
    temporal_clustered_soft_count,
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

    def test_temporal_soft_count_clusters_candidates_and_uses_margins(self):
        def candidate(frame, *, kept, lead, length, progress):
            return {
                "foot": "l_foot",
                "start": frame - 2,
                "end": frame,
                "key_frame": frame,
                "lead_margin_measured": lead,
                "min_lead_margin": 1.0,
                "swing_foot_forward_delta": length,
                "min_step_length": 1.0,
                "root_forward_delta": progress,
                "min_root_progress": 1.0,
                "root_progress_gate_applied": True,
                "min_global_root_progress": None,
                **({"reason_kept": "verified"} if kept else {"reason_filtered": "margin"}),
            }

        kept = SimpleNamespace(
            start=8,
            end=10,
            key_frame=10,
            meta=candidate(
                10,
                kept=True,
                lead=2.0,
                length=2.0,
                progress=2.0,
            ),
        )
        track = SimpleNamespace(
            instances=[[kept]],
            meta={
                "filtered_candidates": [[
                    candidate(11, kept=False, lead=0.9, length=2.0, progress=2.0),
                    candidate(30, kept=False, lead=1.0, length=1.0, progress=1.0),
                ]]
            },
        )

        soft, raw, clusters, spacing_mean, spacing_min = (
            temporal_clustered_soft_count(
                track,
                fps=20,
                cluster_gap_seconds=0.15,
                lead_temperature=0.25,
                length_temperature=0.25,
                progress_temperature=0.25,
            )
        )

        self.assertEqual(raw, 3)
        self.assertEqual(clusters, 2)
        self.assertGreater(soft, 1.0)
        self.assertLess(soft, 1.2)
        self.assertEqual(spacing_mean, spacing_min)
        self.assertGreater(spacing_min, 0.5)

    def test_soft_huber_reward_combines_continuous_error_and_hard_exact(self):
        output = compute_step_count_reward(
            torch.tensor([2, 3]),
            torch.tensor([2, 2]),
            soft_count=torch.tensor([2.5, 3.0]),
            target_scale=torch.tensor([1.0, 2.0]),
            mode="soft_huber_exact",
            huber_delta=1.0,
            exact_bonus=0.2,
        )

        torch.testing.assert_close(
            output.reward,
            torch.tensor([0.075, -0.125]),
        )
        torch.testing.assert_close(output.soft_error, torch.tensor([0.5, 1.0]))

    def test_soft_huber_distinguishes_actions_with_the_same_hard_count(self):
        output = compute_step_count_reward(
            torch.tensor([4, 4, 4, 4]),
            torch.tensor([2, 2, 2, 2]),
            soft_count=torch.tensor([3.7, 3.8, 3.9, 4.0]),
            target_scale=1.0,
            mode="soft_huber_exact",
        )

        self.assertEqual(torch.unique(output.reward).numel(), 4)

    def test_detector_reports_high_frequency_ankle_jitter(self):
        smooth = torch.zeros(40, 22, 3)
        smooth[:, 7, 0] = torch.linspace(0.0, 1.0, 40)
        jitter = smooth.clone()
        jitter[:, 7, 0] += 0.1 * torch.tensor(
            [(-1.0) ** frame for frame in range(40)]
        )
        detector = HardStepDetector(backend="rgdno")

        output = detector.detect_xyz(torch.stack([smooth, jitter]))

        self.assertLess(
            output.ankle_high_frequency_ratio[0].item(),
            output.ankle_high_frequency_ratio[1].item(),
        )


if __name__ == "__main__":
    unittest.main()
