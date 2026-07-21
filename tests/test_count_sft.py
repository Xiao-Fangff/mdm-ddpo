from __future__ import annotations

import unittest

import torch

from mdm_ddpo.sft import (
    CountSFTConfig,
    calibrate_loss_lambda,
    foot_acceleration_consistency_per_sample,
    humanml_foot_position_channels,
)
from mdm_ddpo.policy_io import (
    policy_checkpoint_id,
    trainable_policy_state_id,
)


class CountSFTNumericsTest(unittest.TestCase):
    def test_foot_channels_match_humanml_local_position_layout(self):
        self.assertEqual(
            humanml_foot_position_channels(),
            (
                22,
                23,
                24,
                25,
                26,
                27,
                31,
                32,
                33,
                34,
                35,
                36,
            ),
        )

    def test_acceleration_consistency_is_masked_and_differentiable(self):
        target = torch.zeros(2, 263, 1, 8)
        predicted = target.clone().requires_grad_(True)
        channels = humanml_foot_position_channels()
        with torch.no_grad():
            # Only sample zero has a valid perturbation; sample one's frame 7
            # lies beyond its declared length and must not contribute.
            predicted[0, channels[0], 0, 4] = 1.0
            predicted[1, channels[0], 0, 7] = 100.0
        values = foot_acceleration_consistency_per_sample(
            predicted,
            target,
            torch.tensor([8, 4]),
            feature_std=torch.ones(263),
        )

        self.assertGreater(values[0].item(), 0.0)
        self.assertEqual(values[1].item(), 0.0)
        values.sum().backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_auto_lambda_hits_requested_gradient_ratio(self):
        value = calibrate_loss_lambda(2.0, 8.0, 0.1)
        self.assertAlmostEqual(value, 0.025)
        self.assertAlmostEqual(value * 8.0 / 2.0, 0.1)

    def test_default_sft_mixture_and_anti_jitter_are_conservative(self):
        config = CountSFTConfig()
        self.assertEqual(config.human_loss_weight, 0.5)
        self.assertEqual(config.step_loss_weight, 0.5)
        self.assertEqual(config.anti_jitter_auto_grad_ratio, 0.1)
        self.assertEqual(config.anti_jitter_lambda, 0.0)

    def test_policy_id_audits_exact_trainable_tensors(self):
        state = {
            "adapter.a": torch.tensor([1.0, 2.0]),
            "count.weight": torch.tensor([[3.0]]),
        }
        policy_id = trainable_policy_state_id(state)
        self.assertEqual(
            policy_checkpoint_id({"policy": state, "policy_id": policy_id}),
            policy_id,
        )
        mutated = {name: value.clone() for name, value in state.items()}
        mutated["adapter.a"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "does not match"):
            policy_checkpoint_id({"policy": mutated, "policy_id": policy_id})


if __name__ == "__main__":
    unittest.main()
