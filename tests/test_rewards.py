import torch
import unittest

from mdm_ddpo.rewards import (
    MotionReward,
    RewardOutput,
    add_step_reward,
    combine_reward_components,
)


class RewardCompositionTest(unittest.TestCase):
    def test_reward_components_are_weighted_per_sample(self):
        retrieval = torch.tensor([0.2, 0.6])
        m2m = torch.tensor([0.8, 0.4])
        combined = combine_reward_components(
            retrieval,
            m2m,
            retrieval_weight=2.0,
            m2m_weight=0.5,
        )
        torch.testing.assert_close(combined, torch.tensor([0.8, 1.4]))

    def test_mdm_features_are_renormalized_for_motionreward(self):
        reward = MotionReward.__new__(MotionReward)
        reward.mdm_mean = torch.tensor([[[1.0, 2.0]]])
        reward.mdm_std = torch.tensor([[[2.0, 4.0]]])
        reward.reward_mean = torch.tensor([[[0.0, 1.0]]])
        reward.reward_std = torch.tensor([[[1.0, 2.0]]])
        mdm_normalized = torch.tensor([[[0.5, -0.5]]])

        converted = reward._to_reward_normalization(mdm_normalized)
        torch.testing.assert_close(converted, torch.tensor([[[2.0, -0.5]]]))

    def test_step_reward_is_additive_and_mask_diagnostics_are_preserved(self):
        base = RewardOutput(
            total=torch.tensor([1.0, 2.0]),
            retrieval=torch.tensor([0.4, 0.5]),
            m2m=torch.tensor([0.6, 1.5]),
        )
        combined = add_step_reward(
            base,
            step=torch.tensor([0.0, 1.0]),
            step_mask=torch.tensor([False, True]),
            detected_steps=torch.tensor([-1, 3]),
            target_steps=torch.tensor([-1, 3]),
            absolute_error=torch.tensor([-1, 0]),
            step_weight=0.5,
        )

        torch.testing.assert_close(combined.total, torch.tensor([1.0, 2.5]))
        self.assertEqual(combined.step_mask.tolist(), [False, True])
        self.assertEqual(combined.detected_steps.tolist(), [-1, 3])

    def test_mean_embedding_scoring_preserves_rng_state(self):
        reward = MotionReward.__new__(MotionReward)
        reward.device = torch.device("cpu")
        reward.embedding_mode = "mean"
        torch.manual_seed(123)
        before = torch.get_rng_state()

        with reward._embedding_rng_context():
            torch.randn(8)

        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_sample_embedding_scoring_advances_rng_state(self):
        reward = MotionReward.__new__(MotionReward)
        reward.device = torch.device("cpu")
        reward.embedding_mode = "sample"
        torch.manual_seed(123)
        before = torch.get_rng_state()

        with reward._embedding_rng_context():
            torch.randn(8)

        self.assertFalse(torch.equal(torch.get_rng_state(), before))


if __name__ == "__main__":
    unittest.main()
