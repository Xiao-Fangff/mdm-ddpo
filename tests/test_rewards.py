import torch
import unittest

from mdm_ddpo.rewards import MotionReward, combine_reward_components


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


if __name__ == "__main__":
    unittest.main()
