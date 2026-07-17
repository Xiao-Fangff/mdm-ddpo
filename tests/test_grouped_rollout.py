from __future__ import annotations

import unittest

import torch

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.trainer import (
    Trajectory,
    compute_grouped_advantages,
    repeat_prompt_batch,
)


class GroupedRolloutTest(unittest.TestCase):
    @staticmethod
    def _trajectory(prompt_ids: list[int]) -> Trajectory:
        sample_count = len(prompt_ids)
        rewards = torch.arange(sample_count, dtype=torch.float32)
        return Trajectory(
            latents=torch.zeros(sample_count, 1, 1, 1, 1),
            next_latents=torch.zeros(sample_count, 1, 1, 1, 1),
            timesteps=torch.zeros(sample_count, 1, dtype=torch.long),
            old_log_probs=torch.zeros(sample_count, 1),
            rewards=rewards,
            retrieval_rewards=rewards,
            m2m_rewards=rewards,
            texts=[f"prompt-{index}" for index in range(sample_count)],
            text_embeddings=[None] * sample_count,  # type: ignore[list-item]
            lengths=torch.ones(sample_count, dtype=torch.long),
            gt_motion=torch.zeros(sample_count, 1, 263),
            prompt_ids=torch.tensor(prompt_ids),
        )

    def test_prompt_batch_is_repeated_contiguously(self):
        motion = torch.tensor([[[[1.0]]], [[[2.0]]]])
        lengths = torch.tensor([10, 20])
        texts = ["first", "second"]

        repeated_motion, repeated_lengths, repeated_texts, prompt_ids = (
            repeat_prompt_batch(motion, lengths, texts, samples_per_prompt=3)
        )

        self.assertEqual(repeated_motion[:, 0, 0, 0].tolist(), [1, 1, 1, 2, 2, 2])
        self.assertEqual(repeated_lengths.tolist(), [10, 10, 10, 20, 20, 20])
        self.assertEqual(
            repeated_texts,
            ["first", "first", "first", "second", "second", "second"],
        )
        self.assertEqual(prompt_ids.tolist(), [0, 0, 0, 1, 1, 1])

    def test_advantages_are_standardized_within_each_prompt(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 14.0, 18.0])
        prompt_ids = torch.tensor([0, 0, 0, 1, 1, 1])

        advantages, stats = compute_grouped_advantages(
            rewards,
            prompt_ids,
            epsilon=1.0e-8,
        )

        for prompt_id in (0, 1):
            group = advantages[prompt_ids == prompt_id]
            self.assertAlmostEqual(group.mean().item(), 0.0, places=6)
            self.assertAlmostEqual(
                group.std(unbiased=False).item(),
                1.0,
                places=6,
            )
        self.assertEqual(stats["unique_prompts"], 2.0)
        self.assertAlmostEqual(stats["zero_variance_prompt_fraction"], 0.0)

    def test_constant_prompt_reward_produces_zero_advantage(self):
        advantages, stats = compute_grouped_advantages(
            torch.tensor([2.0, 2.0, 1.0, 3.0]),
            torch.tensor([0, 0, 1, 1]),
            epsilon=1.0e-8,
        )
        torch.testing.assert_close(advantages[:2], torch.zeros(2))
        self.assertEqual(stats["zero_variance_prompt_fraction"], 0.5)

    def test_concatenated_rollouts_keep_prompt_groups_separate(self):
        first = self._trajectory([0, 0, 1, 1])
        second = self._trajectory([0, 0, 1, 1])

        combined = Trajectory.concatenate([first, second])

        self.assertEqual(
            combined.prompt_ids.tolist(),
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_config_requires_divisible_grouped_rollout_batch(self):
        config = TrainConfig(rollout_batch_size=10, samples_per_prompt=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            config.validate()

    def test_default_training_settings_use_grouped_low_variance_rewards(self):
        config = TrainConfig()

        self.assertEqual(config.samples_per_prompt, 4)
        self.assertEqual(config.reward_embedding_mode, "mean")
        self.assertAlmostEqual(config.learning_rate, 3.0e-4)


if __name__ == "__main__":
    unittest.main()
