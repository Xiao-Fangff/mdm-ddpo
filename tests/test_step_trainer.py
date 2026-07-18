from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.trainer import (
    DDPOTrainer,
    FixedStepEvalResult,
    Trajectory,
    compute_component_shrink_advantages,
)


class StepTrainerTest(unittest.TestCase):
    def test_component_shrink_applies_step_only_to_step_prompt_groups(self):
        advantages, stats = compute_component_shrink_advantages(
            retrieval_rewards=torch.zeros(4),
            m2m_rewards=torch.zeros(4),
            prompt_ids=torch.tensor([0, 0, 1, 1]),
            epsilon=1.0e-8,
            retrieval_std_floor=1.0,
            m2m_std_floor=1.0,
            step_rewards=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            step_mask=torch.tensor([True, True, False, False]),
            step_std_floor=1.0,
            step_weight=0.25,
        )

        self.assertGreater(advantages[1].item(), 0.0)
        self.assertLess(advantages[0].item(), 0.0)
        torch.testing.assert_close(advantages[2:], torch.zeros(2))
        self.assertEqual(stats["component_advantage_step_samples"], 2.0)
        self.assertEqual(
            stats["component_advantage_step_zero_variance_prompt_fraction"],
            0.0,
        )
        self.assertLessEqual(
            stats["component_advantage_step_effective_scale_max"],
            1.0,
        )

    def test_component_shrink_rejects_partial_step_prompt_mask(self):
        with self.assertRaisesRegex(ValueError, "constant within"):
            compute_component_shrink_advantages(
                retrieval_rewards=torch.zeros(2),
                m2m_rewards=torch.zeros(2),
                prompt_ids=torch.tensor([0, 0]),
                epsilon=1.0e-8,
                retrieval_std_floor=1.0,
                m2m_std_floor=1.0,
                step_rewards=torch.tensor([0.0, 1.0]),
                step_mask=torch.tensor([True, False]),
                step_std_floor=1.0,
                step_weight=0.25,
            )

    def test_component_shrink_masks_m2m_only_for_step_groups(self):
        advantages, stats = compute_component_shrink_advantages(
            retrieval_rewards=torch.zeros(4),
            m2m_rewards=torch.tensor([0.0, 2.0, 0.0, 2.0]),
            prompt_ids=torch.tensor([0, 0, 1, 1]),
            epsilon=1.0e-8,
            retrieval_std_floor=1.0,
            m2m_std_floor=1.0,
            retrieval_weight=0.0,
            m2m_weight=1.0,
            step_rewards=torch.zeros(4),
            step_mask=torch.tensor([True, True, False, False]),
            step_weight=0.0,
            step_use_m2m_reward=False,
        )

        torch.testing.assert_close(advantages[:2], torch.zeros(2))
        self.assertGreater(advantages[3].item(), 0.0)
        self.assertLess(advantages[2].item(), 0.0)
        self.assertEqual(stats["component_advantage_step_m2m_enabled"], 0.0)
        self.assertEqual(
            stats["component_advantage_step_m2m_contribution_mean_abs"],
            0.0,
        )

    def test_next_batch_mixes_humanml_and_step_targets(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig(
            enable_step_reward=True,
            advantage_mode="group_centered",
            fixed_eval_every=0,
            rollout_batch_size=4,
            samples_per_prompt=2,
            step_samples_per_prompt=2,
            step_data_ratio=0.5,
        )
        human = torch.zeros(1, 263, 1, 8)
        step = torch.ones(1, 263, 1, 6)
        trainer.data_loader = [
            (
                human,
                {"y": {"lengths": torch.tensor([8]), "text": ["human"]}},
            )
        ]
        trainer.step_data_loader = [
            (
                step,
                {
                    "y": {
                        "lengths": torch.tensor([6]),
                        "text": ["walk two steps"],
                        "target_steps": torch.tensor([2]),
                        "step_mask": torch.tensor([True]),
                    }
                },
            )
        ]
        trainer.data_iterator = None
        trainer.step_data_iterator = None

        motion, condition = trainer._next_batch()

        self.assertEqual(motion.shape, (2, 263, 1, 8))
        self.assertEqual(sorted(condition["y"]["target_steps"].tolist()), [-1, 2])
        self.assertEqual(condition["y"]["step_mask"].sum().item(), 1)
        self.assertEqual(set(condition["y"]["text"]), {"human", "walk two steps"})

    def test_anchor_falls_back_to_real_humanml_when_update_group_is_all_step(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig(anchor_batch_size=1)
        trajectory = Trajectory(
            latents=torch.zeros(4, 1, 1, 1, 1),
            next_latents=torch.zeros(4, 1, 1, 1, 1),
            timesteps=torch.zeros(4, 1, dtype=torch.long),
            old_log_probs=torch.zeros(4, 1),
            rewards=torch.zeros(4),
            retrieval_rewards=torch.zeros(4),
            m2m_rewards=torch.zeros(4),
            texts=["step", "step", "human", "human"],
            text_embeddings=[None] * 4,  # type: ignore[list-item]
            lengths=torch.ones(4, dtype=torch.long),
            gt_motion=torch.zeros(4, 1, 263),
            prompt_ids=torch.tensor([0, 0, 1, 1]),
            step_mask=torch.tensor([True, True, False, False]),
        )

        selected = trainer._anchor_sample_indices(
            trajectory,
            torch.tensor([0, 1]),
        )

        self.assertEqual(selected.tolist(), [2])

    def test_fixed_step_delta_uses_lower_is_better_for_mae(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig(
            fixed_eval_bootstrap_samples=100,
            fixed_eval_seed=9,
        )
        trainer.step_reward_calibration = SimpleNamespace(
            global_scale=lambda: 0.5
        )
        trainer.fixed_step_eval_baseline_per_prompt = {
            "total": torch.tensor([1.0, 1.0]),
            "retrieval": torch.tensor([0.5, 0.5]),
            "m2m": torch.tensor([0.5, 0.5]),
            "step_reward": torch.tensor([0.2, 0.4]),
            "exact_fraction": torch.tensor([0.0, 0.5]),
            "within_one_fraction": torch.tensor([0.5, 0.5]),
            "mae": torch.tensor([2.0, 1.0]),
            "detected_mean": torch.tensor([4.0, 4.0]),
        }
        evaluation = FixedStepEvalResult(
            metrics={},
            total_per_prompt=torch.tensor([1.1, 1.1]),
            retrieval_per_prompt=torch.tensor([0.55, 0.55]),
            m2m_per_prompt=torch.tensor([0.55, 0.55]),
            step_reward_per_prompt=torch.tensor([0.4, 0.6]),
            exact_per_prompt=torch.tensor([0.5, 0.5]),
            within_one_per_prompt=torch.tensor([1.0, 0.5]),
            mae_per_prompt=torch.tensor([1.0, 0.5]),
            detected_mean_per_prompt=torch.tensor([3.0, 3.5]),
        )

        metrics = trainer._fixed_step_eval_with_deltas(evaluation)

        self.assertAlmostEqual(metrics["eval_step_reward_delta"], 0.2)
        self.assertAlmostEqual(metrics["eval_normalized_step_delta"], 0.4)
        self.assertAlmostEqual(metrics["eval_step_mae_delta"], -0.75)
        self.assertEqual(metrics["eval_step_mae_improvement_fraction"], 1.0)

    def test_step_prompt_mix_properties_preserve_exact_motion_counts(self):
        config = TrainConfig(
            enable_step_reward=True,
            rollout_batch_size=64,
            samples_per_prompt=4,
            step_samples_per_prompt=16,
            step_data_ratio=0.25,
        )
        self.assertEqual(config.prompts_per_rollout_batch, 13)
        self.assertEqual(config.step_prompts_per_rollout_batch, 1)
        self.assertEqual(config.humanml_prompts_per_rollout_batch, 12)
        self.assertEqual(config.step_rollout_samples, 16)
        self.assertEqual(config.humanml_rollout_samples, 48)
        self.assertEqual(config.step_fixed_eval_prompts_per_batch, 4)

    def test_step_reward_rejects_total_group_shrink_calibration(self):
        config = TrainConfig(
            enable_step_reward=True,
            fixed_eval_every=0,
            advantage_mode="group_shrink",
            rollout_batch_size=64,
        )
        with self.assertRaisesRegex(ValueError, "total calibration"):
            config.validate()

    def test_step_component_shrink_requires_step_calibration(self):
        config = TrainConfig(
            enable_step_reward=True,
            fixed_eval_every=0,
            advantage_mode="component_shrink",
            reward_calibration_path="placeholder.json",
            rollout_batch_size=64,
        )
        with self.assertRaisesRegex(ValueError, "step-reward-calibration"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
