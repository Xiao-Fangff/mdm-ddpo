from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.trainer import (
    DDPOTrainer,
    Trajectory,
    apply_optimizer_hyperparameters,
    compute_grouped_advantages,
    log_prob_consistency_metrics,
    repeat_prompt_batch,
    restore_optimizer_state,
    shuffled_sample_minibatches,
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
            mode="group_whiten",
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

    def test_group_centered_mode_preserves_relative_reward_spread(self):
        rewards = torch.tensor([0.0, 2.0, 10.0, 18.0])
        prompt_ids = torch.tensor([0, 0, 1, 1])

        advantages, _ = compute_grouped_advantages(
            rewards,
            prompt_ids,
            epsilon=1.0e-8,
            mode="group_centered",
        )

        first = advantages[prompt_ids == 0]
        second = advantages[prompt_ids == 1]
        self.assertAlmostEqual(first.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(second.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(
            advantages.std(unbiased=False).item(),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            second.std(unbiased=False).item()
            / first.std(unbiased=False).item(),
            4.0,
            places=6,
        )

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

    def test_config_requires_equal_sized_ppo_minibatches(self):
        config = TrainConfig(
            rollout_batch_size=8,
            rollout_batches_per_epoch=3,
            train_batch_size=10,
            samples_per_prompt=4,
        )
        with self.assertRaisesRegex(ValueError, "rollout_batch_size.*divisible"):
            config.validate()

    def test_ppo_minibatches_shuffle_individual_samples(self):
        generator = torch.Generator().manual_seed(7)

        batches = shuffled_sample_minibatches(
            12,
            4,
            generator=generator,
        )

        flattened = torch.cat(batches)
        self.assertEqual(sorted(flattened.tolist()), list(range(12)))
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        contiguous_blocks = {
            tuple(range(start, start + 4)) for start in range(0, 12, 4)
        }
        self.assertTrue(
            any(tuple(batch.tolist()) not in contiguous_blocks for batch in batches)
        )

    def test_log_prob_consistency_audit_accepts_matching_policy(self):
        old = torch.tensor([0.1, -0.2, 0.3])
        new = old + torch.tensor([1.0e-6, -2.0e-6, 0.0])

        metrics = log_prob_consistency_metrics(old, new, tolerance=1.0e-5)

        self.assertLess(metrics["initial_log_prob_abs_diff_max"], 1.0e-5)
        self.assertAlmostEqual(metrics["initial_ratio_mean"], 1.0, places=5)

    def test_log_prob_consistency_audit_rejects_policy_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "consistency audit failed"):
            log_prob_consistency_metrics(
                torch.zeros(2),
                torch.tensor([0.0, 0.01]),
                tolerance=1.0e-4,
            )

    def test_default_training_settings_use_grouped_low_variance_rewards(self):
        config = TrainConfig()

        self.assertEqual(config.samples_per_prompt, 4)
        self.assertEqual(config.advantage_mode, "group_whiten")
        self.assertEqual(config.reward_embedding_mode, "mean")
        self.assertAlmostEqual(config.timestep_fraction, 0.5)
        self.assertAlmostEqual(config.learning_rate, 3.0e-4)
        self.assertEqual(config.early_stop_patience, 8)

    def test_resume_uses_current_optimizer_hyperparameters(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW(
            [parameter],
            lr=9.0e-4,
            betas=(0.8, 0.9),
            weight_decay=0.2,
            eps=1.0e-6,
        )
        config = TrainConfig(
            learning_rate=2.0e-4,
            adam_beta1=0.91,
            adam_beta2=0.98,
            adam_weight_decay=3.0e-5,
            adam_epsilon=2.0e-8,
        )

        apply_optimizer_hyperparameters(optimizer, config)

        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 2.0e-4)
        self.assertEqual(group["betas"], (0.91, 0.98))
        self.assertEqual(group["weight_decay"], 3.0e-5)
        self.assertEqual(group["eps"], 2.0e-8)

    def test_resume_can_reset_optimizer_state_for_algorithm_migration(self):
        source_parameter = torch.nn.Parameter(torch.tensor(1.0))
        source_optimizer = torch.optim.AdamW([source_parameter], lr=9.0e-4)
        source_parameter.grad = torch.tensor(2.0)
        source_optimizer.step()
        self.assertTrue(source_optimizer.state)

        target_parameter = torch.nn.Parameter(torch.tensor(1.0))
        target_optimizer = torch.optim.AdamW([target_parameter], lr=3.0e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        restored = restore_optimizer_state(
            target_optimizer,
            scaler,
            {
                "optimizer": source_optimizer.state_dict(),
                "scaler": {},
            },
            TrainConfig(reset_optimizer_on_resume=True),
        )

        self.assertFalse(restored)
        self.assertFalse(target_optimizer.state)
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 3.0e-4)

    def test_fixed_eval_tracks_best_reward_and_plateau_count(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig(early_stop_min_delta=0.0)
        trainer.fixed_eval_baseline = {
            "eval_reward": 1.0,
            "eval_reward_retrieval": 0.4,
            "eval_reward_m2m": 0.6,
        }
        trainer.best_eval_reward = 1.0
        trainer.best_eval_epoch = -1
        trainer.evals_without_improvement = 0
        trainer.global_step = 0
        evaluations = iter(
            [
                {
                    "eval_reward": 1.1,
                    "eval_reward_retrieval": 0.45,
                    "eval_reward_m2m": 0.65,
                },
                {
                    "eval_reward": 1.05,
                    "eval_reward_retrieval": 0.43,
                    "eval_reward_m2m": 0.62,
                },
            ]
        )
        trainer.evaluate_fixed_pool = lambda: next(evaluations)

        with tempfile.TemporaryDirectory() as directory:
            trainer.output_dir = Path(directory)
            improved = trainer._run_fixed_eval(epoch=4)
            plateau = trainer._run_fixed_eval(epoch=9)

        self.assertEqual(improved["eval_is_best"], 1.0)
        self.assertAlmostEqual(improved["eval_best_reward_delta"], 0.1)
        self.assertEqual(plateau["eval_is_best"], 0.0)
        self.assertEqual(plateau["eval_evals_without_improvement"], 1.0)
        self.assertEqual(trainer.best_eval_epoch, 4)

    def test_fixed_eval_signature_captures_chunking_and_sampler_settings(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig(
            fixed_eval_prompts=32,
            rollout_batch_size=32,
            samples_per_prompt=4,
            sample_steps=50,
            timestep_fraction=0.5,
        )
        trainer.diffusion = SimpleNamespace(num_timesteps=50)

        signature = trainer._fixed_eval_signature()

        self.assertEqual(signature["eval_prompts"], 32.0)
        self.assertEqual(signature["eval_samples"], 128.0)
        self.assertEqual(signature["eval_prompt_batch_size"], 8.0)
        self.assertEqual(signature["eval_batch_size"], 32.0)
        self.assertEqual(signature["eval_diffusion_steps"], 50.0)
        self.assertEqual(signature["eval_guidance_scale"], 2.5)
        self.assertEqual(signature["eval_precision_code"], 2.0)

    def test_resume_resets_baseline_when_fixed_eval_signature_changes(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig(
            resume="old-checkpoint.pt",
            fixed_eval_prompts=1,
            rollout_batch_size=2,
            samples_per_prompt=2,
        )
        trainer.diffusion = SimpleNamespace(num_timesteps=4)
        trainer.fixed_eval_batch = object()
        trainer.fixed_eval_baseline = {
            "eval_reward": 1.0,
            "eval_reward_retrieval": 0.4,
            "eval_reward_m2m": 0.6,
            "eval_samples": 2.0,
            "eval_prompts": 1.0,
            "eval_seed": float(trainer.config.fixed_eval_seed),
        }
        trainer.best_eval_reward = 1.1
        trainer.best_eval_epoch = 0
        trainer.evals_without_improvement = 5
        trainer.start_epoch = 1
        trainer.global_step = 2
        current_metrics = {
            "eval_reward": 1.05,
            "eval_reward_retrieval": 0.43,
            "eval_reward_m2m": 0.62,
            "eval_reward_std": 0.2,
            **trainer._fixed_eval_signature(),
        }
        trainer.evaluate_fixed_pool = lambda: dict(current_metrics)
        saved_best_epochs = []
        trainer._save_best_snapshot = saved_best_epochs.append

        with tempfile.TemporaryDirectory() as directory:
            trainer.output_dir = Path(directory)
            with self.assertLogs("mdm_ddpo.trainer", level="WARNING"):
                trainer._initialize_fixed_eval()

        self.assertEqual(trainer.fixed_eval_baseline, current_metrics)
        self.assertEqual(trainer.best_eval_reward, 1.05)
        self.assertEqual(trainer.best_eval_epoch, 0)
        self.assertEqual(trainer.evals_without_improvement, 0)
        self.assertEqual(saved_best_epochs, [0])


if __name__ == "__main__":
    unittest.main()
