from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
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

    def test_component_shrink_supports_data_type_specific_weights(self):
        advantages, stats = compute_component_shrink_advantages(
            retrieval_rewards=torch.tensor([0.0, 2.0, 0.0, 2.0]),
            m2m_rewards=torch.tensor([2.0, 0.0, 0.0, 2.0]),
            prompt_ids=torch.tensor([0, 0, 1, 1]),
            epsilon=1.0e-8,
            retrieval_std_floor=1.0,
            m2m_std_floor=1.0,
            retrieval_weight=0.5,
            m2m_weight=0.5,
            step_retrieval_weight=0.2,
            step_m2m_weight=0.0,
            step_rewards=torch.tensor([0.0, 2.0, 0.0, 0.0]),
            step_mask=torch.tensor([True, True, False, False]),
            step_std_floor=1.0,
            step_weight=0.8,
        )

        self.assertLess(advantages[0].item(), 0.0)
        self.assertGreater(advantages[1].item(), 0.0)
        self.assertEqual(stats["component_advantage_step_retrieval_weight"], 0.2)
        self.assertEqual(stats["component_advantage_step_m2m_weight"], 0.0)
        self.assertEqual(
            stats["component_advantage_step_m2m_contribution_mean_abs"],
            0.0,
        )

    def test_step_information_metrics_expose_sparse_hard_ranking(self):
        _, stats = compute_component_shrink_advantages(
            retrieval_rewards=torch.zeros(8),
            m2m_rewards=torch.zeros(8),
            prompt_ids=torch.zeros(8, dtype=torch.long),
            epsilon=1.0e-8,
            retrieval_std_floor=1.0,
            m2m_std_floor=1.0,
            retrieval_weight=0.0,
            m2m_weight=0.0,
            step_rewards=torch.tensor([-2.0] * 7 + [-1.0]),
            step_mask=torch.ones(8, dtype=torch.bool),
            step_std_floor=1.0,
            step_weight=1.0,
        )

        self.assertEqual(
            stats["component_advantage_step_unique_reward_levels_mean"],
            2.0,
        )
        self.assertGreater(
            stats["component_advantage_step_pairwise_reward_tie_fraction"],
            0.7,
        )
        self.assertAlmostEqual(
            stats["component_advantage_step_top1_advantage_concentration"],
            1.0,
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
            "soft_count_mean": torch.tensor([3.8, 4.1]),
            "soft_error_mean": torch.tensor([1.8, 1.1]),
            "soft_mae": torch.tensor([1.8, 1.1]),
            "candidate_count_mean": torch.tensor([5.0, 5.0]),
            "candidate_spacing_mean": torch.tensor([0.5, 0.5]),
            "ankle_high_frequency_ratio": torch.tensor([0.1, 0.1]),
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
            soft_count_mean_per_prompt=torch.tensor([3.0, 3.4]),
            soft_error_mean_per_prompt=torch.tensor([1.0, 0.4]),
            soft_mae_per_prompt=torch.tensor([1.0, 0.4]),
            candidate_count_mean_per_prompt=torch.tensor([4.0, 4.0]),
            candidate_spacing_mean_per_prompt=torch.tensor([0.6, 0.6]),
            ankle_high_frequency_ratio_per_prompt=torch.tensor([0.1, 0.1]),
        )

        metrics = trainer._fixed_step_eval_with_deltas(evaluation)

        self.assertAlmostEqual(metrics["eval_step_reward_delta"], 0.2)
        self.assertAlmostEqual(metrics["eval_normalized_step_delta"], 0.4)
        self.assertAlmostEqual(metrics["eval_step_mae_delta"], -0.75)
        self.assertEqual(metrics["eval_step_mae_improvement_fraction"], 1.0)

        trainer.fixed_step_eval_pool = SimpleNamespace(
            target_steps=torch.tensor([2, 3])
        )
        trainer.fixed_step_eval_baseline_per_prompt = {
            name: values
            for name, values in trainer.fixed_step_eval_baseline_per_prompt.items()
            if name
            not in {
                "soft_count_mean",
                "soft_error_mean",
                "soft_mae",
                "candidate_count_mean",
                "candidate_spacing_mean",
                "ankle_high_frequency_ratio",
            }
        }

        legacy_metrics = trainer._fixed_step_eval_with_deltas(evaluation)

        self.assertAlmostEqual(
            legacy_metrics["eval_step_soft_mae_baseline"],
            1.5,
        )
        self.assertAlmostEqual(
            legacy_metrics["eval_step_soft_count_baseline"],
            4.0,
        )

    def test_step_acceptance_checkpoint_requires_all_hard_improvements(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig()
        trainer.fixed_step_eval_pool = object()
        trainer.best_balanced_score = 0.0
        trainer.best_balanced_epoch = -1
        trainer.best_retrieval_delta = 0.0
        trainer.best_retrieval_epoch = -1
        trainer.best_m2m_delta = 0.0
        trainer.best_m2m_epoch = -1
        trainer.best_step_reward_delta = 0.0
        trainer.best_step_epoch = -1
        trainer.best_step_acceptance_score = None
        trainer.best_step_acceptance_epoch = None
        trainer.evals_without_improvement = 0
        trainer.global_step = 7
        trainer.evaluate_fixed_pool = lambda: object()
        trainer.evaluate_fixed_step_pool = lambda: object()
        trainer._fixed_eval_with_deltas = lambda evaluation: {
            "eval_reward_retrieval_delta": 0.02,
            "eval_reward_retrieval_delta_bootstrap_se": 0.01,
            "eval_reward_m2m_delta": 0.01,
            "eval_reward_m2m_delta_bootstrap_se": 0.01,
            "eval_balanced_score": 0.015,
            "eval_balanced_score_bootstrap_se": 0.01,
        }
        step_metrics = {
            "eval_step_reward_delta": 0.1,
            "eval_step_mae_delta": -0.2,
            "eval_step_mae_delta_bootstrap_se": 0.05,
            "eval_step_exact_fraction_delta": 0.1,
            "eval_step_exact_fraction_delta_bootstrap_se": 0.025,
            "eval_step_within_one_fraction_delta": 0.05,
            "eval_step_within_one_fraction_delta_bootstrap_se": 0.025,
            "step_eval_samples": 32.0,
        }
        trainer._fixed_step_eval_with_deltas = (
            lambda evaluation: dict(step_metrics)
        )
        trainer._append_fixed_eval = lambda record: None
        trainer._append_fixed_eval_per_prompt = lambda **kwargs: None
        trainer._append_fixed_step_eval_per_prompt = lambda **kwargs: None

        accepted = trainer._run_fixed_eval(epoch=3)

        self.assertEqual(accepted["eval_step_acceptance"], 1.0)
        self.assertEqual(accepted["eval_is_best_step_acceptance"], 1.0)
        self.assertEqual(trainer.best_step_acceptance_epoch, 3)

        step_metrics["eval_step_exact_fraction_delta"] = 0.0
        rejected = trainer._run_fixed_eval(epoch=5)

        self.assertEqual(rejected["eval_step_acceptance"], 0.0)
        self.assertEqual(rejected["eval_is_best_step_acceptance"], 0.0)
        self.assertEqual(trainer.best_step_acceptance_epoch, 3)

    def test_step_acceptance_state_is_checkpointed_and_named_snapshot_allowed(self):
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = TrainConfig()
        trainer.global_step = 9
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer._rng_state = lambda: {}
        trainer.fixed_eval_baseline = None
        trainer.fixed_eval_baseline_per_prompt = None
        trainer.fixed_step_eval_baseline_per_prompt = None
        trainer.fixed_eval_pool = None
        trainer.fixed_step_eval_pool = None
        trainer.anchor_lambda_effective = 0.0
        trainer.anchor_lambda_calibrated = False
        trainer.reward_calibration = None
        trainer.step_reward_calibration = None
        trainer.best_balanced_score = 0.0
        trainer.best_balanced_epoch = -1
        trainer.best_retrieval_delta = 0.0
        trainer.best_retrieval_epoch = -1
        trainer.best_m2m_delta = 0.0
        trainer.best_m2m_epoch = -1
        trainer.best_step_reward_delta = 0.1
        trainer.best_step_epoch = 2
        trainer.best_step_acceptance_score = 4.5
        trainer.best_step_acceptance_epoch = 4
        trainer.evals_without_improvement = 0

        payload = trainer._checkpoint_payload(epoch=4)

        self.assertEqual(payload["best_step_acceptance_score"], 4.5)
        self.assertEqual(payload["best_step_acceptance_epoch"], 4)
        with tempfile.TemporaryDirectory() as directory:
            trainer.output_dir = Path(directory)
            path = trainer._save_named_snapshot(
                "best_step_acceptance.pt",
                epoch=4,
            )
            restored = torch.load(path, map_location="cpu", weights_only=False)
            resumed = DDPOTrainer.__new__(DDPOTrainer)
            resumed.config = TrainConfig()
            resumed.device = torch.device("cpu")
            resumed.model = torch.nn.Linear(1, 1)
            resumed.optimizer = torch.optim.AdamW(resumed.model.parameters())
            resumed.scaler = torch.amp.GradScaler("cpu", enabled=False)
            resumed.reward_calibration = None
            resumed.step_reward_calibration = None
            resumed._load_checkpoint(str(path))

        self.assertEqual(restored["best_step_acceptance_score"], 4.5)
        self.assertEqual(resumed.best_step_acceptance_score, 4.5)
        self.assertEqual(resumed.best_step_acceptance_epoch, 4)
        self.assertEqual(resumed.start_epoch, 5)
        with self.assertRaisesRegex(ValueError, "Unsupported named checkpoint"):
            trainer._save_named_snapshot("best_unknown.pt", epoch=4)

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

    def test_k8_half_step_mix_has_four_step_prompts_per_rollout(self):
        config = TrainConfig(
            enable_step_reward=True,
            rollout_batch_size=64,
            samples_per_prompt=4,
            step_samples_per_prompt=8,
            step_data_ratio=0.5,
        )

        self.assertEqual(config.humanml_prompts_per_rollout_batch, 8)
        self.assertEqual(config.step_prompts_per_rollout_batch, 4)
        self.assertEqual(config.humanml_rollout_samples, 32)
        self.assertEqual(config.step_rollout_samples, 32)

    def test_step_specific_advantage_weights_default_to_global_values(self):
        config = TrainConfig(
            advantage_retrieval_weight=0.5,
            advantage_m2m_weight=0.5,
            advantage_step_weight=0.25,
        )
        self.assertEqual(config.effective_step_advantage_retrieval_weight, 0.5)
        self.assertEqual(config.effective_step_advantage_m2m_weight, 0.5)
        self.assertEqual(config.effective_step_advantage_step_weight, 0.25)

    def test_soft_step_reward_requires_calibration_and_progressive_metadata(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            TrainConfig(
                enable_step_reward=True,
                step_reward_mode="soft_huber_exact",
            ).validate()

    def test_soft_step_reward_rejects_nonfinite_shaping_parameters(self):
        with self.assertRaisesRegex(ValueError, "soft-lead-temperature"):
            TrainConfig(
                enable_step_reward=True,
                fixed_eval_every=0,
                rollout_batch_size=64,
                step_soft_lead_temperature=float("nan"),
            ).validate()
        with self.assertRaisesRegex(ValueError, "exact-bonus"):
            TrainConfig(
                enable_step_reward=True,
                fixed_eval_every=0,
                rollout_batch_size=64,
                step_soft_exact_bonus=float("inf"),
            ).validate()

    def test_synthetic_step_rollout_rejects_step_m2m(self):
        with self.assertRaisesRegex(ValueError, "no-step-use-m2m"):
            TrainConfig(
                enable_step_reward=True,
                step_rollout_source="synthetic",
                step_use_m2m_reward=True,
                fixed_eval_every=0,
            ).validate()

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
