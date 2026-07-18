from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.runtime import CachedTextEmbedding
from mdm_ddpo.trainer import (
    DDPOTrainer,
    Trajectory,
    calibrate_anchor_lambda,
    gradient_l2_norm,
)


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, timesteps, **kwargs):
        del timesteps, kwargs
        return self.scale * x


class AnchorTest(unittest.TestCase):
    def test_auto_lambda_hits_requested_initial_gradient_ratio(self):
        value = calibrate_anchor_lambda(
            ppo_grad_norm=2.0,
            anchor_grad_norm=4.0,
            target_ratio=0.2,
        )

        self.assertAlmostEqual(value, 0.1)
        self.assertAlmostEqual(value * 4.0 / 2.0, 0.2)

    def test_gradient_norm_removes_grad_scaler_factor(self):
        norm = gradient_l2_norm(
            [torch.tensor([6.0, 8.0])],
            scale=2.0,
        )

        self.assertAlmostEqual(norm, 5.0)

    def test_anchor_is_invoked_once_per_optimizer_update(self):
        config = TrainConfig(
            device="cpu",
            precision="no",
            fixed_eval_every=0,
            rollout_batch_size=4,
            rollout_batches_per_epoch=1,
            samples_per_prompt=2,
            train_batch_size=2,
            gradient_accumulation_steps=2,
            inner_epochs=2,
            timestep_fraction=1.0,
            anchor_auto_grad_ratio=0.1,
        )
        model = TinyPolicy()
        trainer = DDPOTrainer.__new__(DDPOTrainer)
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.model = model
        trainer.policy_model = model
        trainer.diffusion = object()
        trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.global_step = 0
        trainer.anchor_enabled = True
        sample_count = 4
        timestep_count = 2
        cached = [
            CachedTextEmbedding(
                kind="tensor",
                values=torch.zeros(1),
            )
            for _ in range(sample_count)
        ]
        trajectory = Trajectory(
            latents=torch.zeros(sample_count, timestep_count, 1, 1, 2),
            next_latents=torch.zeros(sample_count, timestep_count, 1, 1, 2),
            timesteps=torch.tensor([[1, 2]]).repeat(sample_count, 1),
            old_log_probs=torch.zeros(sample_count, timestep_count),
            rewards=torch.zeros(sample_count),
            retrieval_rewards=torch.zeros(sample_count),
            m2m_rewards=torch.zeros(sample_count),
            texts=[f"prompt-{index}" for index in range(sample_count)],
            text_embeddings=cached,
            lengths=torch.full((sample_count,), 2),
            gt_motion=torch.zeros(sample_count, 2, 1),
            prompt_ids=torch.tensor([0, 0, 1, 1]),
            advantages=torch.tensor([1.0, -1.0, 0.5, -0.5]),
        )
        calls = []

        def fake_step(
            diffusion,
            policy_model,
            current,
            timesteps,
            **kwargs,
        ):
            del diffusion, timesteps
            previous = kwargs.get("prev_sample", current)
            log_prob = policy_model.scale.expand(current.shape[0])
            return previous, log_prob, current

        def fake_anchor(*args, **kwargs):
            del args, kwargs
            calls.append(1)
            return {
                "anchor_loss": 1.0,
                "anchor_weighted_loss": 0.1,
                "anchor_grad_norm": 2.0,
                "anchor_weighted_grad_norm": 0.2,
                "ppo_grad_norm": 2.0,
                "anchor_grad_ratio": 0.1,
                "anchor_lambda": 0.1,
                "anchor_batch_samples": 2.0,
                "anchor_calls": 1.0,
            }

        with patch(
            "mdm_ddpo.trainer.ddim_step_with_logprob",
            side_effect=fake_step,
        ), patch.object(
            trainer,
            "_audit_first_update_log_probs",
            return_value={"initial_ratio_mean": 1.0},
        ), patch.object(
            trainer,
            "_add_anchor_gradients",
            side_effect=fake_anchor,
        ):
            metrics = trainer.optimize(trajectory)

        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(metrics["anchor_calls"], 2.0)

    def test_anchor_is_off_by_default_and_modes_are_mutually_exclusive(self):
        self.assertEqual(TrainConfig().anchor_lambda, 0.0)
        self.assertEqual(TrainConfig().anchor_auto_grad_ratio, 0.0)
        config = TrainConfig(
            fixed_eval_every=0,
            anchor_lambda=0.1,
            anchor_auto_grad_ratio=0.2,
        )
        with self.assertRaisesRegex(ValueError, "either a fixed"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
