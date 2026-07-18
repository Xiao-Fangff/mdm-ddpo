from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.tracking import SwanLabTracker, format_training_metrics


class FakeRun:
    def __init__(self) -> None:
        self.logs = []

    def log(self, values, step=None):
        self.logs.append((values, step))


class FakeSwanLab:
    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs = None
        self.init_environment = None
        self.finish_calls = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.init_environment = dict(os.environ)
        return self.run

    def finish(self, **kwargs):
        self.finish_calls.append(kwargs)


class SwanLabTrackerTest(unittest.TestCase):
    def test_enabled_tracker_uses_current_api_and_filters_non_scalars(self):
        fake = FakeSwanLab()
        config = TrainConfig(
            use_swanlab=True,
            swanlab_project="motion-test",
            swanlab_run_name="run-one",
            swanlab_workspace="workspace-one",
            swanlab_mode="offline",
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with SwanLabTracker(config, output_dir, fake) as tracker:
                tracker.log(
                    {
                        "python_float": 1.5,
                        "numpy_scalar": np.float32(2.5),
                        "tensor_scalar": torch.tensor(3.5),
                        "tensor_vector": torch.tensor([1.0, 2.0]),
                        "text": "ignored",
                    },
                    step=7,
                )

            self.assertEqual(fake.init_kwargs["project"], "motion-test")
            self.assertEqual(fake.init_kwargs["name"], "run-one")
            self.assertEqual(fake.init_kwargs["workspace"], "workspace-one")
            self.assertEqual(fake.init_kwargs["mode"], "offline")
            self.assertEqual(
                fake.init_kwargs["log_dir"],
                str(output_dir / "swanlab"),
            )
            self.assertNotIn("experiment_name", fake.init_kwargs)
            self.assertNotIn("logdir", fake.init_kwargs)
            self.assertEqual(fake.run.logs[0][1], 7)
            self.assertEqual(
                fake.run.logs[0][0],
                {
                    "python_float": 1.5,
                    "numpy_scalar": np.float32(2.5),
                    "tensor_scalar": 3.5,
                },
            )
            self.assertEqual(fake.finish_calls, [{"state": "success", "error": None}])

    def test_disabled_tracker_has_no_side_effects(self):
        fake = FakeSwanLab()
        config = TrainConfig(use_swanlab=False)
        with SwanLabTracker(config, Path("/tmp/unused"), fake) as tracker:
            tracker.log({"metric": 1.0}, step=0)
        self.assertIsNone(fake.init_kwargs)
        self.assertFalse(fake.run.logs)
        self.assertFalse(fake.finish_calls)

    def test_exception_marks_run_as_crashed(self):
        fake = FakeSwanLab()
        config = TrainConfig(use_swanlab=True, swanlab_mode="offline")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with SwanLabTracker(config, Path("/tmp/swanlab-test"), fake):
                raise RuntimeError("training failed")
        self.assertEqual(
            fake.finish_calls,
            [{"state": "crashed", "error": "training failed"}],
        )

    def test_swanlab_aliases_are_hidden_during_init_and_restored(self):
        fake = FakeSwanLab()
        config = TrainConfig(
            use_swanlab=True,
            swanlab_project="configured-project",
            swanlab_run_name="configured-run",
            swanlab_mode="offline",
        )
        aliases = {
            "SWANLAB_PROJECT": "plain-string-that-is-not-json",
            "SWANLAB_RUN_NAME": "plain-run-name",
            "SWANLAB_MODE": "online",
        }
        with patch.dict(os.environ, aliases, clear=False):
            tracker = SwanLabTracker(config, Path("/tmp/swanlab-test"), fake)
            tracker.start()
            tracker.finish()
            for name in aliases:
                self.assertNotIn(name, fake.init_environment)
                self.assertEqual(os.environ[name], aliases[name])

    def test_training_metrics_are_grouped_for_curves(self):
        metrics = format_training_metrics(
            {
                "epoch": 3,
                "global_step": 12,
                "reward": 0.4,
                "reward_retrieval": 0.3,
                "reward_m2m": 0.5,
                "loss": -0.1,
                "log_ratio_mean": 0.001,
                "log_ratio_std": 0.002,
                "log_ratio_max": 0.004,
                "ratio_std": 0.003,
                "lora_norm": 2.0,
                "update_norm": 0.01,
                "anchor_loss": 0.8,
                "anchor_weighted_loss": 0.08,
                "anchor_grad_norm": 2.0,
                "ppo_grad_norm": 4.0,
                "anchor_grad_ratio": 0.1,
                "anchor_lambda": 0.04,
                "elapsed_seconds": 8.0,
                "reward_within_prompt_std": 0.2,
                "reward_between_prompt_std": 0.7,
                "reward_group_std_min": 0.01,
                "potential_group_whiten_scale_max": 100.0,
                "effective_shrink_scale_max": 12.0,
                "component_advantage_correlation": -0.2,
                "component_advantage_conflict_fraction": 0.4,
                "component_advantage_retrieval_contribution_mean_abs": 0.3,
                "component_advantage_m2m_contribution_mean_abs": 0.25,
                "eval_reward": 1.6,
                "eval_reward_delta": 0.03,
                "eval_reward_retrieval_delta_median": 0.01,
                "eval_reward_retrieval_improvement_fraction": 0.625,
                "eval_reward_retrieval_delta_bootstrap_se": 0.004,
                "eval_normalized_retrieval_delta": 0.2,
                "eval_normalized_m2m_delta": 0.1,
                "eval_balanced_score": 0.15,
                "eval_balanced_score_bootstrap_se": 0.02,
                "eval_feasible": 1.0,
                "eval_is_best_balanced": 1.0,
                "eval_batch_size": 32,
                "eval_samples_per_prompt": 4,
                "eval_diffusion_steps": 50,
                "eval_best_reward": 1.62,
                "eval_best_epoch": 2,
            },
            learning_rate=1.0e-4,
        )
        self.assertEqual(metrics["progress/epoch"], 3)
        self.assertEqual(metrics["progress/global_step"], 12)
        self.assertEqual(metrics["reward/total"], 0.4)
        self.assertEqual(metrics["reward/retrieval"], 0.3)
        self.assertEqual(metrics["reward/m2m"], 0.5)
        self.assertEqual(metrics["ppo/loss"], -0.1)
        self.assertEqual(metrics["ppo/log_ratio_mean"], 0.001)
        self.assertEqual(metrics["ppo/log_ratio_std"], 0.002)
        self.assertEqual(metrics["ppo/log_ratio_abs_max"], 0.004)
        self.assertEqual(metrics["ppo/ratio_std"], 0.003)
        self.assertEqual(metrics["optimization/lora_norm"], 2.0)
        self.assertEqual(metrics["optimization/update_norm"], 0.01)
        self.assertEqual(metrics["anchor/loss"], 0.8)
        self.assertEqual(metrics["anchor/weighted_loss"], 0.08)
        self.assertEqual(metrics["anchor/grad_norm"], 2.0)
        self.assertEqual(metrics["anchor/ppo_grad_norm"], 4.0)
        self.assertEqual(metrics["anchor/grad_ratio"], 0.1)
        self.assertEqual(metrics["anchor/lambda"], 0.04)
        self.assertEqual(metrics["time/epoch_seconds"], 8.0)
        self.assertEqual(metrics["optimization/learning_rate"], 1.0e-4)
        self.assertEqual(metrics["reward/within_prompt_std"], 0.2)
        self.assertEqual(metrics["reward/between_prompt_std"], 0.7)
        self.assertEqual(metrics["reward/group_std_min"], 0.01)
        self.assertEqual(
            metrics["reward/potential_group_whiten_scale_max"],
            100.0,
        )
        self.assertEqual(metrics["advantage/effective_shrink_scale_max"], 12.0)
        self.assertEqual(metrics["advantage/component_correlation"], -0.2)
        self.assertEqual(metrics["advantage/component_conflict_fraction"], 0.4)
        self.assertEqual(
            metrics["advantage/retrieval_contribution_mean_abs"],
            0.3,
        )
        self.assertEqual(
            metrics["advantage/m2m_contribution_mean_abs"],
            0.25,
        )
        self.assertEqual(metrics["eval/reward_total"], 1.6)
        self.assertEqual(metrics["eval/reward_total_delta"], 0.03)
        self.assertEqual(metrics["eval/reward_retrieval_delta_median"], 0.01)
        self.assertEqual(
            metrics["eval/reward_retrieval_improvement_fraction"],
            0.625,
        )
        self.assertEqual(
            metrics["eval/reward_retrieval_delta_bootstrap_se"],
            0.004,
        )
        self.assertEqual(metrics["eval/normalized_retrieval_delta"], 0.2)
        self.assertEqual(metrics["eval/normalized_m2m_delta"], 0.1)
        self.assertEqual(metrics["eval/balanced_score"], 0.15)
        self.assertEqual(metrics["eval/balanced_score_bootstrap_se"], 0.02)
        self.assertEqual(metrics["eval/feasible"], 1.0)
        self.assertEqual(metrics["eval/is_best_balanced"], 1.0)
        self.assertEqual(metrics["eval/batch_size"], 32)
        self.assertEqual(metrics["eval/samples_per_prompt"], 4)
        self.assertEqual(metrics["eval/diffusion_steps"], 50)
        self.assertEqual(metrics["eval/best_reward_total"], 1.62)
        self.assertEqual(metrics["eval/best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
