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
                "elapsed_seconds": 8.0,
            },
            learning_rate=1.0e-4,
        )
        self.assertEqual(metrics["progress/epoch"], 3)
        self.assertEqual(metrics["progress/global_step"], 12)
        self.assertEqual(metrics["reward/total"], 0.4)
        self.assertEqual(metrics["reward/retrieval"], 0.3)
        self.assertEqual(metrics["reward/m2m"], 0.5)
        self.assertEqual(metrics["ppo/loss"], -0.1)
        self.assertEqual(metrics["time/epoch_seconds"], 8.0)
        self.assertEqual(metrics["optimization/learning_rate"], 1.0e-4)


if __name__ == "__main__":
    unittest.main()
