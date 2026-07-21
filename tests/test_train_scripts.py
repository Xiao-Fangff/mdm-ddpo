from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TrainScriptTest(unittest.TestCase):
    def test_step_environment_values_survive_until_cli_is_built(self):
        project_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHON": "/bin/echo",
                "MDM_DDPO_ENABLE_STEP_REWARD": "1",
                "MDM_DDPO_STEP_REWARD_CALIBRATION_PATH": "/tmp/k16.json",
                "MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH": "/tmp/step-pool.pt",
                "MDM_DDPO_STEP_DATA_MANIFEST": "/tmp/manifest.jsonl",
                "MDM_DDPO_STEP_MOTION_ROOT": "/tmp/motions",
                "MDM_DDPO_STEP_DETECTOR_ROOT": "/tmp/detector",
            }
        )

        result = subprocess.run(
            [
                "bash",
                str(project_root / "scripts" / "train_humanml.sh"),
                "--epochs",
                "1",
            ],
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--enable-step-reward", result.stdout)
        self.assertIn(
            "--step-reward-calibration-path /tmp/k16.json",
            result.stdout,
        )
        self.assertIn("--fixed-step-eval-pool-path /tmp/step-pool.pt", result.stdout)
        self.assertIn("--step-data-manifest /tmp/manifest.jsonl", result.stdout)
        self.assertIn("--rollout-batch-size 64", result.stdout)
        self.assertIn("--step-use-m2m-reward", result.stdout)

    def test_k8_diagnostic_script_emits_isolated_count_configuration(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.NamedTemporaryFile() as calibration:
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHON": "/bin/echo",
                    "MDM_DDPO_STEP_REWARD_CALIBRATION_PATH": calibration.name,
                    "MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH": "/tmp/k8-pool.pt",
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    str(
                        project_root
                        / "scripts"
                        / "train_step_k8_diagnostic.sh"
                    ),
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        for expected in (
            "--step-data-ratio 0.5",
            "--step-samples-per-prompt 8",
            "--rollout-batches-per-epoch 4",
            "--step-balanced-sampling",
            "--step-advantage-retrieval-weight 0.2",
            "--step-advantage-m2m-weight 0.0",
            "--step-advantage-step-weight 0.8",
            "--step-reward-mode negative_l1",
            "--fixed-eval-every 2",
        ):
            self.assertIn(expected, result.stdout)

    def test_soft_counterfactual_script_removes_step_side_objectives(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.NamedTemporaryFile() as calibration:
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHON": "/bin/echo",
                    "MDM_DDPO_STEP_REWARD_CALIBRATION_PATH": calibration.name,
                    "MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH": "/tmp/soft-pool.pt",
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    str(
                        project_root
                        / "scripts"
                        / "train_step_soft_counterfactual.sh"
                    ),
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        for expected in (
            "--step-rollout-source synthetic",
            "--step-samples-per-prompt 8",
            "--step-advantage-retrieval-weight 0.0",
            "--step-advantage-m2m-weight 0.0",
            "--step-advantage-step-weight 1.0",
            "--no-step-use-m2m-reward",
            "--step-reward-mode soft_huber_exact",
        ):
            self.assertIn(expected, result.stdout)

    def test_epsilon_count_only_script_freezes_lora_and_uses_diagnostic_lr(self):
        project_root = Path(__file__).resolve().parents[1]
        with (
            tempfile.NamedTemporaryFile() as reward_calibration,
            tempfile.NamedTemporaryFile() as step_calibration,
        ):
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHON": "/bin/echo",
                    "MDM_DDPO_REWARD_CALIBRATION_PATH": reward_calibration.name,
                    "MDM_DDPO_STEP_REWARD_CALIBRATION_PATH": step_calibration.name,
                    "MDM_DDPO_FIXED_EVAL_POOL_PATH": "/tmp/human-pool.pt",
                    "MDM_DDPO_FIXED_STEP_EVAL_POOL_PATH": "/tmp/step-pool.pt",
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    str(
                        project_root
                        / "scripts"
                        / "train_epsilon_count_only_step_k8_short.sh"
                    ),
                ],
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        for expected in (
            "--prediction-type epsilon",
            "--enable-count-conditioning",
            "--no-train-lora",
            "--learning-rate 1e-4",
            "--clip-range 1e-3",
        ):
            self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
