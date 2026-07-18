from __future__ import annotations

import os
import subprocess
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


if __name__ == "__main__":
    unittest.main()
