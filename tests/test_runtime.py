from __future__ import annotations

from enum import Enum, auto
from types import SimpleNamespace
import unittest

from mdm_ddpo.runtime import (
    configure_diffusion_prediction_type,
    diffusion_prediction_type,
    diffusion_runtime_metadata,
    resolve_prediction_type,
    validate_diffusion_runtime_metadata,
)


class MeanType(Enum):
    START_X = auto()
    EPSILON = auto()


class RuntimePredictionTypeTest(unittest.TestCase):
    def test_auto_reads_legacy_predict_epsilon_flag(self):
        self.assertEqual(
            resolve_prediction_type(
                SimpleNamespace(predict_epsilon=True),
            ),
            "epsilon",
        )
        self.assertEqual(
            resolve_prediction_type(
                SimpleNamespace(predict_epsilon=False),
            ),
            "x_start",
        )
        self.assertEqual(
            resolve_prediction_type(SimpleNamespace()),
            "x_start",
        )

    def test_explicit_type_overrides_checkpoint_metadata(self):
        args = SimpleNamespace(predict_epsilon=True)
        self.assertEqual(resolve_prediction_type(args, "x_start"), "x_start")

    def test_auto_accepts_named_prediction_type(self):
        self.assertEqual(
            resolve_prediction_type(SimpleNamespace(prediction_type="eps")),
            "epsilon",
        )
        self.assertEqual(
            resolve_prediction_type(SimpleNamespace(prediction_type="start-x")),
            "x_start",
        )

    def test_configure_overrides_a_hard_coded_external_default(self):
        diffusion = SimpleNamespace(model_mean_type=MeanType.START_X)
        configure_diffusion_prediction_type(diffusion, "epsilon")
        self.assertEqual(diffusion.model_mean_type, MeanType.EPSILON)
        self.assertEqual(diffusion_prediction_type(diffusion), "epsilon")

    def test_non_boolean_legacy_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            resolve_prediction_type(
                SimpleNamespace(predict_epsilon="true"),
            )

    def test_runtime_metadata_audits_schedule_and_variance(self):
        diffusion = SimpleNamespace(
            model_mean_type=MeanType.EPSILON,
            model_var_type=SimpleNamespace(name="FIXED_SMALL"),
            num_timesteps=25,
        )
        metadata = diffusion_runtime_metadata(
            SimpleNamespace(
                diffusion_steps=50,
                noise_schedule="linear",
                sigma_small=True,
                min_snr_gamma=5.0,
                lambda_xstart=1.0,
                lambda_xstart_vel=0.1,
            ),
            diffusion,
        )
        self.assertEqual(metadata["prediction_type"], "epsilon")
        self.assertEqual(metadata["training_diffusion_steps"], 50)
        self.assertEqual(metadata["sample_steps"], 25)
        self.assertEqual(metadata["noise_schedule"], "linear")
        self.assertEqual(metadata["model_var_type"], "FIXED_SMALL")
        self.assertEqual(metadata["min_snr_gamma"], 5.0)
        self.assertEqual(metadata["lambda_xstart"], 1.0)
        self.assertEqual(metadata["lambda_xstart_vel"], 0.1)

    def test_runtime_metadata_rejects_variance_mismatch(self):
        diffusion = SimpleNamespace(
            model_mean_type=MeanType.EPSILON,
            model_var_type=SimpleNamespace(name="FIXED_LARGE"),
            num_timesteps=50,
        )
        with self.assertRaisesRegex(RuntimeError, "variance type"):
            diffusion_runtime_metadata(
                SimpleNamespace(
                    diffusion_steps=50,
                    noise_schedule="linear",
                    sigma_small=True,
                ),
                diffusion,
            )

    def test_artifact_metadata_rejects_a_different_schedule(self):
        expected = {
            "prediction_type": "epsilon",
            "noise_schedule": "linear",
        }
        actual = {
            "prediction_type": "epsilon",
            "noise_schedule": "cosine",
        }
        with self.assertRaisesRegex(ValueError, "Checkpoint MDM diffusion"):
            validate_diffusion_runtime_metadata(
                expected,
                actual,
                source="Checkpoint",
            )


if __name__ == "__main__":
    unittest.main()
