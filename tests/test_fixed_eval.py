from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.trainer import (
    DDPOTrainer,
    FixedEvalPool,
    bootstrap_standard_error,
    load_fixed_eval_pool,
    save_fixed_eval_pool,
    summarize_fixed_eval_component,
    validate_fixed_eval_pool,
)


class FixedEvalPoolTest(unittest.TestCase):
    @staticmethod
    def _pool() -> FixedEvalPool:
        return validate_fixed_eval_pool(
            FixedEvalPool(
                dataset_indices=torch.tensor([7, 3]),
                motion=torch.arange(16, dtype=torch.float32).reshape(2, 2, 1, 4),
                lengths=torch.tensor([3, 4]),
                texts=["walk forward", "turn left"],
                split="val",
                noise_seed=2026,
                prompt_noise_seeds=torch.tensor([2026, 1022029]),
            )
        )

    def test_pool_round_trip_persists_exact_held_out_inputs(self):
        pool = self._pool()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed_eval_pool.pt"
            save_fixed_eval_pool(pool, path)
            restored = load_fixed_eval_pool(path)

        self.assertEqual(restored.pool_id, pool.pool_id)
        self.assertEqual(restored.split, "val")
        self.assertEqual(restored.texts, pool.texts)
        self.assertEqual(restored.noise_seed, pool.noise_seed)
        torch.testing.assert_close(restored.dataset_indices, pool.dataset_indices)
        torch.testing.assert_close(restored.lengths, pool.lengths)
        torch.testing.assert_close(restored.motion, pool.motion)
        torch.testing.assert_close(
            restored.prompt_noise_seeds,
            pool.prompt_noise_seeds,
        )

    def test_pool_checksum_detects_changed_gt_motion(self):
        pool = self._pool()
        changed_motion = pool.motion.clone()
        changed_motion[0, 0, 0, 0] += 1
        corrupted = FixedEvalPool(
            dataset_indices=pool.dataset_indices,
            motion=changed_motion,
            lengths=pool.lengths,
            texts=pool.texts,
            split=pool.split,
            noise_seed=pool.noise_seed,
            prompt_noise_seeds=pool.prompt_noise_seeds,
            pool_id=pool.pool_id,
        )

        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_fixed_eval_pool(corrupted)

    def test_resume_reuses_pool_next_to_checkpoint(self):
        pool = self._pool()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source-run"
            output_dir = root / "resumed-run"
            source_dir.mkdir()
            save_fixed_eval_pool(pool, source_dir / "fixed_eval_pool.pt")
            trainer = DDPOTrainer.__new__(DDPOTrainer)
            trainer.output_dir = output_dir
            output_dir.mkdir()
            trainer.config = TrainConfig(
                resume=str(source_dir / "latest.pt"),
                eval_split="val",
                fixed_eval_prompts=2,
                fixed_eval_seed=2026,
            )

            restored = trainer._load_or_create_fixed_eval_pool()

            copied = load_fixed_eval_pool(output_dir / "fixed_eval_pool.pt")
        self.assertEqual(restored.pool_id, pool.pool_id)
        self.assertEqual(copied.pool_id, pool.pool_id)

    def test_bootstrap_standard_error_is_deterministic(self):
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])

        first = bootstrap_standard_error(values, samples=1000, seed=42)
        second = bootstrap_standard_error(values, samples=1000, seed=42)

        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)

    def test_component_summary_reports_prompt_paired_deltas(self):
        baseline = torch.tensor([0.2, 0.4, 0.6, 0.8])
        current = torch.tensor([0.3, 0.3, 0.8, 0.8])

        metrics = summarize_fixed_eval_component(
            "eval_reward_retrieval",
            current,
            baseline,
            bootstrap_samples=500,
            seed=7,
        )

        self.assertAlmostEqual(metrics["eval_reward_retrieval_delta"], 0.05)
        self.assertAlmostEqual(
            metrics["eval_reward_retrieval_improvement_fraction"],
            0.5,
        )
        self.assertGreaterEqual(
            metrics["eval_reward_retrieval_delta_bootstrap_se"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
