from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mdm_ddpo.step_data import (
    StepMotionDataset,
    create_fixed_step_eval_pool,
    load_fixed_step_eval_pool,
    load_step_manifest,
    parse_step_targets,
    render_step_prompt,
    save_fixed_step_eval_pool,
    stratified_step_split,
)


class StepDataTest(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        feature_dir = root / "features_263"
        feature_dir.mkdir()
        rows = []
        index = 0
        for target in (1, 2):
            for slot in range(4):
                name = f"target{target}-{slot}"
                values = np.full((40 + slot, 263), target + slot / 10, np.float32)
                path = feature_dir / f"{name}.npy"
                np.save(path, values)
                rows.append(
                    {
                        "sample_id": name,
                        "detected_steps": target,
                        "features_263_path": str(path),
                        "frame_count": len(values),
                        "prompt": "source prompt must not be reused",
                    }
                )
                index += 1
        manifest = root / "sample_manifest.jsonl"
        with open(manifest, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return manifest

    def test_target_parser_and_prompt_bank_are_deterministic(self):
        self.assertEqual(parse_step_targets("1,2,2,6"), (1, 2, 6))
        self.assertEqual(
            render_step_prompt(3, 7, 42),
            render_step_prompt(3, 7, 42),
        )
        self.assertIn("step", render_step_prompt(1, 0, 42))

    def test_manifest_split_is_stratified_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = load_step_manifest(
                self._manifest(root),
                motion_root=None,
                targets=(1, 2),
                min_frames=40,
                max_frames=196,
            )
            train_a, eval_a = stratified_step_split(
                records,
                eval_per_target=1,
                split_seed=7,
                prompt_seed=11,
            )
            train_b, eval_b = stratified_step_split(
                records,
                eval_per_target=1,
                split_seed=7,
                prompt_seed=11,
            )

        self.assertEqual([item.sample_id for item in eval_a], [item.sample_id for item in eval_b])
        self.assertEqual({item.target_steps for item in eval_a}, {1, 2})
        self.assertTrue(
            {item.sample_id for item in train_a}.isdisjoint(
                {item.sample_id for item in eval_a}
            )
        )
        self.assertTrue(all(item.prompt for item in train_a + eval_a))
        self.assertTrue(
            all(item.prompt != item.source_prompt for item in train_a + eval_a)
        )

    def test_step_motion_is_normalized_and_zero_padded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = load_step_manifest(
                self._manifest(root),
                motion_root=None,
                targets=(1,),
                min_frames=40,
                max_frames=196,
            )
            train, _ = stratified_step_split(
                records,
                eval_per_target=1,
                split_seed=3,
                prompt_seed=4,
            )
            dataset = StepMotionDataset(
                train,
                mean=np.ones(263, dtype=np.float32),
                std=np.full(263, 2, dtype=np.float32),
                max_frames=64,
            )
            item = dataset[0]

        self.assertEqual(item["motion"].shape, (263, 1, 64))
        self.assertTrue(torch.count_nonzero(item["motion"][:, :, item["length"]:]) == 0)

    def test_fixed_step_pool_round_trip_checksums_exact_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = load_step_manifest(
                self._manifest(root),
                motion_root=None,
                targets=(1, 2),
                min_frames=40,
                max_frames=196,
            )
            _, evaluation = stratified_step_split(
                records,
                eval_per_target=1,
                split_seed=5,
                prompt_seed=6,
            )
            pool = create_fixed_step_eval_pool(
                evaluation,
                mean=np.zeros(263, dtype=np.float32),
                std=np.ones(263, dtype=np.float32),
                max_frames=64,
                noise_seed=123,
                detector_backend="progressive",
            )
            path = root / "fixed_step_eval_pool.pt"
            save_fixed_step_eval_pool(pool, path)
            restored = load_fixed_step_eval_pool(path)

        self.assertEqual(restored.pool_id, pool.pool_id)
        self.assertEqual(restored.sample_ids, pool.sample_ids)
        self.assertEqual(restored.texts, pool.texts)
        torch.testing.assert_close(restored.motion, pool.motion)
        torch.testing.assert_close(restored.target_steps, pool.target_steps)


if __name__ == "__main__":
    unittest.main()
