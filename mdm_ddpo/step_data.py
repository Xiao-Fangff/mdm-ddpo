from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


STEP_POOL_VERSION = 1

NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
}
STEP_PROMPT_TEMPLATES = (
    "walk forward {n} {steps}",
    "take {n} {steps} forward",
    "walk straight forward {n} {steps}",
    "walk straight ahead {n} {steps}",
    "take {n} {steps} straight ahead",
    "go forward {n} {steps}",
    "move forward {n} {steps}",
    "take {n} forward {steps}",
    "advance {n} {steps} forward",
    "take exactly {n} {steps} forward",
    "walk forward for exactly {n} {steps}",
    "go straight ahead for {n} {steps}",
    "move straight forward for {n} {steps}",
    "walk ahead {n} {steps}",
    "a person walks forward {n} {steps}.",
    "a person takes {n} {steps} forward.",
    "a person walks straight ahead {n} {steps}.",
    "a person steps forward {n} {times}.",
    "a person takes exactly {n} {steps} forward and stops.",
    "someone walks forward {n} {steps} and then stops.",
)
ZERO_STEP_PROMPTS = (
    "stand still and do not walk forward",
    "stand in place without taking any steps",
    "stay standing still and do not step forward",
    "remain standing in place",
    "a person stands still and does not walk forward.",
    "a person stands in place without taking any steps.",
)


@dataclass(frozen=True)
class StepSampleRecord:
    manifest_index: int
    sample_id: str
    target_steps: int
    feature_path: Path
    length: int
    prompt: str = ""
    source_prompt: str = ""


@dataclass(frozen=True)
class FixedStepEvalPool:
    manifest_indices: torch.Tensor
    sample_ids: list[str]
    motion: torch.Tensor
    lengths: torch.Tensor
    texts: list[str]
    target_steps: torch.Tensor
    split: str
    noise_seed: int
    prompt_noise_seeds: torch.Tensor
    detector_backend: str
    pool_id: str = ""

    @property
    def prompt_count(self) -> int:
        return len(self.texts)


def render_step_prompt(target: int, slot: int, seed: int) -> str:
    if target < 0 or target > 6:
        raise ValueError(f"Step targets must be in 0..6, got {target}.")
    if slot < 0:
        raise ValueError("Step prompt slot cannot be negative.")
    if target == 0:
        order = list(range(len(ZERO_STEP_PROMPTS)))
        random.Random(seed * 31).shuffle(order)
        return ZERO_STEP_PROMPTS[order[slot % len(order)]]
    order = list(range(len(STEP_PROMPT_TEMPLATES)))
    random.Random(seed * 31 + target).shuffle(order)
    template = STEP_PROMPT_TEMPLATES[order[slot % len(order)]]
    number = str(target) if slot % 2 == 0 else NUMBER_WORDS[target]
    return template.format(
        n=number,
        steps="step" if target == 1 else "steps",
        times="time" if target == 1 else "times",
    )


def parse_step_targets(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
        targets = tuple(int(part) for part in values)
    else:
        targets = tuple(int(item) for item in value)
    if not targets:
        raise ValueError("At least one step target is required.")
    if any(target < 0 or target > 6 for target in targets):
        raise ValueError("Step targets must all be in 0..6.")
    return tuple(sorted(set(targets)))


def _resolve_feature_path(
    raw_path: str,
    *,
    manifest_path: Path,
    motion_root: Path | None,
) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    if motion_root is not None:
        candidates.append(motion_root / raw)
    candidates.extend(
        [
            manifest_path.parent / raw,
            manifest_path.parent / "features_263" / raw.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot resolve step motion features path "
        f"{raw_path!r}; tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_step_manifest(
    manifest_path: str | Path,
    *,
    motion_root: str | Path | None,
    targets: Sequence[int],
    min_frames: int,
    max_frames: int,
) -> list[StepSampleRecord]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Step manifest does not exist: {path}.")
    if min_frames <= 0 or max_frames < min_frames:
        raise ValueError("Step frame limits are invalid.")
    target_set = set(parse_step_targets(targets))
    root = (
        Path(motion_root).expanduser().resolve()
        if motion_root
        else None
    )
    records: list[StepSampleRecord] = []
    sample_ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target = int(row["detected_steps"])
            if target not in target_set:
                continue
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate step sample id: {sample_id}.")
            feature_path = _resolve_feature_path(
                str(row["features_263_path"]),
                manifest_path=path,
                motion_root=root,
            )
            shape = np.load(feature_path, mmap_mode="r").shape
            if len(shape) != 2 or shape[1] != 263:
                raise ValueError(
                    f"Step features must be [T,263], got {shape} at {feature_path}."
                )
            reported_length = int(row.get("frame_count", shape[0]))
            length = min(int(shape[0]), reported_length, max_frames)
            if length < min_frames:
                continue
            sample_ids.add(sample_id)
            records.append(
                StepSampleRecord(
                    manifest_index=manifest_index,
                    sample_id=sample_id,
                    target_steps=target,
                    feature_path=feature_path,
                    length=length,
                    source_prompt=str(row.get("prompt", "")),
                )
            )
    if not records:
        raise ValueError("Step manifest contains no usable requested samples.")
    return records


def stratified_step_split(
    records: Sequence[StepSampleRecord],
    *,
    eval_per_target: int,
    split_seed: int,
    prompt_seed: int,
) -> tuple[list[StepSampleRecord], list[StepSampleRecord]]:
    if eval_per_target <= 0:
        raise ValueError("Step eval samples per target must be positive.")
    grouped: dict[int, list[StepSampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.target_steps, []).append(record)
    training: list[StepSampleRecord] = []
    evaluation: list[StepSampleRecord] = []
    for target in sorted(grouped):
        values = sorted(grouped[target], key=lambda item: item.sample_id)
        random.Random(split_seed + target * 1_000_003).shuffle(values)
        if len(values) <= eval_per_target:
            raise ValueError(
                f"Step target {target} has {len(values)} samples; need more than "
                f"eval_per_target={eval_per_target} to keep train/eval disjoint."
            )
        eval_values = values[:eval_per_target]
        train_values = values[eval_per_target:]
        evaluation.extend(
            replace(
                record,
                prompt=render_step_prompt(target, slot, prompt_seed),
            )
            for slot, record in enumerate(eval_values)
        )
        training.extend(
            replace(
                record,
                prompt=render_step_prompt(target, slot + eval_per_target, prompt_seed),
            )
            for slot, record in enumerate(train_values)
        )
    return training, evaluation


def load_humanml_stats(mdm_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(mdm_root).expanduser().resolve() / "dataset" / "HumanML3D"
    mean_path = root / "Mean.npy"
    std_path = root / "Std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Missing HumanML normalization stats below {root}."
        )
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    if mean.shape != (263,) or std.shape != (263,):
        raise ValueError("HumanML normalization stats must be 263-D.")
    return mean, std


class StepMotionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[StepSampleRecord],
        *,
        mean: np.ndarray,
        std: np.ndarray,
        max_frames: int,
    ) -> None:
        if not records:
            raise ValueError("Step motion dataset cannot be empty.")
        self.records = list(records)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 263)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 263)
        if np.any(self.std <= 0):
            raise ValueError("HumanML standard deviation must be positive.")
        self.max_frames = int(max_frames)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        raw = np.load(record.feature_path).astype(np.float32)[: record.length]
        if not np.isfinite(raw).all():
            raise FloatingPointError(
                f"Step motion contains NaN/Inf: {record.feature_path}."
            )
        length = min(len(raw), self.max_frames)
        normalized = (raw[:length] - self.mean) / self.std
        padded = np.zeros((self.max_frames, 263), dtype=np.float32)
        padded[:length] = normalized
        motion = torch.from_numpy(padded.T[:, None, :].copy())
        return {
            "motion": motion,
            "length": length,
            "text": record.prompt,
            "target_steps": record.target_steps,
            "sample_id": record.sample_id,
            "manifest_index": record.manifest_index,
        }


class BalancedStepTargetSampler(Sampler[int]):
    """Interleave target classes so every short prefix is nearly balanced."""

    def __init__(
        self,
        target_steps: Sequence[int],
        *,
        generator: torch.Generator,
    ) -> None:
        if not target_steps:
            raise ValueError("Balanced step sampler requires non-empty targets.")
        grouped: dict[int, list[int]] = {}
        for index, target in enumerate(target_steps):
            grouped.setdefault(int(target), []).append(index)
        if any(not indices for indices in grouped.values()):
            raise ValueError("Every balanced step target must have samples.")
        self.grouped_indices = {
            target: tuple(indices)
            for target, indices in sorted(grouped.items())
        }
        self.generator = generator
        self.sample_count = len(target_steps)

    def __len__(self) -> int:
        return self.sample_count

    def _shuffled_group(self, target: int) -> list[int]:
        values = self.grouped_indices[target]
        order = torch.randperm(len(values), generator=self.generator).tolist()
        return [values[position] for position in order]

    def __iter__(self):
        targets = list(self.grouped_indices)
        queues = {
            target: self._shuffled_group(target)
            for target in targets
        }
        positions = {target: 0 for target in targets}
        yielded = 0
        while yielded < self.sample_count:
            target_order = torch.randperm(
                len(targets),
                generator=self.generator,
            ).tolist()
            for target_position in target_order:
                target = targets[target_position]
                if yielded >= self.sample_count:
                    break
                if positions[target] >= len(queues[target]):
                    queues[target] = self._shuffled_group(target)
                    positions[target] = 0
                yield queues[target][positions[target]]
                positions[target] += 1
                yielded += 1


def collate_step_motions(items: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, Any]]:
    if not items:
        raise ValueError("Cannot collate an empty step batch.")
    motion = torch.stack([item["motion"] for item in items])
    lengths = torch.tensor([item["length"] for item in items], dtype=torch.long)
    target_steps = torch.tensor(
        [item["target_steps"] for item in items],
        dtype=torch.long,
    )
    return motion, {
        "y": {
            "lengths": lengths,
            "text": [str(item["text"]) for item in items],
            "target_steps": target_steps,
            "step_mask": torch.ones(len(items), dtype=torch.bool),
            "sample_id": [str(item["sample_id"]) for item in items],
            "manifest_index": torch.tensor(
                [item["manifest_index"] for item in items],
                dtype=torch.long,
            ),
        }
    }


def build_step_data_loader(
    dataset: StepMotionDataset,
    *,
    batch_size: int,
    seed: int,
    workers: int,
    pin_memory: bool,
    balanced_targets: bool = True,
) -> DataLoader:
    if len(dataset) < batch_size:
        raise ValueError(
            f"Step dataset has {len(dataset)} samples, fewer than batch size {batch_size}."
        )
    generator = torch.Generator().manual_seed(seed)
    sampler = (
        BalancedStepTargetSampler(
            [record.target_steps for record in dataset.records],
            generator=generator,
        )
        if balanced_targets
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not balanced_targets,
        sampler=sampler,
        num_workers=workers,
        drop_last=True,
        collate_fn=collate_step_motions,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def fixed_step_eval_pool_id(pool: FixedStepEvalPool) -> str:
    digest = hashlib.sha256()
    digest.update(str(STEP_POOL_VERSION).encode())
    digest.update(pool.split.encode())
    digest.update(pool.detector_backend.encode())
    digest.update(str(pool.noise_seed).encode())
    for tensor in (
        pool.manifest_indices,
        pool.motion,
        pool.lengths,
        pool.target_steps,
        pool.prompt_noise_seeds,
    ):
        _hash_tensor(digest, tensor)
    for values in (pool.sample_ids, pool.texts):
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def validate_fixed_step_eval_pool(
    pool: FixedStepEvalPool,
) -> FixedStepEvalPool:
    count = pool.prompt_count
    if count <= 0:
        raise ValueError("Fixed step-eval pool cannot be empty.")
    if len(pool.sample_ids) != count:
        raise ValueError("Fixed step-eval sample id count is invalid.")
    if pool.motion.ndim != 4 or pool.motion.shape[0] != count:
        raise ValueError("Fixed step-eval motion shape is invalid.")
    for tensor in (
        pool.manifest_indices,
        pool.lengths,
        pool.target_steps,
        pool.prompt_noise_seeds,
    ):
        if tensor.shape != (count,):
            raise ValueError("Fixed step-eval vector shape is invalid.")
    normalized = FixedStepEvalPool(
        manifest_indices=pool.manifest_indices.detach().cpu().long(),
        sample_ids=list(pool.sample_ids),
        motion=pool.motion.detach().cpu().float(),
        lengths=pool.lengths.detach().cpu().long(),
        texts=list(pool.texts),
        target_steps=pool.target_steps.detach().cpu().long(),
        split=str(pool.split),
        noise_seed=int(pool.noise_seed),
        prompt_noise_seeds=pool.prompt_noise_seeds.detach().cpu().long(),
        detector_backend=str(pool.detector_backend),
    )
    calculated = fixed_step_eval_pool_id(normalized)
    if pool.pool_id and pool.pool_id != calculated:
        raise ValueError("Fixed step-eval pool checksum mismatch.")
    return replace(normalized, pool_id=calculated)


def create_fixed_step_eval_pool(
    records: Sequence[StepSampleRecord],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    max_frames: int,
    noise_seed: int,
    detector_backend: str,
) -> FixedStepEvalPool:
    dataset = StepMotionDataset(
        records,
        mean=mean,
        std=std,
        max_frames=max_frames,
    )
    items = [dataset[index] for index in range(len(dataset))]
    motion, condition = collate_step_motions(items)
    count = len(items)
    return validate_fixed_step_eval_pool(
        FixedStepEvalPool(
            manifest_indices=condition["y"]["manifest_index"],
            sample_ids=list(condition["y"]["sample_id"]),
            motion=motion,
            lengths=condition["y"]["lengths"],
            texts=list(condition["y"]["text"]),
            target_steps=condition["y"]["target_steps"],
            split="val",
            noise_seed=noise_seed,
            prompt_noise_seeds=(
                torch.arange(count, dtype=torch.long) * 1_000_003
                + noise_seed
            ),
            detector_backend=detector_backend,
        )
    )


def save_fixed_step_eval_pool(
    pool: FixedStepEvalPool,
    path: str | Path,
) -> Path:
    pool = validate_fixed_step_eval_pool(pool)
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    torch.save(
        {
            "version": STEP_POOL_VERSION,
            "pool_id": pool.pool_id,
            "manifest_indices": pool.manifest_indices,
            "sample_ids": pool.sample_ids,
            "motion": pool.motion,
            "lengths": pool.lengths,
            "texts": pool.texts,
            "target_steps": pool.target_steps,
            "split": pool.split,
            "noise_seed": pool.noise_seed,
            "prompt_noise_seeds": pool.prompt_noise_seeds,
            "detector_backend": pool.detector_backend,
        },
        temporary,
    )
    os.replace(temporary, resolved)
    return resolved


def load_fixed_step_eval_pool(path: str | Path) -> FixedStepEvalPool:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != STEP_POOL_VERSION:
        raise ValueError("Unsupported fixed step-eval pool version.")
    required = {
        "pool_id",
        "manifest_indices",
        "sample_ids",
        "motion",
        "lengths",
        "texts",
        "target_steps",
        "split",
        "noise_seed",
        "prompt_noise_seeds",
        "detector_backend",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"Fixed step-eval pool is missing fields: {missing}")
    return validate_fixed_step_eval_pool(
        FixedStepEvalPool(
            manifest_indices=payload["manifest_indices"],
            sample_ids=list(payload["sample_ids"]),
            motion=payload["motion"],
            lengths=payload["lengths"],
            texts=list(payload["texts"]),
            target_steps=payload["target_steps"],
            split=str(payload["split"]),
            noise_seed=int(payload["noise_seed"]),
            prompt_noise_seeds=payload["prompt_noise_seeds"],
            detector_backend=str(payload["detector_backend"]),
            pool_id=str(payload["pool_id"]),
        )
    )


def target_histogram(records: Iterable[StepSampleRecord]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for record in records:
        key = str(record.target_steps)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


__all__ = [
    "BalancedStepTargetSampler",
    "FixedStepEvalPool",
    "StepMotionDataset",
    "StepSampleRecord",
    "build_step_data_loader",
    "collate_step_motions",
    "create_fixed_step_eval_pool",
    "fixed_step_eval_pool_id",
    "load_fixed_step_eval_pool",
    "load_humanml_stats",
    "load_step_manifest",
    "parse_step_targets",
    "render_step_prompt",
    "save_fixed_step_eval_pool",
    "stratified_step_split",
    "target_histogram",
    "validate_fixed_step_eval_pool",
]
