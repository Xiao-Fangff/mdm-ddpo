from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch


HUMANML22_JOINT_NAMES = (
    "pelvis",
    "l_hip",
    "r_hip",
    "spine1",
    "l_knee",
    "r_knee",
    "spine2",
    "l_ankle",
    "r_ankle",
    "spine3",
    "l_foot",
    "r_foot",
    "neck",
    "l_collar",
    "r_collar",
    "head",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
)

RGDNO_MIN_FRAMES = 12
RGDNO_HARD_THRESHOLD = 0.005


@dataclass(frozen=True)
class StepRewardOutput:
    reward: torch.Tensor
    detected_steps: torch.Tensor
    target_steps: torch.Tensor
    mask: torch.Tensor
    absolute_error: torch.Tensor

    @property
    def exact(self) -> torch.Tensor:
        return self.mask & (self.absolute_error == 0)

    @property
    def within_one(self) -> torch.Tensor:
        return self.mask & (self.absolute_error <= 1)


def _as_joint_batch(joints: torch.Tensor) -> torch.Tensor:
    joints = torch.as_tensor(joints)
    if joints.ndim == 3:
        joints = joints.unsqueeze(0)
    if joints.ndim != 4 or joints.shape[2:] != (22, 3):
        raise ValueError(
            "Expected joints with shape [T,22,3] or [B,T,22,3], "
            f"got {tuple(joints.shape)}."
        )
    return joints


def _resolve_lengths(
    joints: torch.Tensor,
    lengths: Sequence[int] | torch.Tensor | None,
    *,
    minimum: int = 1,
) -> list[int]:
    batch_size, max_frames = joints.shape[:2]
    if lengths is None:
        values = [max_frames] * batch_size
    elif isinstance(lengths, torch.Tensor):
        values = [
            int(value)
            for value in lengths.detach().cpu().reshape(-1).tolist()
        ]
    else:
        values = [int(value) for value in lengths]
    if len(values) != batch_size:
        raise ValueError(f"Expected {batch_size} lengths, got {len(values)}.")
    for length in values:
        if length < minimum:
            raise ValueError(
                f"Step detector requires at least {minimum} frames, got {length}."
            )
        if length > max_frames:
            raise ValueError(
                f"Motion length {length} exceeds padded length {max_frames}."
            )
    return values


def _five_frame_mean(signal: torch.Tensor) -> torch.Tensor:
    return sum(
        signal[offset : signal.shape[0] - 4 + offset]
        for offset in range(5)
    ) / 5.0


def _ankle_motion_energy(
    joints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    displacement = joints[1:] - joints[:-1]
    energy = displacement.square().sum(dim=-1)
    return _five_frame_mean(energy[:, 7]), _five_frame_mean(energy[:, 8])


def _transition_count(binary: torch.Tensor) -> torch.Tensor:
    earlier = binary[:-1]
    later = binary[1:]
    one_to_zero = ((earlier == 1) & (later == 0)).sum()
    zero_to_one = ((earlier == 0) & (later == 1)).sum()
    return torch.maximum(one_to_zero, zero_to_one)


def rgdno_hard_step_count(
    joints: torch.Tensor,
    lengths: Sequence[int] | torch.Tensor | None = None,
    *,
    threshold: float = RGDNO_HARD_THRESHOLD,
) -> torch.Tensor:
    """Count ankle-energy transitions using RFT_MLD's RG-DNO convention."""
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("RG-DNO hard threshold must be finite and positive.")
    batch = _as_joint_batch(joints)
    resolved_lengths = _resolve_lengths(
        batch,
        lengths,
        minimum=RGDNO_MIN_FRAMES,
    )
    counts: list[torch.Tensor] = []
    for sample, length in zip(batch, resolved_lengths):
        left_energy, right_energy = _ankle_motion_energy(sample[:length])
        left = _transition_count((left_energy > threshold).to(torch.int64))
        right = _transition_count((right_energy > threshold).to(torch.int64))
        counts.append(left + right)
    return torch.stack(counts)


def recover_humanml263_joints(
    normalized_motion: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Recover HumanML3D xyz joints from MDM-normalized 263-D features."""
    if normalized_motion.ndim != 3 or normalized_motion.shape[-1] != 263:
        raise ValueError("Motion must have shape [B,T,263].")
    mean = torch.as_tensor(
        mean,
        device=normalized_motion.device,
        dtype=torch.float32,
    ).reshape(1, 1, 263)
    std = torch.as_tensor(
        std,
        device=normalized_motion.device,
        dtype=torch.float32,
    ).reshape(1, 1, 263)
    raw_motion = normalized_motion.float() * std + mean
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    with torch.no_grad():
        joints = recover_from_ric(raw_motion, 22)
    if joints.shape != (*normalized_motion.shape[:2], 22, 3):
        raise RuntimeError(
            "recover_from_ric returned an unexpected shape: "
            f"{tuple(joints.shape)}."
        )
    return joints


class HardStepDetector:
    """Evaluation-only hard step detector for generated HumanML motions."""

    def __init__(
        self,
        *,
        backend: str = "progressive",
        fps: int = 20,
        motion_rule_root: str = "",
        lead_threshold: float = 0.138,
        rgdno_threshold: float = RGDNO_HARD_THRESHOLD,
        progressive_detector: Callable[..., Any] | None = None,
        motion_batch_factory: Callable[..., Any] | None = None,
    ) -> None:
        if backend not in {"progressive", "rgdno"}:
            raise ValueError(
                "Step detector backend must be 'progressive' or 'rgdno'."
            )
        if fps <= 0:
            raise ValueError("Step detector FPS must be positive.")
        if lead_threshold <= 0 or not math.isfinite(lead_threshold):
            raise ValueError("Progressive lead threshold must be positive.")
        if rgdno_threshold <= 0 or not math.isfinite(rgdno_threshold):
            raise ValueError("RG-DNO threshold must be positive.")
        self.backend = backend
        self.fps = int(fps)
        self.motion_rule_root = str(motion_rule_root)
        self.lead_threshold = float(lead_threshold)
        self.rgdno_threshold = float(rgdno_threshold)
        self._progressive_detector = progressive_detector
        self._motion_batch_factory = motion_batch_factory

    def _load_progressive_backend(
        self,
    ) -> tuple[Callable[..., Any], Callable[..., Any]]:
        if (
            self._progressive_detector is not None
            and self._motion_batch_factory is not None
        ):
            return self._progressive_detector, self._motion_batch_factory
        if self.motion_rule_root:
            root = Path(self.motion_rule_root).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(
                    f"Motion-Rule hard-detector root does not exist: {root}."
                )
            root_value = str(root)
            if root_value not in sys.path:
                sys.path.insert(0, root_value)
        try:
            from motion_reward.core.types import MotionBatch
            from motion_reward.templates.progressive_step import progressive_step
        except ImportError as exc:
            raise ImportError(
                "The progressive hard detector requires Motion-Rule-co. "
                "Set --step-detector-root to its repository root or use "
                "--step-detector-backend rgdno."
            ) from exc

        def factory(xyz: np.ndarray, *, fps: int) -> Any:
            return MotionBatch(
                xyz=xyz,
                joint_names=list(HUMANML22_JOINT_NAMES),
                fps=fps,
                meta={
                    "joint_profile": "humanml22_canonical_order",
                    "entity_fallback_policy": "strict",
                },
            )

        self._progressive_detector = progressive_step
        self._motion_batch_factory = factory
        return progressive_step, factory

    @torch.no_grad()
    def count_xyz(
        self,
        joints: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = _as_joint_batch(joints)
        if self.backend == "rgdno":
            return rgdno_hard_step_count(
                batch,
                lengths,
                threshold=self.rgdno_threshold,
            ).to(device=batch.device)

        resolved_lengths = _resolve_lengths(batch, lengths)
        detector, batch_factory = self._load_progressive_backend()
        xyz = batch.detach().float().cpu().numpy()
        counts: list[int] = []
        for sample_index, length in enumerate(resolved_lengths):
            motion_batch = batch_factory(
                xyz[sample_index : sample_index + 1, :length],
                fps=self.fps,
            )
            track = detector(
                motion_batch,
                direction="forward",
                frame="body",
                foot="any",
                step_candidate_source="lead_offsets",
                lead_threshold=self.lead_threshold,
            )
            counts.append(len(track.instances[0]))
        return torch.tensor(counts, device=batch.device, dtype=torch.long)

    @torch.no_grad()
    def count_normalized(
        self,
        normalized_motion: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        joints = recover_humanml263_joints(
            normalized_motion,
            mean=mean,
            std=std,
        )
        return self.count_xyz(joints, lengths)


def compute_step_count_reward(
    detected_steps: torch.Tensor,
    target_steps: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    mode: str = "exp",
    temperature: float = 1.0,
    linear_tolerance: float = 3.0,
) -> StepRewardOutput:
    """Turn integer hard-detector counts into a bounded terminal reward."""
    detected = torch.as_tensor(detected_steps).reshape(-1).long()
    target = torch.as_tensor(
        target_steps,
        device=detected.device,
    ).reshape(-1).long()
    if detected.shape != target.shape:
        raise ValueError("Detected and target step tensors must have one shape.")
    if mask is None:
        resolved_mask = target >= 0
    else:
        resolved_mask = torch.as_tensor(
            mask,
            device=detected.device,
        ).reshape(-1).bool()
        if resolved_mask.shape != target.shape:
            raise ValueError("Step reward mask must match target_steps.")
        resolved_mask = resolved_mask & (target >= 0)
    absolute_error = (detected - target).abs()
    error = absolute_error.float()
    if mode == "exp":
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("Step reward temperature must be positive.")
        reward = torch.exp(-error / float(temperature))
    elif mode == "linear":
        if linear_tolerance <= 0 or not math.isfinite(linear_tolerance):
            raise ValueError("Step reward linear tolerance must be positive.")
        reward = (1.0 - error / float(linear_tolerance)).clamp_min(0.0)
    elif mode == "exact":
        reward = (absolute_error == 0).float()
    elif mode == "negative_l1":
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("Step reward temperature must be positive.")
        reward = -error / float(temperature)
    else:
        raise ValueError(
            "Step reward mode must be exp, linear, exact, or negative_l1."
        )
    reward = torch.where(resolved_mask, reward, torch.zeros_like(reward))
    return StepRewardOutput(
        reward=reward,
        detected_steps=detected,
        target_steps=target,
        mask=resolved_mask,
        absolute_error=absolute_error,
    )


__all__ = [
    "HUMANML22_JOINT_NAMES",
    "HardStepDetector",
    "RGDNO_HARD_THRESHOLD",
    "RGDNO_MIN_FRAMES",
    "StepRewardOutput",
    "compute_step_count_reward",
    "recover_humanml263_joints",
    "rgdno_hard_step_count",
]
