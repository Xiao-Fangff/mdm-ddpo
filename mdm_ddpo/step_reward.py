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
class StepDetectionOutput:
    """Hard count plus event-level continuous detector diagnostics."""

    hard_count: torch.Tensor
    soft_count: torch.Tensor
    raw_candidate_count: torch.Tensor
    candidate_count: torch.Tensor
    candidate_spacing_mean: torch.Tensor
    candidate_spacing_min: torch.Tensor
    ankle_high_frequency_ratio: torch.Tensor


@dataclass(frozen=True)
class StepRewardOutput:
    reward: torch.Tensor
    detected_steps: torch.Tensor
    target_steps: torch.Tensor
    mask: torch.Tensor
    absolute_error: torch.Tensor
    soft_count: torch.Tensor | None = None
    soft_error: torch.Tensor | None = None

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


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _margin_gate(
    measured: Any,
    threshold: Any,
    *,
    temperature: float,
) -> float | None:
    measured_value = _finite_float(measured)
    threshold_value = _finite_float(threshold)
    if measured_value is None or threshold_value is None:
        return None
    denominator = max(abs(threshold_value), 1.0e-6) * temperature
    normalized_margin = (measured_value - threshold_value) / denominator
    # Avoid overflow while retaining an effectively hard tail.
    normalized_margin = max(-30.0, min(30.0, normalized_margin))
    return 1.0 / (1.0 + math.exp(-normalized_margin))


def _candidate_confidence(
    candidate: dict[str, Any],
    *,
    lead_temperature: float,
    length_temperature: float,
    progress_temperature: float,
) -> float:
    gates: list[float] = []
    for measured_name, threshold_name, temperature in (
        (
            "lead_margin_measured",
            "min_lead_margin",
            lead_temperature,
        ),
        (
            "swing_foot_forward_delta",
            "min_step_length",
            length_temperature,
        ),
    ):
        gate = _margin_gate(
            candidate.get(measured_name),
            candidate.get(threshold_name),
            temperature=temperature,
        )
        if gate is not None:
            gates.append(gate)

    if bool(candidate.get("root_progress_gate_applied", False)):
        gate = _margin_gate(
            candidate.get("root_forward_delta"),
            candidate.get("min_root_progress"),
            temperature=progress_temperature,
        )
        if gate is not None:
            gates.append(gate)

    if candidate.get("min_global_root_progress") is not None:
        gate = _margin_gate(
            candidate.get("global_root_forward_delta"),
            candidate.get("min_global_root_progress"),
            temperature=progress_temperature,
        )
        if gate is not None:
            gates.append(gate)

    if not gates:
        return 1.0 if candidate.get("reason_kept") else 0.0
    confidence = 1.0
    for gate in gates:
        confidence *= gate
    return float(max(0.0, min(1.0, confidence)))


def _candidate_payload(instance: Any) -> dict[str, Any]:
    if isinstance(instance, dict):
        return dict(instance)
    metadata = dict(getattr(instance, "meta", {}) or {})
    for name in ("start", "end", "key_frame"):
        if name not in metadata and hasattr(instance, name):
            metadata[name] = getattr(instance, name)
    metadata.setdefault("reason_kept", "progressive_step_verified")
    return metadata


def _candidate_frame(candidate: dict[str, Any]) -> int | None:
    for name in ("key_frame", "landing_frame", "start", "end"):
        value = _finite_float(candidate.get(name))
        if value is not None:
            return int(round(value))
    return None


def temporal_clustered_soft_count(
    track: Any,
    *,
    fps: int,
    cluster_gap_seconds: float,
    lead_temperature: float,
    length_temperature: float,
    progress_temperature: float,
) -> tuple[float, int, int, float, float]:
    """Convert progressive-step metadata into a jitter-resistant soft count.

    Kept and filtered candidates are clustered on the time axis.  Every
    cluster contributes only its maximum signed-margin confidence, preventing
    several near-identical ankle events from being rewarded multiple times.
    """

    instances_by_batch = getattr(track, "instances", [[]])
    kept = list(instances_by_batch[0]) if instances_by_batch else []
    metadata = dict(getattr(track, "meta", {}) or {})
    filtered_by_batch = metadata.get("filtered_candidates", [[]])
    filtered = list(filtered_by_batch[0]) if filtered_by_batch else []
    hard_count = len(kept)

    candidates = [_candidate_payload(value) for value in kept]
    candidates.extend(_candidate_payload(value) for value in filtered)
    by_identity: dict[tuple[Any, ...], tuple[int, float]] = {}
    for candidate in candidates:
        frame = _candidate_frame(candidate)
        if frame is None:
            continue
        confidence = _candidate_confidence(
            candidate,
            lead_temperature=lead_temperature,
            length_temperature=length_temperature,
            progress_temperature=progress_temperature,
        )
        identity = (
            candidate.get("foot"),
            candidate.get("start"),
            candidate.get("end"),
            frame,
        )
        previous = by_identity.get(identity)
        if previous is None or confidence > previous[1]:
            by_identity[identity] = (frame, confidence)

    events = sorted(by_identity.values(), key=lambda item: item[0])
    if not events:
        # Test doubles and legacy Motion-Rule tracks may expose only instances.
        return float(hard_count), hard_count, hard_count, 0.0, 0.0

    gap_frames = max(1, int(round(cluster_gap_seconds * fps)))
    clusters: list[list[tuple[int, float]]] = []
    for event in events:
        if not clusters or event[0] - clusters[-1][-1][0] > gap_frames:
            clusters.append([event])
        else:
            clusters[-1].append(event)
    representatives = [max(cluster, key=lambda item: item[1]) for cluster in clusters]
    soft_count = sum(confidence for _, confidence in representatives)
    frames = [frame for frame, _ in representatives]
    spacings = [
        (later - earlier) / float(fps)
        for earlier, later in zip(frames, frames[1:])
    ]
    spacing_mean = sum(spacings) / len(spacings) if spacings else 0.0
    spacing_min = min(spacings) if spacings else 0.0
    return (
        float(soft_count),
        len(events),
        len(clusters),
        float(spacing_mean),
        float(spacing_min),
    )


def _ankle_high_frequency_ratio(
    joints: torch.Tensor,
    *,
    fps: int,
    cutoff_hz: float,
) -> torch.Tensor:
    """Fraction of ankle-velocity spectrum above a gait-safe cutoff."""

    if joints.shape[0] < 4:
        return torch.zeros((), device=joints.device, dtype=torch.float32)
    velocity = joints[1:, (7, 8)].float() - joints[:-1, (7, 8)].float()
    spectrum = torch.fft.rfft(velocity, dim=0)
    energy = spectrum.abs().square().sum(dim=(1, 2))
    frequencies = torch.fft.rfftfreq(
        velocity.shape[0],
        d=1.0 / float(fps),
        device=velocity.device,
    )
    # Exclude DC, which represents sustained translation rather than jitter.
    active = frequencies > 0
    denominator = energy[active].sum()
    if denominator <= 0:
        return torch.zeros((), device=joints.device, dtype=torch.float32)
    return energy[frequencies >= cutoff_hz].sum().div(denominator).float()


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
        soft_lead_temperature: float = 1.0,
        soft_length_temperature: float = 1.0,
        soft_progress_temperature: float = 1.0,
        soft_cluster_gap_seconds: float = 0.15,
        ankle_high_frequency_cutoff_hz: float = 4.0,
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
        for name, value in (
            ("soft lead temperature", soft_lead_temperature),
            ("soft length temperature", soft_length_temperature),
            ("soft progress temperature", soft_progress_temperature),
            ("soft cluster gap", soft_cluster_gap_seconds),
            ("ankle high-frequency cutoff", ankle_high_frequency_cutoff_hz),
        ):
            if value <= 0 or not math.isfinite(value):
                raise ValueError(f"Step detector {name} must be positive.")
        if ankle_high_frequency_cutoff_hz >= fps / 2:
            raise ValueError(
                "Ankle high-frequency cutoff must be below the Nyquist rate."
            )
        self.backend = backend
        self.fps = int(fps)
        self.motion_rule_root = str(motion_rule_root)
        self.lead_threshold = float(lead_threshold)
        self.rgdno_threshold = float(rgdno_threshold)
        self.soft_lead_temperature = float(soft_lead_temperature)
        self.soft_length_temperature = float(soft_length_temperature)
        self.soft_progress_temperature = float(soft_progress_temperature)
        self.soft_cluster_gap_seconds = float(soft_cluster_gap_seconds)
        self.ankle_high_frequency_cutoff_hz = float(
            ankle_high_frequency_cutoff_hz
        )
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
    def detect_xyz(
        self,
        joints: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor | None = None,
    ) -> StepDetectionOutput:
        batch = _as_joint_batch(joints)
        resolved_lengths = _resolve_lengths(
            batch,
            lengths,
            minimum=(RGDNO_MIN_FRAMES if self.backend == "rgdno" else 1),
        )
        high_frequency = torch.stack(
            [
                _ankle_high_frequency_ratio(
                    sample[:length],
                    fps=self.fps,
                    cutoff_hz=self.ankle_high_frequency_cutoff_hz,
                )
                for sample, length in zip(batch, resolved_lengths)
            ]
        ).to(device=batch.device, dtype=torch.float32)
        if self.backend == "rgdno":
            hard = rgdno_hard_step_count(
                batch,
                resolved_lengths,
                threshold=self.rgdno_threshold,
            ).to(device=batch.device)
            values = hard.float()
            zeros = torch.zeros_like(values)
            return StepDetectionOutput(
                hard_count=hard,
                soft_count=values,
                raw_candidate_count=hard.clone(),
                candidate_count=hard.clone(),
                candidate_spacing_mean=zeros,
                candidate_spacing_min=zeros,
                ankle_high_frequency_ratio=high_frequency,
            )

        detector, batch_factory = self._load_progressive_backend()
        xyz = batch.detach().float().cpu().numpy()
        counts: list[int] = []
        soft_counts: list[float] = []
        raw_candidate_counts: list[int] = []
        candidate_counts: list[int] = []
        spacing_means: list[float] = []
        spacing_mins: list[float] = []
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
            hard_count = len(track.instances[0])
            soft_count, raw_count, cluster_count, spacing_mean, spacing_min = (
                temporal_clustered_soft_count(
                    track,
                    fps=self.fps,
                    cluster_gap_seconds=self.soft_cluster_gap_seconds,
                    lead_temperature=self.soft_lead_temperature,
                    length_temperature=self.soft_length_temperature,
                    progress_temperature=self.soft_progress_temperature,
                )
            )
            counts.append(hard_count)
            soft_counts.append(soft_count)
            raw_candidate_counts.append(raw_count)
            candidate_counts.append(cluster_count)
            spacing_means.append(spacing_mean)
            spacing_mins.append(spacing_min)
        return StepDetectionOutput(
            hard_count=torch.tensor(
                counts,
                device=batch.device,
                dtype=torch.long,
            ),
            soft_count=torch.tensor(
                soft_counts,
                device=batch.device,
                dtype=torch.float32,
            ),
            raw_candidate_count=torch.tensor(
                raw_candidate_counts,
                device=batch.device,
                dtype=torch.long,
            ),
            candidate_count=torch.tensor(
                candidate_counts,
                device=batch.device,
                dtype=torch.long,
            ),
            candidate_spacing_mean=torch.tensor(
                spacing_means,
                device=batch.device,
                dtype=torch.float32,
            ),
            candidate_spacing_min=torch.tensor(
                spacing_mins,
                device=batch.device,
                dtype=torch.float32,
            ),
            ankle_high_frequency_ratio=high_frequency,
        )

    @torch.no_grad()
    def count_xyz(
        self,
        joints: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.detect_xyz(joints, lengths).hard_count

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

    @torch.no_grad()
    def detect_normalized(
        self,
        normalized_motion: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> StepDetectionOutput:
        joints = recover_humanml263_joints(
            normalized_motion,
            mean=mean,
            std=std,
        )
        return self.detect_xyz(joints, lengths)


def compute_step_count_reward(
    detected_steps: torch.Tensor,
    target_steps: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    mode: str = "exp",
    temperature: float = 1.0,
    linear_tolerance: float = 3.0,
    soft_count: torch.Tensor | None = None,
    target_scale: torch.Tensor | float | None = None,
    huber_delta: float = 1.0,
    exact_bonus: float = 0.15,
) -> StepRewardOutput:
    """Turn hard/soft detector counts into a masked terminal reward."""
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
    elif mode == "soft_huber_exact":
        if soft_count is None or target_scale is None:
            raise ValueError(
                "soft_huber_exact requires soft_count and calibrated "
                "target_scale values."
            )
        if huber_delta <= 0 or not math.isfinite(huber_delta):
            raise ValueError("Soft step Huber delta must be positive.")
        if exact_bonus < 0 or not math.isfinite(exact_bonus):
            raise ValueError("Soft step exact bonus cannot be negative.")
        resolved_soft_count = torch.as_tensor(
            soft_count,
            device=detected.device,
            dtype=torch.float32,
        ).reshape(-1)
        if resolved_soft_count.shape != target.shape:
            raise ValueError("Soft count must match detected step tensors.")
        resolved_scale = torch.as_tensor(
            target_scale,
            device=detected.device,
            dtype=torch.float32,
        )
        if resolved_scale.ndim == 0:
            resolved_scale = resolved_scale.expand_as(resolved_soft_count)
        else:
            resolved_scale = resolved_scale.reshape(-1)
        if resolved_scale.shape != target.shape:
            raise ValueError("Target scales must be scalar or match targets.")
        if not torch.isfinite(resolved_scale).all() or (resolved_scale <= 0).any():
            raise ValueError("Target scales must be finite and positive.")
        normalized = (resolved_soft_count - target.float()) / resolved_scale
        absolute_normalized = normalized.abs()
        huber = torch.where(
            absolute_normalized <= huber_delta,
            0.5 * normalized.square(),
            huber_delta * (absolute_normalized - 0.5 * huber_delta),
        )
        reward = -huber + float(exact_bonus) * (absolute_error == 0).float()
    else:
        raise ValueError(
            "Step reward mode must be exp, linear, exact, negative_l1, or "
            "soft_huber_exact."
        )
    reward = torch.where(resolved_mask, reward, torch.zeros_like(reward))
    return StepRewardOutput(
        reward=reward,
        detected_steps=detected,
        target_steps=target,
        mask=resolved_mask,
        absolute_error=absolute_error,
        soft_count=(
            resolved_soft_count
            if mode == "soft_huber_exact"
            else (
                torch.as_tensor(
                    soft_count,
                    device=detected.device,
                    dtype=torch.float32,
                ).reshape(-1)
                if soft_count is not None
                else None
            )
        ),
        soft_error=(
            (
                resolved_soft_count - target.float()
                if mode == "soft_huber_exact"
                else torch.as_tensor(
                    soft_count,
                    device=detected.device,
                    dtype=torch.float32,
                ).reshape(-1) - target.float()
            )
            if soft_count is not None
            else None
        ),
    )


__all__ = [
    "HUMANML22_JOINT_NAMES",
    "HardStepDetector",
    "RGDNO_HARD_THRESHOLD",
    "RGDNO_MIN_FRAMES",
    "StepDetectionOutput",
    "StepRewardOutput",
    "compute_step_count_reward",
    "recover_humanml263_joints",
    "rgdno_hard_step_count",
    "temporal_clustered_soft_count",
]
