from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .config import (
    DEFAULT_DATA_CACHE_DIR,
    DEFAULT_MDM_ROOT,
    DEFAULT_MOTIONRFT_ROOT,
    DEFAULT_STEP_DATA_MANIFEST,
    DEFAULT_STEP_MOTION_ROOT,
    TrainConfig,
)
from .count_conditioning import (
    count_conditioning_metadata,
    count_conditioning_signature,
    set_count_conditioning_trainable,
    validate_count_conditioning_signature,
)
from .lora import (
    configure_trainable_policy,
    load_trainable_state_dict,
    trainable_state_dict,
)
from .policy_io import trainable_policy_state_id
from .runtime import (
    autocast_context,
    bootstrap_external_repositories,
    build_data_loader,
    build_mdm,
    build_model_kwargs,
    diffusion_prediction_type,
    diffusion_runtime_metadata,
    load_model_args,
    resolve_device,
    seed_everything,
)
from .step_data import (
    StepMotionDataset,
    build_step_data_loader,
    create_balanced_step_sft_records,
    load_humanml_stats,
    load_step_manifest,
    parse_step_targets,
    stratified_step_split,
)


LOGGER = logging.getLogger(__name__)
COUNT_SFT_FORMAT = "count_conditioning_sft_v1"
FOOT_JOINTS = (7, 8, 10, 11)


@dataclass
class CountSFTConfig:
    mdm_root: str = DEFAULT_MDM_ROOT
    motionrft_root: str = DEFAULT_MOTIONRFT_ROOT
    model_path: str = TrainConfig.model_path
    model_args_path: str = TrainConfig.model_args_path
    prediction_type: str = "auto"
    data_cache_dir: str = DEFAULT_DATA_CACHE_DIR
    step_data_manifest: str = DEFAULT_STEP_DATA_MANIFEST
    step_motion_root: str = DEFAULT_STEP_MOTION_ROOT
    output_dir: str = "outputs/count_conditioning_sft"
    resume: str = ""

    seed: int = 42
    device: str = "cuda:0"
    precision: str = "bf16"
    allow_tf32: bool = True
    data_workers: int = 0
    pin_memory: bool = True

    epochs: int = 3
    # 0 means one pass through the balanced step subset per epoch.
    steps_per_epoch: int = 0
    human_batch_size: int = 32
    step_batch_size: int = 32
    human_loss_weight: float = 0.5
    step_loss_weight: float = 0.5
    lora_learning_rate: float = 3.0e-5
    count_learning_rate: float = 1.0e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1.0e-4
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0

    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_target_regex: str = TrainConfig.lora_target_regex

    step_targets: str = "1,2,3,4,5,6"
    step_split_seed: int = 20260600
    step_prompt_seed: int = 20260612
    step_eval_samples_per_target: int = 8
    step_min_frames: int = 40
    step_max_frames: int = 196
    length_bins: int = 8
    max_samples_per_bin_target: int = 0

    anti_jitter_lambda: float = 0.0
    anti_jitter_auto_grad_ratio: float = 0.1
    save_every: int = 1
    log_every: int = 10

    @property
    def target_values(self) -> tuple[int, ...]:
        return parse_step_targets(self.step_targets)

    def validate(self) -> None:
        if self.prediction_type not in {"auto", "x_start", "epsilon"}:
            raise ValueError("Invalid SFT prediction type.")
        if self.precision not in {"no", "fp16", "bf16"}:
            raise ValueError("SFT precision must be no, fp16, or bf16.")
        for name in (
            "epochs",
            "human_batch_size",
            "step_batch_size",
            "lora_rank",
            "step_eval_samples_per_target",
            "step_min_frames",
            "step_max_frames",
            "length_bins",
            "save_every",
            "log_every",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"SFT {name} must be positive.")
        if self.steps_per_epoch < 0:
            raise ValueError("SFT steps_per_epoch cannot be negative.")
        if self.step_max_frames < self.step_min_frames:
            raise ValueError("SFT step frame limits are invalid.")
        if self.max_samples_per_bin_target < 0:
            raise ValueError("SFT per-cell sample cap cannot be negative.")
        weights = (self.human_loss_weight, self.step_loss_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("SFT mixture loss weights must be finite and non-negative.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1.0e-8):
            raise ValueError("SFT HumanML and step loss weights must sum to 1.")
        for name in (
            "lora_learning_rate",
            "count_learning_rate",
            "adam_epsilon",
            "max_grad_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"SFT {name} must be finite and positive.")
        if self.anti_jitter_lambda < 0 or self.anti_jitter_auto_grad_ratio < 0:
            raise ValueError("SFT anti-jitter settings cannot be negative.")
        if self.anti_jitter_lambda > 0 and self.anti_jitter_auto_grad_ratio > 0:
            raise ValueError(
                "Use fixed anti-jitter lambda or automatic gradient ratio, not both."
            )
        required = {
            "MDM root": self.mdm_root,
            "MotionRFT root": self.motionrft_root,
            "MDM checkpoint": self.model_path,
            "MDM args": self.model_args_path,
            "step manifest": self.step_data_manifest,
            "step motion root": self.step_motion_root,
        }
        if self.resume:
            required["resume checkpoint"] = self.resume
        missing = [
            f"{label}: {Path(path).expanduser()}"
            for label, path in required.items()
            if not Path(path).expanduser().exists()
        ]
        if missing:
            raise FileNotFoundError("Missing SFT paths:\n  " + "\n  ".join(missing))


class NativeMDMTrainingModel(torch.nn.Module):
    """Adapter required by the external MDM training_losses implementation."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(*args, **kwargs)


def humanml_foot_position_channels() -> tuple[int, ...]:
    channels: list[int] = []
    for joint in FOOT_JOINTS:
        if joint <= 0:
            raise ValueError("HumanML local-position channels exclude the root joint.")
        start = 4 + (joint - 1) * 3
        channels.extend(range(start, start + 3))
    return tuple(channels)


def foot_acceleration_consistency_per_sample(
    pred_xstart: torch.Tensor,
    target_xstart: torch.Tensor,
    lengths: torch.Tensor,
    *,
    feature_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Physical-scale second-difference consistency for ankles and feet."""

    if pred_xstart.shape != target_xstart.shape or pred_xstart.ndim != 4:
        raise ValueError("Anti-jitter motions must be matching [B,263,1,T] tensors.")
    if pred_xstart.shape[1:3] != (263, 1):
        raise ValueError("Anti-jitter currently supports HumanML 263-D motions only.")
    if pred_xstart.shape[-1] < 3:
        raise ValueError("Anti-jitter needs at least three motion frames.")
    lengths = torch.as_tensor(lengths, device=pred_xstart.device).long().reshape(-1)
    if lengths.shape[0] != pred_xstart.shape[0]:
        raise ValueError("Anti-jitter lengths do not match the motion batch.")
    channels = torch.tensor(
        humanml_foot_position_channels(),
        device=pred_xstart.device,
        dtype=torch.long,
    )
    predicted = pred_xstart.index_select(1, channels).squeeze(2).float()
    target = target_xstart.index_select(1, channels).squeeze(2).float()
    if feature_std is not None:
        scale = torch.as_tensor(
            feature_std,
            device=pred_xstart.device,
            dtype=torch.float32,
        ).reshape(-1)
        if scale.shape != (263,):
            raise ValueError("HumanML feature std must contain 263 values.")
        selected_scale = scale.index_select(0, channels).reshape(1, -1, 1)
        predicted = predicted * selected_scale
        target = target * selected_scale
    predicted_acceleration = (
        predicted[..., 2:] - 2.0 * predicted[..., 1:-1] + predicted[..., :-2]
    )
    target_acceleration = (
        target[..., 2:] - 2.0 * target[..., 1:-1] + target[..., :-2]
    )
    frame_indices = torch.arange(
        2,
        pred_xstart.shape[-1],
        device=pred_xstart.device,
    )
    valid = frame_indices.unsqueeze(0) < lengths.unsqueeze(1)
    squared = (predicted_acceleration - target_acceleration).square()
    numerator = (squared * valid.unsqueeze(1)).sum(dim=(1, 2))
    denominator = valid.sum(dim=1).clamp_min(1).to(squared) * squared.shape[1]
    return numerator / denominator


def gradient_l2_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    if not values:
        return 0.0
    return torch.stack(values).sum().sqrt().item()


def calibrate_loss_lambda(
    primary_grad_norm: float,
    auxiliary_grad_norm: float,
    target_ratio: float,
    *,
    epsilon: float = 1.0e-12,
) -> float:
    if target_ratio < 0:
        raise ValueError("Auxiliary target gradient ratio cannot be negative.")
    if (
        not math.isfinite(primary_grad_norm)
        or not math.isfinite(auxiliary_grad_norm)
        or primary_grad_norm <= epsilon
        or auxiliary_grad_norm <= epsilon
    ):
        raise ValueError("Cannot calibrate auxiliary loss from zero/non-finite gradients.")
    return target_ratio * primary_grad_norm / auxiliary_grad_norm


def _weighted_group_mean(
    values: torch.Tensor,
    human_count: int,
    human_weight: float,
    step_weight: float,
) -> torch.Tensor:
    if values.ndim != 1 or not 0 < human_count < values.shape[0]:
        raise ValueError("SFT per-sample loss grouping is invalid.")
    return (
        human_weight * values[:human_count].mean()
        + step_weight * values[human_count:].mean()
    )


def native_losses_with_pred_xstart(
    *,
    model: torch.nn.Module,
    diffusion: Any,
    training_model: NativeMDMTrainingModel,
    motion: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    model_kwargs: dict[str, Any],
    dataset: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run the unmodified native loss once and retain its differentiable x0."""

    captured: list[torch.Tensor] = []

    def capture_output(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        del module, args
        captured.append(output)

    handle = model.register_forward_hook(capture_output)
    x_t = diffusion.q_sample(motion, timesteps, noise=noise)
    try:
        terms = diffusion.training_losses(
            training_model,
            motion,
            timesteps,
            model_kwargs=model_kwargs,
            noise=noise,
            dataset=dataset,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "Native MDM SFT expected exactly one model forward, got "
            f"{len(captured)}."
        )
    model_output = captured[0]
    prediction_type = diffusion_prediction_type(diffusion)
    if prediction_type == "epsilon":
        pred_xstart = diffusion._predict_xstart_from_eps(
            x_t,
            timesteps,
            model_output,
        )
    elif prediction_type == "x_start":
        pred_xstart = model_output
    else:  # pragma: no cover - guarded by diffusion_prediction_type.
        raise ValueError(f"Unsupported SFT prediction type: {prediction_type}.")
    return terms, pred_xstart, x_t


def _snr_auxiliary_weight(diffusion: Any, timesteps: torch.Tensor) -> torch.Tensor:
    alpha_bar = torch.from_numpy(diffusion.alphas_cumprod).to(
        device=timesteps.device,
        dtype=torch.float32,
    )[timesteps]
    snr = alpha_bar / (1.0 - alpha_bar)
    return torch.clamp(snr, max=1.0)


def _parameter_norm(parameters: list[torch.nn.Parameter]) -> float:
    if not parameters:
        return 0.0
    return torch.stack(
        [parameter.detach().float().square().sum() for parameter in parameters]
    ).sum().sqrt().item()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


class CountConditioningSFTTrainer:
    def __init__(self, config: CountSFTConfig) -> None:
        config.validate()
        self.config = config
        bootstrap_config = TrainConfig(
            mdm_root=config.mdm_root,
            motionrft_root=config.motionrft_root,
            model_path=config.model_path,
            model_args_path=config.model_args_path,
            prediction_type=config.prediction_type,
            data_cache_dir=config.data_cache_dir,
            data_workers=config.data_workers,
            pin_memory=config.pin_memory,
            seed=config.seed,
            device=config.device,
            precision=config.precision,
            enable_count_conditioning=True,
            train_count_conditioning=True,
            fixed_eval_every=0,
            sample_steps=0,
        )
        self.bootstrap_config = bootstrap_config
        bootstrap_external_repositories(bootstrap_config)
        seed_everything(config.seed)
        self.device = resolve_device(config.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sft_config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        self.human_loader = build_data_loader(
            bootstrap_config,
            prompt_batch_size=config.human_batch_size,
        )
        self.model_args = load_model_args(bootstrap_config)
        self.model, _, self.diffusion, _ = build_mdm(
            bootstrap_config,
            self.model_args,
            self.human_loader,
            self.device,
        )
        self.diffusion_metadata = diffusion_runtime_metadata(
            self.model_args,
            self.diffusion,
        )
        self.lora_report = configure_trainable_policy(
            self.model,
            mode="lora",
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_target_regex=config.lora_target_regex,
        )
        set_count_conditioning_trainable(self.model, True)
        self.training_model = NativeMDMTrainingModel(self.model)

        mean, std = load_humanml_stats(config.mdm_root)
        self.feature_std = torch.from_numpy(std).to(self.device)
        records = load_step_manifest(
            config.step_data_manifest,
            motion_root=config.step_motion_root,
            targets=config.target_values,
            min_frames=config.step_min_frames,
            max_frames=config.step_max_frames,
        )
        training_records, evaluation_records = stratified_step_split(
            records,
            eval_per_target=config.step_eval_samples_per_target,
            split_seed=config.step_split_seed,
            prompt_seed=config.step_prompt_seed,
        )
        balanced_records, self.selection_audit = create_balanced_step_sft_records(
            training_records,
            targets=config.target_values,
            length_bins=config.length_bins,
            seed=config.seed,
            prompt_seed=config.step_prompt_seed,
            max_samples_per_bin_target=config.max_samples_per_bin_target,
        )
        self.selection_audit["held_out_samples"] = len(evaluation_records)
        self.selection_audit["held_out_sample_ids"] = [
            record.sample_id for record in evaluation_records
        ]
        (self.output_dir / "step_sft_selection.json").write_text(
            json.dumps(self.selection_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        step_dataset = StepMotionDataset(
            balanced_records,
            mean=mean,
            std=std,
            max_frames=config.step_max_frames,
        )
        self.step_loader = build_step_data_loader(
            step_dataset,
            batch_size=config.step_batch_size,
            seed=config.seed + 17,
            workers=config.data_workers,
            pin_memory=config.pin_memory,
            balanced_targets=True,
        )
        self.effective_steps_per_epoch = (
            config.steps_per_epoch
            if config.steps_per_epoch > 0
            else len(self.step_loader)
        )
        if self.effective_steps_per_epoch <= 0:
            raise RuntimeError("Balanced step SFT loader contains no full batch.")

        named_trainable = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        self.count_parameters = [
            parameter
            for name, parameter in named_trainable
            if name.startswith("count_conditioning.")
        ]
        self.lora_parameters = [
            parameter
            for name, parameter in named_trainable
            if not name.startswith("count_conditioning.")
        ]
        if not self.count_parameters or not self.lora_parameters:
            raise RuntimeError("SFT requires both count and LoRA trainable tensors.")
        self.trainable_parameters = self.lora_parameters + self.count_parameters
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.lora_parameters,
                    "lr": config.lora_learning_rate,
                    "name": "lora",
                },
                {
                    "params": self.count_parameters,
                    "lr": config.count_learning_rate,
                    "name": "count_conditioning",
                },
            ],
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and config.precision == "fp16",
        )
        self.start_epoch = 1
        self.global_step = 0
        self.anti_jitter_lambda_effective = float(config.anti_jitter_lambda)
        self.anti_jitter_calibrated = config.anti_jitter_auto_grad_ratio <= 0
        self.pending_loader_state: dict[str, torch.Tensor] | None = None
        if config.resume:
            self._load_checkpoint(config.resume)

        runtime = {
            "format": COUNT_SFT_FORMAT,
            "model_path": str(Path(config.model_path).expanduser().resolve()),
            "model_args_path": str(
                Path(config.model_args_path).expanduser().resolve()
            ),
            "mdm_diffusion": self.diffusion_metadata,
            "count_conditioning": count_conditioning_metadata(self.model),
            "lora_adapters": self.lora_report.adapters,
            "lora_trainable_parameters": self.lora_report.trainable_parameters,
            "count_trainable_parameters": sum(
                parameter.numel() for parameter in self.count_parameters
            ),
            "steps_per_epoch": self.effective_steps_per_epoch,
        }
        (self.output_dir / "runtime_metadata.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        LOGGER.info("Native count SFT runtime: %s", json.dumps(runtime, sort_keys=True))

    @staticmethod
    def _next_batch(
        loader: Any,
        iterator: Iterator[Any],
    ) -> tuple[Any, Iterator[Any]]:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _mixed_batch(
        self,
        human_batch: tuple[torch.Tensor, dict[str, Any]],
        step_batch: tuple[torch.Tensor, dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, int]:
        human_motion, human_condition = human_batch
        step_motion, step_condition = step_batch
        target_frames = human_motion.shape[-1]
        if step_motion.shape[-1] > target_frames:
            step_motion = step_motion[..., :target_frames]
            step_condition["y"]["lengths"] = step_condition["y"][
                "lengths"
            ].clamp_max(target_frames)
        elif step_motion.shape[-1] < target_frames:
            step_motion = torch.nn.functional.pad(
                step_motion,
                (0, target_frames - step_motion.shape[-1]),
            )
        motion = torch.cat([human_motion, step_motion], dim=0).to(
            self.device,
            dtype=torch.float32,
            non_blocking=self.config.pin_memory,
        )
        human_count = human_motion.shape[0]
        lengths = torch.cat(
            [human_condition["y"]["lengths"], step_condition["y"]["lengths"]]
        ).long()
        texts = list(human_condition["y"]["text"]) + list(
            step_condition["y"]["text"]
        )
        target_steps = torch.cat(
            [
                torch.full((human_count,), -1, dtype=torch.long),
                step_condition["y"]["target_steps"].long(),
            ]
        )
        return motion, lengths, texts, target_steps, human_count

    def _train_step(
        self,
        human_batch: tuple[torch.Tensor, dict[str, Any]],
        step_batch: tuple[torch.Tensor, dict[str, Any]],
    ) -> dict[str, float]:
        motion, lengths, texts, target_steps, human_count = self._mixed_batch(
            human_batch,
            step_batch,
        )
        self.model.train()
        frozen_text_model = getattr(self.model, "clip_model", None)
        if frozen_text_model is not None:
            frozen_text_model.eval()
        model_kwargs = build_model_kwargs(
            self.model,
            texts,
            lengths,
            motion.shape[-1],
            device=self.device,
            guidance_scale=1.0,
            target_steps=target_steps,
        )
        timesteps = torch.randint(
            0,
            self.diffusion.num_timesteps,
            (motion.shape[0],),
            device=self.device,
        )
        noise = torch.randn_like(motion)
        with autocast_context(self.device, self.config.precision):
            terms, pred_xstart, _ = native_losses_with_pred_xstart(
                model=self.model,
                diffusion=self.diffusion,
                training_model=self.training_model,
                motion=motion,
                timesteps=timesteps,
                noise=noise,
                model_kwargs=model_kwargs,
                dataset=self.human_loader.dataset,
            )
            native_loss = _weighted_group_mean(
                terms["loss"],
                human_count,
                self.config.human_loss_weight,
                self.config.step_loss_weight,
            )
            jitter_per_sample = foot_acceleration_consistency_per_sample(
                pred_xstart,
                motion,
                lengths,
                feature_std=self.feature_std,
            )
            if diffusion_prediction_type(self.diffusion) == "epsilon":
                jitter_per_sample = jitter_per_sample * _snr_auxiliary_weight(
                    self.diffusion,
                    timesteps,
                )
            anti_jitter_loss = _weighted_group_mean(
                jitter_per_sample,
                human_count,
                self.config.human_loss_weight,
                self.config.step_loss_weight,
            )

        native_grad_norm = 0.0
        anti_jitter_grad_norm = 0.0
        if not self.anti_jitter_calibrated:
            native_grad_norm = gradient_l2_norm(
                native_loss,
                self.trainable_parameters,
                retain_graph=True,
            )
            anti_jitter_grad_norm = gradient_l2_norm(
                anti_jitter_loss,
                self.trainable_parameters,
                retain_graph=True,
            )
            self.anti_jitter_lambda_effective = calibrate_loss_lambda(
                native_grad_norm,
                anti_jitter_grad_norm,
                self.config.anti_jitter_auto_grad_ratio,
            )
            self.anti_jitter_calibrated = True
            LOGGER.info(
                "Calibrated anti-jitter lambda=%.8g from native_grad=%.6g, "
                "jitter_grad=%.6g, target_ratio=%.3f",
                self.anti_jitter_lambda_effective,
                native_grad_norm,
                anti_jitter_grad_norm,
                self.config.anti_jitter_auto_grad_ratio,
            )
        total_loss = native_loss + (
            self.anti_jitter_lambda_effective * anti_jitter_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Count SFT loss is non-finite.")

        before = [parameter.detach().clone() for parameter in self.trainable_parameters]
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        for parameter in self.trainable_parameters:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Count SFT produced a non-finite gradient.")
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters,
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1
        update_norm = torch.stack(
            [
                parameter.detach().float().sub(previous.float()).square().sum()
                for parameter, previous in zip(self.trainable_parameters, before)
            ]
        ).sum().sqrt().item()

        metrics: dict[str, float] = {
            "global_step": float(self.global_step),
            "loss": total_loss.detach().float().item(),
            "native_loss": native_loss.detach().float().item(),
            "human_native_loss": terms["loss"][:human_count].mean().detach().float().item(),
            "step_native_loss": terms["loss"][human_count:].mean().detach().float().item(),
            "anti_jitter_loss": anti_jitter_loss.detach().float().item(),
            "anti_jitter_weighted_loss": (
                self.anti_jitter_lambda_effective
                * anti_jitter_loss.detach().float().item()
            ),
            "anti_jitter_lambda": self.anti_jitter_lambda_effective,
            "anti_jitter_target_grad_ratio": self.config.anti_jitter_auto_grad_ratio,
            "anti_jitter_initial_grad_norm": anti_jitter_grad_norm,
            "native_initial_grad_norm": native_grad_norm,
            "grad_norm": float(grad_norm_tensor.detach().float()),
            "update_norm": update_norm,
            "lora_norm": _parameter_norm(self.lora_parameters),
            "count_norm": _parameter_norm(self.count_parameters),
            "count_projection_norm": count_conditioning_metadata(self.model)[
                "projection_norm"
            ],
            "timestep_mean": timesteps.float().mean().item(),
        }
        for name in (
            "rot_mse",
            "rot_mse_unweighted",
            "xstart_mse",
            "xstart_aux",
            "xstart_vel_mse",
            "xstart_vel_aux",
        ):
            if name in terms:
                metrics[name] = terms[name].mean().detach().float().item()
        active_targets = target_steps[human_count:]
        for target in self.config.target_values:
            metrics[f"step_target_{target}_samples"] = float(
                (active_targets == target).sum()
            )
        return metrics

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])

    def _loader_state(self) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        if self.human_loader.generator is not None:
            output["human"] = self.human_loader.generator.get_state()
        if self.step_loader.generator is not None:
            output["step"] = self.step_loader.generator.get_state()
        return output

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        policy = trainable_state_dict(self.model)
        return {
            "format": COUNT_SFT_FORMAT,
            "epoch": epoch,
            "global_step": self.global_step,
            "train_mode": "lora",
            "config": asdict(self.config),
            "policy": policy,
            "policy_id": trainable_policy_state_id(policy),
            "mdm_diffusion": dict(self.diffusion_metadata),
            "count_conditioning": count_conditioning_signature(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": self._rng_state(),
            "loader_rng": self._loader_state(),
            "anti_jitter_lambda_effective": self.anti_jitter_lambda_effective,
            "anti_jitter_calibrated": self.anti_jitter_calibrated,
            "step_selection": self.selection_audit,
        }

    def _save_checkpoint(self, epoch: int) -> Path:
        path = self.output_dir / f"checkpoint_{epoch:06d}.pt"
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch), temporary)
        os.replace(temporary, path)
        shutil.copy2(path, self.output_dir / "latest.pt")
        LOGGER.info("Saved count SFT checkpoint: %s", path)
        return path

    def _load_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format") != COUNT_SFT_FORMAT:
            raise ValueError("Unsupported count SFT checkpoint format.")
        checkpoint_config = payload.get("config", {})
        structural = (
            "model_path",
            "model_args_path",
            "prediction_type",
            "lora_rank",
            "lora_alpha",
            "lora_target_regex",
            "step_targets",
            "length_bins",
            "human_batch_size",
            "step_batch_size",
        )
        mismatched = {
            name: (checkpoint_config.get(name), getattr(self.config, name))
            for name in structural
            if checkpoint_config.get(name) != getattr(self.config, name)
        }
        if mismatched:
            raise ValueError(f"SFT resume structural settings differ: {mismatched}.")
        if payload.get("mdm_diffusion") != self.diffusion_metadata:
            raise ValueError("SFT resume diffusion metadata does not match.")
        validate_count_conditioning_signature(
            self.model,
            payload.get("count_conditioning"),
            source="SFT resume checkpoint",
        )
        load_trainable_state_dict(self.model, payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload.get("scaler", {}))
        if payload.get("rng"):
            self._restore_rng_state(payload["rng"])
        self.pending_loader_state = payload.get("loader_rng")
        self.start_epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload.get("global_step", 0))
        self.anti_jitter_lambda_effective = float(
            payload.get("anti_jitter_lambda_effective", 0.0)
        )
        self.anti_jitter_calibrated = bool(
            payload.get("anti_jitter_calibrated", False)
        )
        LOGGER.info("Resumed count SFT from %s", checkpoint_path)

    def train(self) -> Path:
        if self.pending_loader_state:
            if "human" in self.pending_loader_state and self.human_loader.generator:
                self.human_loader.generator.set_state(
                    self.pending_loader_state["human"].cpu()
                )
            if "step" in self.pending_loader_state and self.step_loader.generator:
                self.step_loader.generator.set_state(
                    self.pending_loader_state["step"].cpu()
                )
        human_iterator = iter(self.human_loader)
        step_iterator = iter(self.step_loader)
        last_checkpoint: Path | None = None
        metrics_path = self.output_dir / "sft_metrics.jsonl"
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            started = time.time()
            epoch_metrics: list[dict[str, float]] = []
            for step_in_epoch in range(1, self.effective_steps_per_epoch + 1):
                human_batch, human_iterator = self._next_batch(
                    self.human_loader,
                    human_iterator,
                )
                step_batch, step_iterator = self._next_batch(
                    self.step_loader,
                    step_iterator,
                )
                metrics = self._train_step(human_batch, step_batch)
                metrics.update(
                    {
                        "epoch": float(epoch),
                        "step_in_epoch": float(step_in_epoch),
                    }
                )
                epoch_metrics.append(metrics)
                if step_in_epoch % self.config.log_every == 0 or step_in_epoch == 1:
                    _append_jsonl(metrics_path, metrics)
                    LOGGER.info("count SFT metrics: %s", json.dumps(metrics, sort_keys=True))
            summary = {
                name: float(np.mean([record[name] for record in epoch_metrics]))
                for name in epoch_metrics[0]
            }
            summary.update(
                {
                    "epoch": float(epoch),
                    "global_step": float(self.global_step),
                    "elapsed_seconds": time.time() - started,
                    "record_type": "epoch",
                }
            )
            _append_jsonl(metrics_path, summary)
            LOGGER.info("count SFT epoch summary: %s", json.dumps(summary, sort_keys=True))
            if epoch % self.config.save_every == 0 or epoch == self.config.epochs:
                last_checkpoint = self._save_checkpoint(epoch)
        if last_checkpoint is None:
            raise RuntimeError("Count SFT completed without saving a checkpoint.")
        return last_checkpoint


__all__ = [
    "COUNT_SFT_FORMAT",
    "CountConditioningSFTTrainer",
    "CountSFTConfig",
    "calibrate_loss_lambda",
    "foot_acceleration_consistency_per_sample",
    "gradient_l2_norm",
    "humanml_foot_position_channels",
    "native_losses_with_pred_xstart",
]
