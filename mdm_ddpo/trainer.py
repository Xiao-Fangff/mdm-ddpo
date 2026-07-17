from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainConfig
from .diffusion import ddim_step_with_logprob
from .lora import (
    LoRAReport,
    configure_trainable_policy,
    load_trainable_state_dict,
    parameter_counts,
    trainable_state_dict,
)
from .rewards import MotionReward, RewardOutput
from .runtime import (
    autocast_context,
    bootstrap_external_repositories,
    build_data_loader,
    build_mdm,
    build_model_kwargs,
    build_policy_model,
    CachedTextEmbedding,
    load_model_args,
    resolve_device,
    resolve_reward_device,
    seed_everything,
    split_text_embeddings,
)
from .tracking import SwanLabTracker, format_training_metrics


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBatch:
    motion: torch.Tensor
    lengths: torch.Tensor
    texts: list[str]


def repeat_prompt_batch(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    texts: list[str],
    samples_per_prompt: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Repeat each conditioning item contiguously for grouped DDPO rollouts."""
    prompt_count = motion.shape[0]
    if lengths.shape[0] != prompt_count or len(texts) != prompt_count:
        raise ValueError("Motion, length, and text prompt counts must match.")
    if samples_per_prompt < 2:
        raise ValueError("Grouped DDPO requires at least two samples per prompt.")
    repeated_motion = motion.repeat_interleave(samples_per_prompt, dim=0)
    repeated_lengths = lengths.repeat_interleave(samples_per_prompt, dim=0)
    repeated_texts = [
        text
        for text in texts
        for _ in range(samples_per_prompt)
    ]
    prompt_ids = torch.arange(prompt_count).repeat_interleave(samples_per_prompt)
    return repeated_motion, repeated_lengths, repeated_texts, prompt_ids


def compute_grouped_advantages(
    rewards: torch.Tensor,
    prompt_ids: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Whiten rewards within each prompt instead of across prompt difficulty."""
    if rewards.ndim != 1 or prompt_ids.shape != rewards.shape:
        raise ValueError("Rewards and prompt_ids must be matching 1-D tensors.")
    if epsilon <= 0:
        raise ValueError("Advantage epsilon must be positive.")

    advantages = torch.zeros_like(rewards)
    group_means: list[torch.Tensor] = []
    group_stds: list[torch.Tensor] = []
    unique_prompt_ids = torch.unique(prompt_ids, sorted=True)
    for prompt_id in unique_prompt_ids:
        mask = prompt_ids == prompt_id
        group_rewards = rewards[mask]
        if group_rewards.numel() < 2:
            raise ValueError(
                f"Prompt group {int(prompt_id)} has fewer than two samples."
            )
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False)
        advantages[mask] = (group_rewards - group_mean) / (group_std + epsilon)
        group_means.append(group_mean)
        group_stds.append(group_std)

    means = torch.stack(group_means)
    stds = torch.stack(group_stds)
    stats = {
        "unique_prompts": float(len(unique_prompt_ids)),
        "reward_within_prompt_std": stds.mean().item(),
        "reward_between_prompt_std": means.std(unbiased=False).item(),
        "zero_variance_prompt_fraction": (
            (stds < epsilon).float().mean().item()
        ),
    }
    return advantages, stats


@dataclass
class Trajectory:
    latents: torch.Tensor
    next_latents: torch.Tensor
    timesteps: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    retrieval_rewards: torch.Tensor
    m2m_rewards: torch.Tensor
    texts: list[str]
    text_embeddings: list[CachedTextEmbedding]
    lengths: torch.Tensor
    gt_motion: torch.Tensor
    prompt_ids: torch.Tensor
    advantages: torch.Tensor | None = None
    group_stats: dict[str, float] | None = None

    @classmethod
    def concatenate(cls, batches: list["Trajectory"]) -> "Trajectory":
        if not batches:
            raise ValueError("Cannot concatenate an empty rollout list.")
        prompt_id_parts: list[torch.Tensor] = []
        prompt_offset = 0
        for batch in batches:
            unique_ids, local_ids = torch.unique(
                batch.prompt_ids,
                sorted=True,
                return_inverse=True,
            )
            prompt_id_parts.append(local_ids + prompt_offset)
            prompt_offset += len(unique_ids)

        return cls(
            latents=torch.cat([batch.latents for batch in batches], dim=0),
            next_latents=torch.cat(
                [batch.next_latents for batch in batches], dim=0
            ),
            timesteps=torch.cat([batch.timesteps for batch in batches], dim=0),
            old_log_probs=torch.cat(
                [batch.old_log_probs for batch in batches], dim=0
            ),
            rewards=torch.cat([batch.rewards for batch in batches], dim=0),
            retrieval_rewards=torch.cat(
                [batch.retrieval_rewards for batch in batches], dim=0
            ),
            m2m_rewards=torch.cat(
                [batch.m2m_rewards for batch in batches], dim=0
            ),
            texts=[text for batch in batches for text in batch.texts],
            text_embeddings=[
                embedding
                for batch in batches
                for embedding in batch.text_embeddings
            ],
            lengths=torch.cat([batch.lengths for batch in batches], dim=0),
            gt_motion=torch.cat([batch.gt_motion for batch in batches], dim=0),
            prompt_ids=torch.cat(prompt_id_parts, dim=0),
        )


class DDPOTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        bootstrap_external_repositories(config)
        seed_everything(config.seed)

        self.device = resolve_device(config.device)
        self.reward_device = resolve_reward_device(config, self.device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32

        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, indent=2, sort_keys=True)

        self.model_args = load_model_args(config)
        self.data_loader = build_data_loader(config)
        self.data_iterator: Any | None = None
        self.model, self.diffusion, self.sample_steps = build_mdm(
            config,
            self.model_args,
            self.data_loader,
            self.device,
        )

        self.lora_report: LoRAReport | None = configure_trainable_policy(
            self.model,
            mode=config.train_mode,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_target_regex=config.lora_target_regex,
        )
        self.model.eval()
        self.policy_model = build_policy_model(
            self.model,
            config.guidance_scale,
        )
        self.policy_model.eval()

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("The DDPO policy has no trainable parameters.")
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and config.precision == "fp16"
        )

        self.reward_model = MotionReward(config, self.reward_device)
        self.start_epoch = 0
        self.global_step = 0
        self.fixed_eval_baseline: dict[str, float] | None = None
        if config.resume:
            self._load_checkpoint(config.resume)
        self.fixed_eval_batch = (
            self._build_fixed_eval_batch()
            if config.fixed_eval_every > 0
            else None
        )

        total, trainable = parameter_counts(self.model)
        LOGGER.info(
            "Policy parameters: trainable=%s / total=%s (%.3f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total,
        )
        if self.lora_report is not None:
            LOGGER.info(
                "Injected %d LoRA adapters with %s trainable parameters.",
                self.lora_report.adapters,
                f"{self.lora_report.trainable_parameters:,}",
            )
        LOGGER.info(
            "Diffusion steps=%d, policy transitions per sample=%d, "
            "policy device=%s, reward device=%s",
            self.sample_steps,
            self.sample_steps - 1,
            self.device,
            self.reward_device,
        )
        if (
            config.precision != "no"
            and config.train_batch_size != config.rollout_batch_size
        ):
            LOGGER.warning(
                "Low-precision old/new log-probability agreement is best when "
                "--train-batch-size equals --rollout-batch-size."
            )

    def preflight_summary(self) -> dict[str, Any]:
        total, trainable = parameter_counts(self.model)
        return {
            "dataset_samples": len(self.data_loader.dataset),
            "diffusion_steps": self.sample_steps,
            "policy_transitions": self.sample_steps - 1,
            "policy_parameters": total,
            "trainable_parameters": trainable,
            "lora_adapters": (
                self.lora_report.adapters if self.lora_report is not None else 0
            ),
            "reward_t5_tensors_loaded_separately": len(
                self.reward_model.missing_checkpoint_keys
            ),
            "reward_backbone_missing_tensors": 0,
            "reward_normalization_mean_max_delta": (
                self.reward_model.normalization_mean_delta
            ),
            "reward_normalization_std_max_delta": (
                self.reward_model.normalization_std_delta
            ),
            "policy_device": str(self.device),
            "reward_device": str(self.reward_device),
            "prompts_per_rollout_batch": (
                self.config.prompts_per_rollout_batch
            ),
            "samples_per_prompt": self.config.samples_per_prompt,
            "fixed_eval_enabled": self.fixed_eval_batch is not None,
        }

    def _build_fixed_eval_batch(self) -> PromptBatch:
        """Select a deterministic prompt pool without advancing training RNG."""
        rng_state = self._rng_state()
        try:
            seed_everything(self.config.fixed_eval_seed)
            loader = DataLoader(
                self.data_loader.dataset,
                batch_size=self.config.prompts_per_rollout_batch,
                shuffle=False,
                num_workers=0,
                drop_last=True,
                collate_fn=self.data_loader.collate_fn,
            )
            motion, condition = next(iter(loader))
        finally:
            self._restore_rng_state(rng_state)
        return PromptBatch(
            motion=motion.detach().cpu(),
            lengths=condition["y"]["lengths"].detach().cpu().long(),
            texts=list(condition["y"]["text"]),
        )

    @torch.no_grad()
    def evaluate_fixed_pool(self) -> dict[str, float]:
        """Evaluate identical prompts and diffusion noise with mean embeddings."""
        if self.fixed_eval_batch is None:
            raise RuntimeError("Fixed evaluation is disabled.")

        motion, lengths, texts, _ = repeat_prompt_batch(
            self.fixed_eval_batch.motion,
            self.fixed_eval_batch.lengths,
            self.fixed_eval_batch.texts,
            self.config.samples_per_prompt,
        )
        motion = motion.to(self.device, dtype=torch.float32)
        batch_size, _, _, num_frames = motion.shape
        model_kwargs = build_model_kwargs(
            self.model,
            texts,
            lengths,
            num_frames,
            device=self.device,
            guidance_scale=self.config.guidance_scale,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.fixed_eval_seed)
        current = torch.randn(
            motion.shape,
            device=self.device,
            dtype=motion.dtype,
            generator=generator,
        )
        for step in tqdm(
            range(self.diffusion.num_timesteps - 1, -1, -1),
            desc="fixed evaluation",
            leave=False,
            dynamic_ncols=True,
        ):
            timestep = torch.full(
                (batch_size,),
                step,
                device=self.device,
                dtype=torch.long,
            )
            with autocast_context(self.device, self.config.precision):
                current, _, _ = ddim_step_with_logprob(
                    self.diffusion,
                    self.policy_model,
                    current,
                    timestep,
                    model_kwargs=model_kwargs,
                    eta=self.config.ddim_eta,
                    mask=model_kwargs["y"]["mask"],
                    clip_denoised=self.config.clip_denoised,
                    generator=generator,
                )

        generated_motion = current.squeeze(2).permute(0, 2, 1).contiguous()
        gt_motion = motion.squeeze(2).permute(0, 2, 1).contiguous()
        previous_mode = self.reward_model.embedding_mode
        self.reward_model.embedding_mode = "mean"
        try:
            reward_output = self.reward_model.score(
                texts=texts,
                generated_motion=generated_motion,
                lengths=lengths,
                gt_motion=gt_motion,
            )
        finally:
            self.reward_model.embedding_mode = previous_mode

        return {
            "eval_reward": reward_output.total.float().mean().item(),
            "eval_reward_std": reward_output.total.float().std(
                unbiased=False
            ).item(),
            "eval_reward_retrieval": (
                reward_output.retrieval.float().mean().item()
            ),
            "eval_reward_m2m": reward_output.m2m.float().mean().item(),
            "eval_samples": float(batch_size),
        }

    def _next_batch(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.data_iterator is None:
            self.data_iterator = iter(self.data_loader)
        try:
            return next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            return next(self.data_iterator)

    @torch.no_grad()
    def _rollout_batch(self, epoch: int, batch_index: int) -> Trajectory:
        motion, condition = self._next_batch()
        lengths = condition["y"]["lengths"].long()
        texts = list(condition["y"]["text"])
        motion, lengths, texts, prompt_ids = repeat_prompt_batch(
            motion,
            lengths,
            texts,
            self.config.samples_per_prompt,
        )
        motion = motion.to(
            self.device,
            dtype=torch.float32,
            non_blocking=self.config.pin_memory,
        )
        batch_size, _, _, num_frames = motion.shape
        model_kwargs = build_model_kwargs(
            self.model,
            texts,
            lengths,
            num_frames,
            device=self.device,
            guidance_scale=self.config.guidance_scale,
        )
        motion_mask = model_kwargs["y"]["mask"]
        cached_text_embeddings = split_text_embeddings(
            model_kwargs["y"]["text_embed"]
        )

        current = torch.randn_like(motion)
        latents: list[torch.Tensor] = []
        next_latents: list[torch.Tensor] = []
        timesteps: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []

        step_iterator = tqdm(
            range(self.diffusion.num_timesteps - 1, -1, -1),
            desc=f"epoch {epoch} rollout {batch_index}",
            leave=False,
            dynamic_ncols=True,
        )
        for step in step_iterator:
            timestep = torch.full(
                (batch_size,),
                step,
                device=self.device,
                dtype=torch.long,
            )
            with autocast_context(self.device, self.config.precision):
                previous, log_prob, _ = ddim_step_with_logprob(
                    self.diffusion,
                    self.policy_model,
                    current,
                    timestep,
                    model_kwargs=model_kwargs,
                    eta=self.config.ddim_eta,
                    mask=motion_mask,
                    clip_denoised=self.config.clip_denoised,
                )
            # t=0 is deterministic and is deliberately absent from PPO.
            if step > 0:
                latents.append(current.detach().cpu())
                next_latents.append(previous.detach().cpu())
                timesteps.append(timestep.detach().cpu())
                log_probs.append(log_prob.detach().float().cpu())
            current = previous

        generated_motion = current.squeeze(2).permute(0, 2, 1).contiguous()
        gt_motion = motion.squeeze(2).permute(0, 2, 1).contiguous()
        reward_output: RewardOutput = self.reward_model.score(
            texts=texts,
            generated_motion=generated_motion,
            lengths=lengths,
            gt_motion=gt_motion,
        )

        return Trajectory(
            latents=torch.stack(latents, dim=1),
            next_latents=torch.stack(next_latents, dim=1),
            timesteps=torch.stack(timesteps, dim=1),
            old_log_probs=torch.stack(log_probs, dim=1),
            rewards=reward_output.total.detach().float().cpu(),
            retrieval_rewards=reward_output.retrieval.detach().float().cpu(),
            m2m_rewards=reward_output.m2m.detach().float().cpu(),
            texts=texts,
            text_embeddings=cached_text_embeddings,
            lengths=lengths.detach().cpu(),
            gt_motion=gt_motion.detach().float().cpu(),
            prompt_ids=prompt_ids,
        )

    def collect_rollouts(self, epoch: int) -> Trajectory:
        self.model.eval()
        # A fresh randomized subset per DDPO epoch also makes epoch-boundary
        # resume reproducible from the DataLoader generator state.
        self.data_iterator = iter(self.data_loader)
        batches = [
            self._rollout_batch(epoch, batch_index)
            for batch_index in range(self.config.rollout_batches_per_epoch)
        ]
        trajectory = Trajectory.concatenate(batches)
        trajectory.advantages, trajectory.group_stats = compute_grouped_advantages(
            trajectory.rewards,
            trajectory.prompt_ids,
            self.config.advantage_epsilon,
        )
        if trajectory.group_stats["zero_variance_prompt_fraction"] > 0:
            LOGGER.warning(
                "%.1f%% of prompt groups have effectively zero reward variance.",
                100.0 * trajectory.group_stats["zero_variance_prompt_fraction"],
            )
        return trajectory

    def _selected_timesteps(
        self,
        num_samples: int,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, int]:
        per_sample = max(
            1,
            int(math.ceil(num_timesteps * self.config.timestep_fraction)),
        )
        selected = torch.stack(
            [
                torch.randperm(num_timesteps)[:per_sample]
                for _ in range(num_samples)
            ],
            dim=0,
        )
        return selected, per_sample

    def optimize(self, trajectory: Trajectory) -> dict[str, float]:
        if trajectory.advantages is None:
            raise ValueError("Advantages must be computed before optimization.")
        self.model.eval()
        num_samples, num_timesteps = trajectory.timesteps.shape
        metric_values: dict[str, list[float]] = {
            "loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "ratio": [],
            "grad_norm": [],
            "skipped_updates": [],
        }

        self.optimizer.zero_grad(set_to_none=True)
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        for inner_epoch in range(self.config.inner_epochs):
            selected_timesteps, timesteps_per_sample = self._selected_timesteps(
                num_samples,
                num_timesteps,
            )
            num_sample_minibatches = math.ceil(
                num_samples / self.config.train_batch_size
            )
            minibatch_order = torch.randperm(num_sample_minibatches)
            progress = tqdm(
                total=num_sample_minibatches * timesteps_per_sample,
                desc=f"DDPO inner epoch {inner_epoch}",
                leave=False,
                dynamic_ncols=True,
            )
            for minibatch_position, minibatch_tensor in enumerate(minibatch_order):
                sample_minibatch_index = int(minibatch_tensor.item())
                start = sample_minibatch_index * self.config.train_batch_size
                end = min(start + self.config.train_batch_size, num_samples)
                sample_indices = torch.arange(start, end)
                advantages = trajectory.advantages[sample_indices].to(
                    self.device,
                    non_blocking=self.config.pin_memory,
                )
                lengths = trajectory.lengths[sample_indices]
                texts = [
                    trajectory.texts[index]
                    for index in sample_indices.tolist()
                ]
                cached_text_embeddings = [
                    trajectory.text_embeddings[index]
                    for index in sample_indices.tolist()
                ]
                model_kwargs = build_model_kwargs(
                    self.model,
                    texts,
                    lengths,
                    trajectory.latents.shape[-1],
                    device=self.device,
                    guidance_scale=self.config.guidance_scale,
                    cached_text_embeddings=cached_text_embeddings,
                )

                group_start = (
                    minibatch_position
                    // self.config.gradient_accumulation_steps
                ) * self.config.gradient_accumulation_steps
                group_end = min(
                    group_start + self.config.gradient_accumulation_steps,
                    num_sample_minibatches,
                )
                accumulation_divisor = (
                    group_end - group_start
                ) * timesteps_per_sample

                minibatch_timesteps = selected_timesteps[sample_indices]
                for timestep_position in range(timesteps_per_sample):
                    time_indices = minibatch_timesteps[:, timestep_position]
                    current = trajectory.latents[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)
                    previous = trajectory.next_latents[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)
                    timesteps = trajectory.timesteps[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)
                    old_log_probs = trajectory.old_log_probs[
                        sample_indices, time_indices
                    ].to(self.device, non_blocking=self.config.pin_memory)

                    with autocast_context(self.device, self.config.precision):
                        _, new_log_probs, _ = ddim_step_with_logprob(
                            self.diffusion,
                            self.policy_model,
                            current,
                            timesteps,
                            model_kwargs=model_kwargs,
                            eta=self.config.ddim_eta,
                            prev_sample=previous,
                            mask=model_kwargs["y"]["mask"],
                            clip_denoised=self.config.clip_denoised,
                        )
                        clipped_advantages = advantages.clamp(
                            -self.config.adv_clip_max,
                            self.config.adv_clip_max,
                        )
                        log_ratio = (
                            new_log_probs - old_log_probs
                        ).clamp(-20.0, 20.0)
                        ratio = log_ratio.exp()
                        unclipped_loss = -clipped_advantages * ratio
                        clipped_loss = -clipped_advantages * ratio.clamp(
                            1.0 - self.config.clip_range,
                            1.0 + self.config.clip_range,
                        )
                        loss = torch.maximum(
                            unclipped_loss,
                            clipped_loss,
                        ).mean()

                    self.scaler.scale(
                        loss / accumulation_divisor
                    ).backward()

                    approx_kl = (
                        0.5
                        * (new_log_probs - old_log_probs).square().mean()
                    )
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_range
                    ).float().mean()
                    metric_values["loss"].append(
                        loss.detach().float().item()
                    )
                    metric_values["approx_kl"].append(
                        approx_kl.detach().float().item()
                    )
                    metric_values["clip_fraction"].append(
                        clip_fraction.detach().float().item()
                    )
                    metric_values["ratio"].append(
                        ratio.detach().float().mean().item()
                    )
                    progress.update(1)

                    should_step = (
                        timestep_position + 1 == timesteps_per_sample
                        and minibatch_position + 1 == group_end
                    )
                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = clip_grad_norm_(
                            trainable_parameters,
                            self.config.max_grad_norm,
                        )
                        finite_gradients = bool(
                            torch.isfinite(grad_norm).item()
                        )
                        if finite_gradients:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.global_step += 1
                            metric_values["grad_norm"].append(
                                float(grad_norm)
                            )
                            metric_values["skipped_updates"].append(0.0)
                        else:
                            LOGGER.warning(
                                "Skipping an optimizer update because the "
                                "gradient norm is non-finite. Consider "
                                "--precision bf16 or no."
                            )
                            # After unscale_, GradScaler has enough information
                            # to reduce its scale even when step is skipped.
                            self.scaler.update()
                            metric_values["skipped_updates"].append(1.0)
                        self.optimizer.zero_grad(set_to_none=True)
            progress.close()

        return {
            name: float(np.mean(values)) if values else 0.0
            for name, values in metric_values.items()
        }

    def _rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        advantages = trajectory.advantages
        assert advantages is not None
        group_stats = trajectory.group_stats or {}
        return {
            "reward": trajectory.rewards.mean().item(),
            "reward_std": trajectory.rewards.std(unbiased=False).item(),
            "reward_retrieval": trajectory.retrieval_rewards.mean().item(),
            "reward_m2m": trajectory.m2m_rewards.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "rollout_samples": float(len(trajectory.rewards)),
            "samples_per_prompt": float(self.config.samples_per_prompt),
            **group_stats,
        }

    def _append_metrics(self, record: dict[str, Any]) -> None:
        with open(self.output_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_fixed_eval(self, record: dict[str, Any]) -> None:
        with open(
            self.output_dir / "fixed_eval.jsonl",
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _fixed_eval_with_deltas(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        if self.fixed_eval_baseline is None:
            raise RuntimeError("Fixed evaluation baseline has not been initialized.")
        result = dict(metrics)
        for name in ("eval_reward", "eval_reward_retrieval", "eval_reward_m2m"):
            result[f"{name}_baseline"] = self.fixed_eval_baseline[name]
            result[f"{name}_delta"] = metrics[name] - self.fixed_eval_baseline[name]
        return result

    def _initialize_fixed_eval(self) -> dict[str, Any] | None:
        if self.fixed_eval_batch is None:
            return None
        if self.fixed_eval_baseline is None:
            if self.config.resume:
                LOGGER.warning(
                    "The resumed checkpoint has no fixed-eval baseline; using "
                    "the resumed policy as the new baseline."
                )
            self.fixed_eval_baseline = self.evaluate_fixed_pool()
        record: dict[str, Any] = {
            "event": "baseline",
            "epoch": self.start_epoch - 1,
            "global_step": self.global_step,
            **self._fixed_eval_with_deltas(self.fixed_eval_baseline),
        }
        self._append_fixed_eval(record)
        LOGGER.info("fixed evaluation baseline: %s", json.dumps(record, sort_keys=True))
        return record

    def _run_fixed_eval(self, epoch: int) -> dict[str, float]:
        metrics = self._fixed_eval_with_deltas(self.evaluate_fixed_pool())
        self._append_fixed_eval(
            {
                "event": "evaluation",
                "epoch": epoch,
                "global_step": self.global_step,
                **metrics,
            }
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
        if self.data_loader.generator is not None:
            state["data_loader"] = self.data_loader.generator.get_state()
        return state

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            for device_index, device_state in enumerate(
                state["cuda"][: torch.cuda.device_count()]
            ):
                torch.cuda.set_rng_state(
                    device_state.cpu(),
                    device=device_index,
                )
        if (
            "data_loader" in state
            and self.data_loader.generator is not None
        ):
            self.data_loader.generator.set_state(
                state["data_loader"].cpu()
            )

    def _save_checkpoint(self, epoch: int) -> Path:
        checkpoint_path = self.output_dir / f"checkpoint_{epoch:06d}.pt"
        temporary_path = checkpoint_path.with_suffix(".tmp")
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
            "train_mode": self.config.train_mode,
            "policy": trainable_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": self._rng_state(),
            "fixed_eval_baseline": self.fixed_eval_baseline,
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        shutil.copy2(checkpoint_path, self.output_dir / "latest.pt")
        LOGGER.info("Saved checkpoint: %s", checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint_mode = checkpoint.get("train_mode")
        if checkpoint_mode != self.config.train_mode:
            raise ValueError(
                f"Checkpoint train_mode={checkpoint_mode!r} does not match "
                f"current mode={self.config.train_mode!r}."
            )
        load_trainable_state_dict(self.model, checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("rng"):
            self._restore_rng_state(checkpoint["rng"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.fixed_eval_baseline = checkpoint.get("fixed_eval_baseline")
        LOGGER.info(
            "Resumed from %s at epoch=%d, global_step=%d",
            checkpoint_path,
            self.start_epoch,
            self.global_step,
        )

    def train(self) -> None:
        if self.start_epoch >= self.config.epochs:
            LOGGER.info(
                "Nothing to do: resume epoch %d >= configured epochs %d.",
                self.start_epoch,
                self.config.epochs,
            )
            return

        with SwanLabTracker(self.config, self.output_dir) as tracker:
            baseline_record = self._initialize_fixed_eval()
            if baseline_record is not None:
                tracker.log(
                    format_training_metrics(
                        baseline_record,
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    ),
                    step=0,
                )
            last_saved_epoch = -1
            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_started = time.time()
                trajectory = self.collect_rollouts(epoch)
                rollout_metrics = self._rollout_metrics(trajectory)
                optimization_metrics = self.optimize(trajectory)
                fixed_eval_metrics: dict[str, float] = {}
                if (
                    self.fixed_eval_batch is not None
                    and (epoch + 1) % self.config.fixed_eval_every == 0
                ):
                    fixed_eval_metrics = self._run_fixed_eval(epoch)
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "elapsed_seconds": time.time() - epoch_started,
                    **rollout_metrics,
                    **optimization_metrics,
                    **fixed_eval_metrics,
                }
                self._append_metrics(record)
                tracker.log(
                    format_training_metrics(
                        record,
                        learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                    ),
                    step=epoch + 1,
                )
                if epoch % self.config.log_every == 0:
                    LOGGER.info(
                        "epoch metrics: %s",
                        json.dumps(record, sort_keys=True),
                    )
                if (epoch + 1) % self.config.save_every == 0:
                    self._save_checkpoint(epoch)
                    last_saved_epoch = epoch

            final_epoch = self.config.epochs - 1
            if last_saved_epoch != final_epoch:
                self._save_checkpoint(final_epoch)
