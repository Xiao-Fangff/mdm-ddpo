from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainConfig


@dataclass
class RewardOutput:
    total: torch.Tensor
    retrieval: torch.Tensor
    m2m: torch.Tensor
    step: torch.Tensor | None = None
    step_mask: torch.Tensor | None = None
    detected_steps: torch.Tensor | None = None
    target_steps: torch.Tensor | None = None
    step_absolute_error: torch.Tensor | None = None

    def means(self) -> dict[str, float]:
        return {
            "reward": self.total.float().mean().item(),
            "reward_retrieval": self.retrieval.float().mean().item(),
            "reward_m2m": self.m2m.float().mean().item(),
            **(
                {"reward_step": self.step.float().mean().item()}
                if self.step is not None
                else {}
            ),
        }


def combine_reward_components(
    retrieval: torch.Tensor,
    m2m: torch.Tensor,
    *,
    retrieval_weight: float,
    m2m_weight: float,
) -> torch.Tensor:
    if retrieval.shape != m2m.shape:
        raise ValueError("Retrieval and M2M rewards must have the same shape.")
    return retrieval_weight * retrieval + m2m_weight * m2m


def apply_step_m2m_policy(
    reward: RewardOutput,
    *,
    step_mask: torch.Tensor,
    m2m_weight: float,
    enabled: bool,
) -> RewardOutput:
    """Optionally remove M2M only from step-labelled sample totals."""
    if step_mask.shape != reward.total.shape:
        raise ValueError("Step mask must match the base reward shape.")
    mask = step_mask.to(device=reward.total.device, dtype=torch.bool)
    total = reward.total
    if not enabled:
        total = total - (
            mask.to(dtype=total.dtype)
            * float(m2m_weight)
            * reward.m2m.to(total)
        )
    return RewardOutput(
        total=total,
        retrieval=reward.retrieval,
        m2m=reward.m2m,
        step=reward.step,
        step_mask=reward.step_mask,
        detected_steps=reward.detected_steps,
        target_steps=reward.target_steps,
        step_absolute_error=reward.step_absolute_error,
    )


def add_step_reward(
    reward: RewardOutput,
    *,
    step: torch.Tensor,
    step_mask: torch.Tensor,
    detected_steps: torch.Tensor,
    target_steps: torch.Tensor,
    absolute_error: torch.Tensor,
    step_weight: float,
) -> RewardOutput:
    if step.shape != reward.total.shape:
        raise ValueError("Step reward must match the base reward shape.")
    for value in (
        step_mask,
        detected_steps,
        target_steps,
        absolute_error,
    ):
        if value.shape != reward.total.shape:
            raise ValueError("Step reward diagnostics must match base rewards.")
    total = reward.total + float(step_weight) * step.to(reward.total)
    return RewardOutput(
        total=total,
        retrieval=reward.retrieval,
        m2m=reward.m2m,
        step=step.to(reward.total),
        step_mask=step_mask.to(device=reward.total.device, dtype=torch.bool),
        detected_steps=detected_steps.to(reward.total.device),
        target_steps=target_steps.to(reward.total.device),
        step_absolute_error=absolute_error.to(reward.total.device),
    )


class MotionReward:
    """Terminal retrieval + M2M reward adapted from RFT_MLD."""

    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        from motionreward.models import MultiReprRetrievalWithLoRA
        from motionreward.utils.config_utils import get_model_config

        checkpoint = torch.load(
            config.reward_backbone_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        checkpoint_config: dict[str, Any] = checkpoint.get("model_config", {})
        model_size = checkpoint_config.get("model_size", "tiny")
        model_config = get_model_config(model_size)
        model_config.update(
            {
                key: value
                for key, value in checkpoint_config.items()
                if key in model_config or key == "use_unified_dim"
            }
        )

        self.model = MultiReprRetrievalWithLoRA(
            t5_path=config.reward_t5_path,
            temp=0.1,
            thr=0.8,
            latent_dim=model_config["latent_dim"],
            unified_dim=model_config["unified_dim"],
            encoder_num_layers=model_config["encoder_num_layers"],
            encoder_num_heads=model_config["encoder_num_heads"],
            encoder_ff_size=model_config["encoder_ff_size"],
            text_num_layers=model_config["text_num_layers"],
            text_num_heads=model_config["text_num_heads"],
            text_ff_size=model_config["text_ff_size"],
            proj_hidden_dim=model_config["proj_hidden_dim"],
            proj_num_layers=model_config["proj_num_layers"],
            use_unified_dim=model_config.get("use_unified_dim", True),
            lora_rank=int(checkpoint_config.get("lora_rank", 16)),
            lora_alpha=float(checkpoint_config.get("lora_alpha", 32)),
            lora_dropout=0.0,
        )
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected MotionReward checkpoint tensors: "
                + ", ".join(incompatible.unexpected_keys[:8])
            )
        missing_non_t5 = [
            key for key in incompatible.missing_keys if not key.startswith("clip.")
        ]
        if missing_non_t5:
            raise RuntimeError(
                "MotionReward backbone is incomplete; missing non-T5 tensors: "
                + ", ".join(missing_non_t5[:8])
            )
        self.missing_checkpoint_keys = tuple(incompatible.missing_keys)
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.device = device
        self.retrieval_weight = float(config.retrieval_weight)
        self.m2m_weight = float(config.m2m_weight)
        self.embedding_mode = config.reward_embedding_mode

        mdm_stats_root = Path(config.mdm_root) / "dataset" / "HumanML3D"
        reward_stats_root = (
            Path(config.motionrft_root) / "datasets" / "humanml3d"
        )
        stats_paths = {
            "MDM mean": mdm_stats_root / "Mean.npy",
            "MDM std": mdm_stats_root / "Std.npy",
            "MotionReward mean": reward_stats_root / "Mean.npy",
            "MotionReward std": reward_stats_root / "Std.npy",
        }
        missing_stats = [
            f"{label}: {path}"
            for label, path in stats_paths.items()
            if not path.exists()
        ]
        if missing_stats:
            raise FileNotFoundError(
                "Missing HumanML normalization statistics:\n  "
                + "\n  ".join(missing_stats)
            )
        self.mdm_mean = self._load_stat(stats_paths["MDM mean"])
        self.mdm_std = self._load_stat(stats_paths["MDM std"])
        self.reward_mean = self._load_stat(
            stats_paths["MotionReward mean"]
        )
        self.reward_std = self._load_stat(
            stats_paths["MotionReward std"]
        )
        self.normalization_mean_delta = float(
            (self.mdm_mean - self.reward_mean).abs().max().item()
        )
        self.normalization_std_delta = float(
            (self.mdm_std - self.reward_std).abs().max().item()
        )

    def _load_stat(self, path: Path) -> torch.Tensor:
        values = np.load(path)
        if values.shape != (263,):
            raise ValueError(
                f"Expected 263-D HumanML stats at {path}, got {values.shape}."
            )
        return torch.as_tensor(
            values,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 263)

    def _to_reward_normalization(
        self,
        mdm_normalized_motion: torch.Tensor,
    ) -> torch.Tensor:
        raw_motion = (
            mdm_normalized_motion * self.mdm_std + self.mdm_mean
        )
        return (raw_motion - self.reward_mean) / self.reward_std

    def _select_embedding(
        self,
        latent: torch.Tensor,
        distribution: torch.distributions.Normal,
    ) -> torch.Tensor:
        value = distribution.loc if self.embedding_mode == "mean" else latent
        return value.squeeze(0)

    def _embedding_rng_context(self):
        """Keep deterministic mean rewards from perturbing training RNG."""
        devices: list[int] = []
        if self.device.type == "cuda":
            devices.append(
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
        return torch.random.fork_rng(
            devices=devices,
            enabled=self.embedding_mode == "mean",
        )

    def _text_embedding(self, texts: list[str]) -> torch.Tensor:
        from motionreward.models.lora_retrieval import process_T5_outputs

        lengths, token_embeddings, _ = process_T5_outputs(
            texts,
            self.model.clip,
            device=self.device,
        )
        latent, distribution = self.model.encode_text(token_embeddings, lengths)
        return self._select_embedding(latent, distribution)

    def _motion_embedding(
        self,
        motion: torch.Tensor,
        lengths: list[int],
        *,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent, distribution = self.model.encode_motion(
            motion.float(),
            lengths,
            repr_type="263",
            timestep=timestep,
        )
        return self._select_embedding(latent, distribution)

    @torch.no_grad()
    def score(
        self,
        *,
        texts: list[str],
        generated_motion: torch.Tensor,
        lengths: torch.Tensor,
        gt_motion: torch.Tensor,
    ) -> RewardOutput:
        """Score normalized HumanML features shaped [B, T, 263]."""

        if generated_motion.ndim != 3 or generated_motion.shape[-1] != 263:
            raise ValueError("generated_motion must have shape [B, T, 263].")
        if gt_motion.shape != generated_motion.shape:
            raise ValueError("gt_motion and generated_motion must have matching shapes.")
        if len(texts) != generated_motion.shape[0]:
            raise ValueError("The number of texts must match the motion batch.")

        return_device = generated_motion.device
        generated = self._to_reward_normalization(
            generated_motion.to(self.device, dtype=torch.float32)
        )
        ground_truth = self._to_reward_normalization(
            gt_motion.to(self.device, dtype=torch.float32)
        )
        length_list = lengths.detach().cpu().long().tolist()
        batch = generated.shape[0]
        clean_timestep = torch.zeros(
            (),
            device=self.device,
            dtype=torch.long,
        )

        # MotionReward's encoders call dist.rsample() even when we select the
        # distribution mean. Forking RNG here makes mean-mode scoring truly
        # deterministic from the training loop's point of view.
        with self._embedding_rng_context():
            if self.retrieval_weight != 0:
                # RFT_MLD invokes the motion encoder independently per reward.
                generated_retrieval_embedding = self._motion_embedding(
                    generated,
                    length_list,
                    timestep=clean_timestep,
                )
                text_embedding = self._text_embedding(texts)
                retrieval = F.cosine_similarity(
                    text_embedding,
                    generated_retrieval_embedding,
                    dim=-1,
                )
            else:
                retrieval = torch.zeros(batch, device=self.device)

            if self.m2m_weight != 0:
                gt_embedding = self._motion_embedding(ground_truth, length_list)
                generated_m2m_embedding = self._motion_embedding(
                    generated,
                    length_list,
                    timestep=clean_timestep,
                )
                m2m = F.cosine_similarity(
                    gt_embedding,
                    generated_m2m_embedding,
                    dim=-1,
                )
            else:
                m2m = torch.zeros(batch, device=self.device)

        total = combine_reward_components(
            retrieval,
            m2m,
            retrieval_weight=self.retrieval_weight,
            m2m_weight=self.m2m_weight,
        )
        return RewardOutput(
            total=total.to(return_device),
            retrieval=retrieval.to(return_device),
            m2m=m2m.to(return_device),
        )
