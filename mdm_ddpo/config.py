from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MDM_ROOT = "/home/zhiwei/projects/motion-diffusion-model"
DEFAULT_MOTIONRFT_ROOT = "/home/zhiwei/projects/MotionRFT"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CACHE_DIR = str(DEFAULT_PROJECT_ROOT / ".cache" / "mdm")
DEFAULT_STEP_DATA_ROOT = (
    DEFAULT_MOTIONRFT_ROOT
    + "/RFT_MLD/third_party/motion-rule-data/offline_reward_validation/"
    "walk_step_five_manifests_0_6_random400"
)
DEFAULT_STEP_DATA_MANIFEST = DEFAULT_STEP_DATA_ROOT + "/sample_manifest.jsonl"
DEFAULT_STEP_MOTION_ROOT = (
    DEFAULT_MOTIONRFT_ROOT + "/RFT_MLD/third_party/motion-rule"
)


@dataclass
class TrainConfig:
    mdm_root: str = DEFAULT_MDM_ROOT
    motionrft_root: str = DEFAULT_MOTIONRFT_ROOT
    model_path: str = (
        DEFAULT_MDM_ROOT
        + "/save/humanml_trans_dec_512_bert/model000600000.pt"
    )
    model_args_path: str = (
        DEFAULT_MDM_ROOT + "/save/humanml_trans_dec_512_bert/args.json"
    )
    prediction_type: str = "auto"
    enable_count_conditioning: bool = False
    train_count_conditioning: bool = True
    initial_policy_path: str = ""
    reward_backbone_path: str = (
        DEFAULT_MOTIONRFT_ROOT
        + "/checkpoints/motionreward/stage1_retrieval_backbone_r128.pth"
    )
    reward_t5_path: str = DEFAULT_MOTIONRFT_ROOT + "/deps/sentence-t5-large"
    reward_calibration_path: str = ""
    step_reward_calibration_path: str = ""
    step_data_manifest: str = DEFAULT_STEP_DATA_MANIFEST
    step_motion_root: str = DEFAULT_STEP_MOTION_ROOT
    step_detector_root: str = DEFAULT_STEP_MOTION_ROOT
    output_dir: str = "outputs/mdm_ddpo"
    resume: str = ""
    reset_optimizer_on_resume: bool = False

    dataset: str = "humanml"
    split: str = "train"
    eval_split: str = "val"
    data_cache_dir: str = DEFAULT_DATA_CACHE_DIR
    data_workers: int = 4
    pin_memory: bool = True
    enable_step_reward: bool = False
    step_data_ratio: float = 0.25
    step_rollout_source: str = "reference"
    step_synthetic_seed: int = 20260719
    step_targets: str = "1,2,3,4,5,6"
    step_split_seed: int = 20260600
    step_prompt_seed: int = 20260612
    step_eval_samples_per_target: int = 8
    step_balanced_sampling: bool = True
    step_min_frames: int = 40
    step_max_frames: int = 196

    seed: int = 42
    device: str = "cuda:0"
    reward_device: str = "same"
    precision: str = "bf16"
    allow_tf32: bool = True

    epochs: int = 100
    sample_steps: int = 0
    guidance_scale: float = 2.5
    ddim_eta: float = 1.0
    clip_denoised: bool = False
    rollout_batch_size: int = 32
    rollout_batches_per_epoch: int = 4
    samples_per_prompt: int = 4
    step_samples_per_prompt: int = 16

    train_batch_size: int = 32
    inner_epochs: int = 1
    timestep_fraction: float = 0.5
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1.0e-4
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    clip_range: float = 1.0e-4
    adv_clip_max: float = 5.0
    advantage_epsilon: float = 1.0e-8
    advantage_mode: str = "group_whiten"
    advantage_std_floor_quantile: str = "p25"
    advantage_retrieval_weight: float = 0.5
    advantage_m2m_weight: float = 0.5
    advantage_step_weight: float = 0.25
    step_advantage_retrieval_weight: float | None = None
    step_advantage_m2m_weight: float | None = None
    step_advantage_step_weight: float | None = None
    log_prob_audit_tolerance: float = 1.0e-4
    anchor_lambda: float = 0.0
    anchor_auto_grad_ratio: float = 0.0
    anchor_batch_size: int = 0

    train_mode: str = "lora"
    train_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_target_regex: str = (
        r"(seqTransDecoder|seqTransEncoder|embed_text|output_process)"
    )

    retrieval_weight: float = 1.0
    m2m_weight: float = 1.0
    step_use_m2m_reward: bool = True
    step_reward_weight: float = 0.5
    step_reward_mode: str = "exp"
    step_reward_temperature: float = 1.0
    step_reward_linear_tolerance: float = 3.0
    step_detector_backend: str = "progressive"
    step_detector_fps: int = 20
    step_detector_lead_threshold: float = 0.138
    step_detector_rgdno_threshold: float = 0.005
    step_soft_lead_temperature: float = 1.0
    step_soft_length_temperature: float = 1.0
    step_soft_progress_temperature: float = 1.0
    step_soft_cluster_gap_seconds: float = 0.15
    step_ankle_high_frequency_cutoff_hz: float = 4.0
    step_soft_huber_delta: float = 1.0
    step_soft_exact_bonus: float = 0.15
    step_soft_target_scale_floor: float = 0.25
    reward_embedding_mode: str = "mean"

    fixed_eval_every: int = 5
    fixed_eval_seed: int = 20260717
    fixed_eval_prompts: int = 128
    fixed_eval_samples_per_prompt: int = 4
    fixed_step_eval_samples_per_prompt: int = 16
    fixed_eval_bootstrap_samples: int = 2000
    fixed_eval_pool_path: str = ""
    fixed_step_eval_pool_path: str = ""
    early_stop_patience: int = 8
    early_stop_min_delta: float = 0.0
    early_stop_min_delta_mode: str = "auto"
    early_stop_se_multiplier: float = 1.0
    checkpoint_feasible_se_multiplier: float = 1.0

    save_every: int = 1
    log_every: int = 1
    use_swanlab: bool = False
    swanlab_project: str = "mdm-ddpo"
    swanlab_run_name: str = ""
    swanlab_workspace: str = ""
    swanlab_mode: str = "online"
    swanlab_log_dir: str = ""
    preflight: bool = False
    dry_run: bool = False
    allow_uncalibrated_soft_step_reward: bool = False

    def validate(self) -> None:
        if self.prediction_type not in {"auto", "x_start", "epsilon"}:
            raise ValueError(
                "--prediction-type must be one of: auto, x_start, epsilon."
            )
        if not self.train_lora and self.train_mode != "lora":
            raise ValueError("--no-train-lora requires --train-mode lora.")
        if self.resume and self.initial_policy_path:
            raise ValueError(
                "--resume and --initial-policy-path are mutually exclusive."
            )
        if self.initial_policy_path and not self.enable_count_conditioning:
            raise ValueError(
                "--initial-policy-path requires --enable-count-conditioning."
            )
        if self.dataset != "humanml":
            raise ValueError(
                "The MotionReward 263-D adapter currently supports dataset='humanml' only."
            )
        if self.split != "train":
            raise ValueError(
                "DDPO rollouts must use the HumanML3D train split; "
                "set --split train."
            )
        if self.eval_split not in {"val", "test"}:
            raise ValueError("--eval-split must be 'val' or 'test'.")
        if self.fixed_eval_every > 0 and self.eval_split == "test":
            raise ValueError(
                "The HumanML3D test split cannot be used for checkpoint "
                "selection; use --eval-split val."
            )
        if not 0 < self.ddim_eta <= 1:
            raise ValueError(
                "DDPO requires a stochastic DDIM sampler with --ddim-eta in (0, 1]."
            )
        if self.sample_steps == 1 or self.sample_steps < 0:
            raise ValueError(
                "--sample-steps must be 0 (checkpoint default) or at least 2."
            )
        for name in (
            "epochs",
            "rollout_batch_size",
            "rollout_batches_per_epoch",
            "train_batch_size",
            "inner_epochs",
            "gradient_accumulation_steps",
            "save_every",
            "log_every",
            "samples_per_prompt",
            "step_samples_per_prompt",
            "fixed_eval_prompts",
            "fixed_eval_samples_per_prompt",
            "fixed_step_eval_samples_per_prompt",
            "fixed_eval_bootstrap_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
        if self.anchor_lambda < 0:
            raise ValueError("--anchor-lambda cannot be negative.")
        if self.anchor_auto_grad_ratio < 0:
            raise ValueError("--anchor-auto-grad-ratio cannot be negative.")
        if self.anchor_lambda > 0 and self.anchor_auto_grad_ratio > 0:
            raise ValueError(
                "Use either a fixed --anchor-lambda or "
                "--anchor-auto-grad-ratio, not both."
            )
        if self.anchor_batch_size < 0:
            raise ValueError("--anchor-batch-size cannot be negative.")
        if self.samples_per_prompt < 2:
            raise ValueError(
                "--samples-per-prompt must be at least 2 for grouped advantages."
            )
        if self.step_samples_per_prompt < 2:
            raise ValueError(
                "--step-samples-per-prompt must be at least 2 for grouped "
                "step advantages."
            )
        if self.fixed_eval_samples_per_prompt < 2:
            raise ValueError(
                "--fixed-eval-samples-per-prompt must be at least 2."
            )
        if self.fixed_step_eval_samples_per_prompt < 2:
            raise ValueError(
                "--fixed-step-eval-samples-per-prompt must be at least 2."
            )
        if not self.enable_step_reward and (
            self.rollout_batch_size % self.samples_per_prompt != 0
        ):
            raise ValueError(
                "--rollout-batch-size must be divisible by "
                "--samples-per-prompt."
            )
        rollout_samples = (
            self.rollout_batch_size * self.rollout_batches_per_epoch
        )
        if rollout_samples % self.train_batch_size != 0:
            raise ValueError(
                "rollout_batch_size * rollout_batches_per_epoch must be "
                "divisible by --train-batch-size so every PPO minibatch has "
                "the same number of samples."
            )
        if self.fixed_eval_every < 0:
            raise ValueError("--fixed-eval-every cannot be negative.")
        if self.early_stop_patience < 0:
            raise ValueError("--early-stop-patience cannot be negative.")
        if self.early_stop_min_delta < 0:
            raise ValueError("--early-stop-min-delta cannot be negative.")
        if self.early_stop_min_delta_mode not in {"fixed", "auto"}:
            raise ValueError(
                "--early-stop-min-delta-mode must be 'fixed' or 'auto'."
            )
        if self.early_stop_se_multiplier < 0:
            raise ValueError("--early-stop-se-multiplier cannot be negative.")
        if self.checkpoint_feasible_se_multiplier < 0:
            raise ValueError(
                "--checkpoint-feasible-se-multiplier cannot be negative."
            )
        if self.fixed_eval_every > 0 and not self.reward_calibration_path:
            raise ValueError(
                "Fixed validation checkpoint selection requires "
                "--reward-calibration-path."
            )
        if self.advantage_mode not in {
            "group_centered",
            "group_whiten",
            "group_shrink",
            "component_shrink",
        }:
            raise ValueError(
                "--advantage-mode must be group_centered, group_whiten, "
                "group_shrink, or component_shrink."
            )
        if self.advantage_std_floor_quantile not in {"p25", "p50"}:
            raise ValueError(
                "--advantage-std-floor-quantile must be 'p25' or 'p50'."
            )
        if self.advantage_mode in {"group_shrink", "component_shrink"}:
            if (
                not self.reward_calibration_path
                and not (
                    self.enable_step_reward
                    and self.advantage_mode == "group_shrink"
                )
            ):
                raise ValueError(
                    f"--advantage-mode {self.advantage_mode} requires "
                    "--reward-calibration-path."
                )
        if self.enable_step_reward:
            if self.step_rollout_source not in {"reference", "synthetic"}:
                raise ValueError(
                    "--step-rollout-source must be reference or synthetic."
                )
            if (
                self.step_rollout_source == "synthetic"
                and self.step_use_m2m_reward
            ):
                raise ValueError(
                    "Synthetic step rollouts have no target-specific GT "
                    "motion; use --no-step-use-m2m-reward."
                )
            if not 0 < self.step_data_ratio < 1:
                raise ValueError("--step-data-ratio must be in (0,1).")
            requested_step_samples = (
                self.rollout_batch_size * self.step_data_ratio
            )
            if not math.isclose(
                requested_step_samples,
                round(requested_step_samples),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    "--step-data-ratio must produce an integral number of "
                    "step motion samples per rollout batch."
                )
            if self.step_rollout_samples % self.step_samples_per_prompt != 0:
                raise ValueError(
                    "The step motion allocation "
                    "(rollout_batch_size * step_data_ratio) must be divisible "
                    "by --step-samples-per-prompt so every step prompt has a "
                    "complete group."
                )
            if self.humanml_rollout_samples % self.samples_per_prompt != 0:
                raise ValueError(
                    "The HumanML motion allocation "
                    "(rollout_batch_size * (1 - step_data_ratio)) must be "
                    "divisible by --samples-per-prompt so every HumanML prompt "
                    "has a complete group."
                )
            if self.prompts_per_rollout_batch < 2:
                raise ValueError(
                    "Step mixing requires at least two prompts per rollout batch."
                )
            if self.step_prompts_per_rollout_batch <= 0:
                raise ValueError(
                    "--step-data-ratio is too small for the rollout prompt batch."
                )
            if self.humanml_prompts_per_rollout_batch <= 0:
                raise ValueError(
                    "Step mixing must retain at least one HumanML prompt per batch."
                )
            if self.advantage_mode == "group_shrink":
                raise ValueError(
                    "group_shrink total calibration is incompatible with an added "
                    "step component; use component_shrink, group_centered, or "
                    "group_whiten."
                )
            if (
                self.advantage_mode == "component_shrink"
                and self.effective_step_advantage_step_weight > 0
                and not self.step_reward_calibration_path
            ):
                raise ValueError(
                    "Step component_shrink requires "
                    "--step-reward-calibration-path."
                )
            try:
                from .step_data import parse_step_targets

                parse_step_targets(self.step_targets)
            except ValueError as exc:
                raise ValueError(f"Invalid --step-targets: {exc}") from exc
            if self.step_eval_samples_per_target <= 0:
                raise ValueError(
                    "--step-eval-samples-per-target must be positive."
                )
            if self.step_min_frames <= 0 or self.step_max_frames < self.step_min_frames:
                raise ValueError("Step motion frame limits are invalid.")
            if self.step_detector_backend not in {"progressive", "rgdno"}:
                raise ValueError(
                    "--step-detector-backend must be progressive or rgdno."
                )
            if self.step_reward_mode not in {
                "exp",
                "linear",
                "exact",
                "negative_l1",
                "soft_huber_exact",
            }:
                raise ValueError(
                    "--step-reward-mode must be exp, linear, exact, "
                    "negative_l1, or soft_huber_exact."
                )
            if (
                self.step_reward_mode == "soft_huber_exact"
                and self.step_detector_backend != "progressive"
            ):
                raise ValueError(
                    "soft_huber_exact requires the progressive detector "
                    "metadata backend."
                )
            if (
                self.step_reward_mode == "soft_huber_exact"
                and not self.step_reward_calibration_path
                and not self.allow_uncalibrated_soft_step_reward
            ):
                raise ValueError(
                    "soft_huber_exact requires --step-reward-calibration-path "
                    "for immutable per-target scales."
                )
            if self.step_detector_fps <= 0:
                raise ValueError("--step-detector-fps must be positive.")
            for name in (
                "step_reward_temperature",
                "step_reward_linear_tolerance",
                "step_detector_lead_threshold",
                "step_detector_rgdno_threshold",
                "step_soft_lead_temperature",
                "step_soft_length_temperature",
                "step_soft_progress_temperature",
                "step_soft_cluster_gap_seconds",
                "step_ankle_high_frequency_cutoff_hz",
                "step_soft_huber_delta",
                "step_soft_target_scale_floor",
            ):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"--{name.replace('_', '-')} must be positive.")
            if (
                self.step_ankle_high_frequency_cutoff_hz
                >= self.step_detector_fps / 2
            ):
                raise ValueError(
                    "--step-ankle-high-frequency-cutoff-hz must be below "
                    "the detector Nyquist frequency."
                )
            if (
                not math.isfinite(self.step_soft_exact_bonus)
                or self.step_soft_exact_bonus < 0
            ):
                raise ValueError("--step-soft-exact-bonus cannot be negative.")
            if self.step_reward_weight < 0:
                raise ValueError("--step-reward-weight cannot be negative.")
        if (
            self.advantage_retrieval_weight < 0
            or self.advantage_m2m_weight < 0
            or self.advantage_step_weight < 0
            or (
                self.advantage_retrieval_weight == 0
                and self.advantage_m2m_weight == 0
            )
        ):
            raise ValueError(
                "Component advantage weights must be non-negative and not "
                "both zero."
            )
        for name in (
            "step_advantage_retrieval_weight",
            "step_advantage_m2m_weight",
            "step_advantage_step_weight",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(
                    f"--{name.replace('_', '-')} cannot be negative."
                )
        if self.enable_step_reward and all(
            weight == 0
            for weight in (
                self.effective_step_advantage_retrieval_weight,
                self.effective_step_advantage_m2m_weight,
                self.effective_step_advantage_step_weight,
            )
        ):
            raise ValueError(
                "At least one step-labelled component advantage weight must "
                "be non-zero."
            )
        if not 0 < self.timestep_fraction <= 1:
            raise ValueError("--timestep-fraction must be in (0, 1].")
        if self.train_mode not in {"lora", "full"}:
            raise ValueError("--train-mode must be either 'lora' or 'full'.")
        if self.train_mode == "lora" and self.lora_rank <= 0:
            raise ValueError("--lora-rank must be positive in LoRA mode.")
        if self.train_mode == "lora" and self.lora_alpha <= 0:
            raise ValueError("--lora-alpha must be positive in LoRA mode.")
        if self.reward_embedding_mode not in {"sample", "mean"}:
            raise ValueError("--reward-embedding-mode must be 'sample' or 'mean'.")
        if self.retrieval_weight == 0 and self.m2m_weight == 0:
            raise ValueError("At least one reward weight must be non-zero.")
        if self.precision not in {"no", "fp16", "bf16"}:
            raise ValueError("--precision must be one of: no, fp16, bf16.")
        if self.data_workers < 0:
            raise ValueError("--data-workers cannot be negative.")
        if self.swanlab_mode not in {"disabled", "online", "local", "offline"}:
            raise ValueError(
                "--swanlab-mode must be one of: disabled, online, local, offline."
            )
        if self.use_swanlab and not self.swanlab_project.strip():
            raise ValueError("--swanlab-project cannot be empty when SwanLab is enabled.")
        for name in (
            "learning_rate",
            "max_grad_norm",
            "clip_range",
            "advantage_epsilon",
            "log_prob_audit_tolerance",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")

        required = {
            "MDM root": self.mdm_root,
            "MotionRFT root": self.motionrft_root,
            "MDM checkpoint": self.model_path,
            "MDM args": self.model_args_path,
            "reward backbone": self.reward_backbone_path,
            "reward T5": self.reward_t5_path,
        }
        if self.enable_step_reward:
            required.update(
                {
                    "step data manifest": self.step_data_manifest,
                    "step motion root": self.step_motion_root,
                }
            )
            if self.step_detector_backend == "progressive":
                required["step detector root"] = self.step_detector_root
        missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
        if (
            self.reward_calibration_path
            and not Path(self.reward_calibration_path).expanduser().exists()
        ):
            missing.append(
                "reward calibration: "
                f"{Path(self.reward_calibration_path).expanduser()}"
            )
        if (
            self.step_reward_calibration_path
            and not Path(self.step_reward_calibration_path).expanduser().exists()
        ):
            missing.append(
                "step reward calibration: "
                f"{Path(self.step_reward_calibration_path).expanduser()}"
            )
        if (
            self.initial_policy_path
            and not Path(self.initial_policy_path).expanduser().exists()
        ):
            missing.append(
                "initial policy: "
                f"{Path(self.initial_policy_path).expanduser()}"
            )
        if missing:
            raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def prompts_per_rollout_batch(self) -> int:
        """Total prompt groups assembled into one mixed rollout batch."""
        return (
            self.humanml_prompts_per_rollout_batch
            + self.step_prompts_per_rollout_batch
        )

    @property
    def human_samples_per_prompt(self) -> int:
        """Alias that makes the two independent grouped rollout sizes clear."""
        return self.samples_per_prompt

    @property
    def step_rollout_samples(self) -> int:
        """Number of step-labelled motion samples in one rollout batch."""
        if not self.enable_step_reward:
            return 0
        return int(round(self.rollout_batch_size * self.step_data_ratio))

    @property
    def humanml_rollout_samples(self) -> int:
        """Number of ordinary HumanML motion samples in one rollout batch."""
        return self.rollout_batch_size - self.step_rollout_samples

    @property
    def step_target_values(self) -> tuple[int, ...]:
        from .step_data import parse_step_targets

        return parse_step_targets(self.step_targets)

    @property
    def step_prompts_per_rollout_batch(self) -> int:
        if not self.enable_step_reward:
            return 0
        return self.step_rollout_samples // self.step_samples_per_prompt

    @property
    def humanml_prompts_per_rollout_batch(self) -> int:
        return self.humanml_rollout_samples // self.samples_per_prompt

    @property
    def humanml_fixed_eval_prompts_per_batch(self) -> int:
        """Maximum held-out HumanML prompts evaluated in one diffusion batch."""
        return max(
            1,
            self.rollout_batch_size // self.fixed_eval_samples_per_prompt,
        )

    @property
    def step_fixed_eval_prompts_per_batch(self) -> int:
        """Maximum held-out step prompts evaluated in one diffusion batch."""
        return max(
            1,
            self.rollout_batch_size // self.fixed_step_eval_samples_per_prompt,
        )

    @property
    def effective_step_advantage_retrieval_weight(self) -> float:
        return float(
            self.advantage_retrieval_weight
            if self.step_advantage_retrieval_weight is None
            else self.step_advantage_retrieval_weight
        )

    @property
    def effective_step_advantage_m2m_weight(self) -> float:
        return float(
            self.advantage_m2m_weight
            if self.step_advantage_m2m_weight is None
            else self.step_advantage_m2m_weight
        )

    @property
    def effective_step_advantage_step_weight(self) -> float:
        return float(
            self.advantage_step_weight
            if self.step_advantage_step_weight is None
            else self.step_advantage_step_weight
        )

    def step_detector_config(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "backend": self.step_detector_backend,
            "fps": int(self.step_detector_fps),
            "lead_threshold": float(self.step_detector_lead_threshold),
            "rgdno_threshold": float(self.step_detector_rgdno_threshold),
        }
        if self.step_reward_mode == "soft_huber_exact":
            output["soft"] = {
                "lead_temperature": float(self.step_soft_lead_temperature),
                "length_temperature": float(
                    self.step_soft_length_temperature
                ),
                "progress_temperature": float(
                    self.step_soft_progress_temperature
                ),
                "cluster_gap_seconds": float(
                    self.step_soft_cluster_gap_seconds
                ),
                "ankle_high_frequency_cutoff_hz": float(
                    self.step_ankle_high_frequency_cutoff_hz
                ),
            }
        return output

    def step_reward_config(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "mode": self.step_reward_mode,
            "temperature": float(self.step_reward_temperature),
            "linear_tolerance": float(self.step_reward_linear_tolerance),
        }
        if self.step_reward_mode == "soft_huber_exact":
            output.update(
                {
                    "huber_delta": float(self.step_soft_huber_delta),
                    "exact_bonus": float(self.step_soft_exact_bonus),
                    "target_scale_floor": float(
                        self.step_soft_target_scale_floor
                    ),
                }
            )
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune HumanML3D MDM with DDPO and MotionReward."
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--mdm-root", default=DEFAULT_MDM_ROOT)
    paths.add_argument("--motionrft-root", default=DEFAULT_MOTIONRFT_ROOT)
    paths.add_argument(
        "--model-path",
        default=TrainConfig.model_path,
        help="Pretrained MDM checkpoint.",
    )
    paths.add_argument(
        "--model-args-path",
        default=TrainConfig.model_args_path,
        help="args.json paired with the pretrained MDM checkpoint.",
    )
    paths.add_argument(
        "--prediction-type",
        choices=["auto", "x_start", "epsilon"],
        default="auto",
        help=(
            "How the MDM checkpoint parameterizes its denoising output. "
            "'auto' reads prediction_type or predict_epsilon from args.json "
            "and falls back to x_start for legacy checkpoints."
        ),
    )
    paths.add_argument(
        "--enable-count-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject a zero-initialized explicit count embedding for targets "
            "0..6; ordinary HumanML uses the no-count id."
        ),
    )
    paths.add_argument(
        "--train-count-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow DDPO to update an installed count-conditioning module.",
    )
    paths.add_argument(
        "--initial-policy-path",
        default="",
        help=(
            "Load LoRA/count tensors from a native-SFT policy checkpoint "
            "without restoring DDPO optimizer or epoch state."
        ),
    )
    paths.add_argument(
        "--reward-backbone-path",
        default=TrainConfig.reward_backbone_path,
        help="MotionReward Stage-1 retrieval backbone.",
    )
    paths.add_argument(
        "--reward-t5-path",
        default=TrainConfig.reward_t5_path,
        help="Local sentence-t5-large directory used by MotionReward.",
    )
    paths.add_argument(
        "--reward-calibration-path",
        default="",
        help=(
            "Immutable reward_calibration.json generated by "
            "tools/calibrate_reward_stats.py."
        ),
    )
    paths.add_argument(
        "--step-reward-calibration-path",
        default="",
        help="Immutable hard-step reward calibration JSON.",
    )
    paths.add_argument(
        "--step-data-manifest",
        default=DEFAULT_STEP_DATA_MANIFEST,
    )
    paths.add_argument(
        "--step-motion-root",
        default=DEFAULT_STEP_MOTION_ROOT,
        help="Root used to resolve relative features_263_path entries.",
    )
    paths.add_argument(
        "--step-detector-root",
        default=DEFAULT_STEP_MOTION_ROOT,
        help="Motion-Rule-co root for the progressive hard detector.",
    )
    paths.add_argument("--output-dir", default=TrainConfig.output_dir)
    paths.add_argument("--resume", default="", help="DDPO checkpoint to resume.")
    paths.add_argument(
        "--reset-optimizer-on-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load policy weights and training position from --resume, but "
            "start AdamW and GradScaler from fresh state. Recommended when "
            "migrating an older run to materially different DDPO settings."
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default="humanml", choices=["humanml"])
    data.add_argument("--split", default="train")
    data.add_argument(
        "--eval-split",
        default="val",
        choices=["val", "test"],
        help=(
            "Held-out split used by fixed validation. The test split is "
            "forbidden while fixed validation selects checkpoints."
        ),
    )
    data.add_argument(
        "--data-cache-dir",
        default=DEFAULT_DATA_CACHE_DIR,
        help="Shared writable MDM dataset cache.",
    )
    data.add_argument("--data-workers", type=int, default=4)
    data.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    data.add_argument(
        "--enable-step-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    data.add_argument("--step-data-ratio", type=float, default=0.25)
    data.add_argument(
        "--step-rollout-source",
        choices=["reference", "synthetic"],
        default="reference",
        help=(
            "Use pseudo-labelled reference motions or target/length-"
            "independent synthetic step conditions."
        ),
    )
    data.add_argument("--step-synthetic-seed", type=int, default=20260719)
    data.add_argument("--step-targets", default="1,2,3,4,5,6")
    data.add_argument("--step-split-seed", type=int, default=20260600)
    data.add_argument("--step-prompt-seed", type=int, default=20260612)
    data.add_argument("--step-eval-samples-per-target", type=int, default=8)
    data.add_argument(
        "--step-balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Interleave step targets uniformly so short rollout windows do "
            "not over-sample an easy or difficult target."
        ),
    )
    data.add_argument("--step-min-frames", type=int, default=40)
    data.add_argument("--step-max-frames", type=int, default=196)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default="cuda:0")
    runtime.add_argument(
        "--reward-device",
        default="same",
        help="'same', 'cpu', or an explicit torch device.",
    )
    runtime.add_argument(
        "--precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    runtime.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    rollout = parser.add_argument_group("rollout")
    rollout.add_argument("--epochs", type=int, default=100)
    rollout.add_argument(
        "--sample-steps",
        type=int,
        default=0,
        help="0 uses the checkpoint diffusion step count.",
    )
    rollout.add_argument("--guidance-scale", type=float, default=2.5)
    rollout.add_argument("--ddim-eta", type=float, default=1.0)
    rollout.add_argument(
        "--clip-denoised",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    rollout.add_argument("--rollout-batch-size", type=int, default=32)
    rollout.add_argument("--rollout-batches-per-epoch", type=int, default=4)
    rollout.add_argument(
        "--samples-per-prompt",
        type=int,
        default=4,
        help=(
            "Independent motions generated for each HumanML prompt. "
            "Advantages are normalized within these prompt groups."
        ),
    )
    rollout.add_argument(
        "--step-samples-per-prompt",
        type=int,
        default=16,
        help=(
            "Independent motions generated for each step-labelled prompt. "
            "This is independent of --samples-per-prompt."
        ),
    )

    train = parser.add_argument_group("DDPO optimization")
    train.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Number of rollout samples per PPO forward batch.",
    )
    train.add_argument("--inner-epochs", type=int, default=1)
    train.add_argument(
        "--timestep-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of stochastic diffusion transitions sampled per motion "
            "for each PPO epoch. The validated 50-step default is 0.5."
        ),
    )
    train.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help=(
            "Number of sample minibatches to accumulate; every selected "
            "diffusion timestep is included before an optimizer step."
        ),
    )
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--adam-weight-decay", type=float, default=1.0e-4)
    train.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--clip-range", type=float, default=1.0e-4)
    train.add_argument("--adv-clip-max", type=float, default=5.0)
    train.add_argument("--advantage-epsilon", type=float, default=1.0e-8)
    train.add_argument(
        "--log-prob-audit-tolerance",
        type=float,
        default=1.0e-4,
        help=(
            "Maximum allowed absolute old/new log-probability difference "
            "before the first optimizer update for each rollout."
        ),
    )
    train.add_argument(
        "--anchor-lambda",
        type=float,
        default=0.0,
        help="Fixed coefficient for the optional native MDM diffusion anchor.",
    )
    train.add_argument(
        "--anchor-auto-grad-ratio",
        type=float,
        default=0.0,
        help=(
            "Calibrate anchor lambda once so its initial gradient norm is "
            "this fraction of the PPO gradient norm (for example 0.1 or 0.2)."
        ),
    )
    train.add_argument(
        "--anchor-batch-size",
        type=int,
        default=0,
        help=(
            "Distinct real motions per optimizer update used by the anchor; "
            "0 uses up to --train-batch-size."
        ),
    )
    train.add_argument(
        "--advantage-mode",
        choices=[
            "group_centered",
            "group_whiten",
            "group_shrink",
            "component_shrink",
        ],
        default="group_whiten",
        help=(
            "Subtract each prompt's reward mean, then either divide by every "
            "prompt's own standard deviation (group_whiten, validated default) "
            "apply one global scale (group_centered), use a fixed calibrated "
            "shrinkage floor (group_shrink), or shrink retrieval/M2M "
            "separately before combining them (component_shrink)."
        ),
    )
    train.add_argument(
        "--advantage-std-floor-quantile",
        choices=["p25", "p50"],
        default="p25",
        help="Calibration within-group std quantile used as shrinkage floor.",
    )
    train.add_argument(
        "--advantage-retrieval-weight",
        type=float,
        default=0.5,
    )
    train.add_argument(
        "--advantage-m2m-weight",
        type=float,
        default=0.5,
    )
    train.add_argument(
        "--advantage-step-weight",
        type=float,
        default=0.25,
    )
    train.add_argument(
        "--step-advantage-retrieval-weight",
        type=float,
        default=None,
        help=(
            "Retrieval advantage weight for step-labelled groups only; "
            "default inherits --advantage-retrieval-weight."
        ),
    )
    train.add_argument(
        "--step-advantage-m2m-weight",
        type=float,
        default=None,
        help=(
            "M2M advantage weight for step-labelled groups only; default "
            "inherits --advantage-m2m-weight."
        ),
    )
    train.add_argument(
        "--step-advantage-step-weight",
        type=float,
        default=None,
        help=(
            "Hard-step advantage weight for step-labelled groups; default "
            "inherits --advantage-step-weight."
        ),
    )

    policy = parser.add_argument_group("policy parameters")
    policy.add_argument("--train-mode", choices=["lora", "full"], default="lora")
    policy.add_argument(
        "--train-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Update injected LoRA tensors. Disable after count SFT to run "
            "count-adapter-only DDPO while preserving ordinary HumanML."
        ),
    )
    policy.add_argument("--lora-rank", type=int, default=8)
    policy.add_argument("--lora-alpha", type=float, default=8.0)
    policy.add_argument(
        "--lora-target-regex",
        default=TrainConfig.lora_target_regex,
    )

    reward = parser.add_argument_group("reward")
    reward.add_argument("--retrieval-weight", type=float, default=1.0)
    reward.add_argument("--m2m-weight", type=float, default=1.0)
    reward.add_argument(
        "--step-use-m2m-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include M2M in step-labelled prompt rewards. Disabling this "
            "masks M2M only for step samples; HumanML M2M is unchanged."
        ),
    )
    reward.add_argument("--step-reward-weight", type=float, default=0.5)
    reward.add_argument(
        "--step-reward-mode",
        choices=[
            "exp",
            "linear",
            "exact",
            "negative_l1",
            "soft_huber_exact",
        ],
        default="exp",
    )
    reward.add_argument("--step-reward-temperature", type=float, default=1.0)
    reward.add_argument(
        "--step-reward-linear-tolerance",
        type=float,
        default=3.0,
    )
    reward.add_argument(
        "--step-detector-backend",
        choices=["progressive", "rgdno"],
        default="progressive",
    )
    reward.add_argument("--step-detector-fps", type=int, default=20)
    reward.add_argument(
        "--step-detector-lead-threshold",
        type=float,
        default=0.138,
    )
    reward.add_argument(
        "--step-detector-rgdno-threshold",
        type=float,
        default=0.005,
    )
    reward.add_argument("--step-soft-lead-temperature", type=float, default=1.0)
    reward.add_argument("--step-soft-length-temperature", type=float, default=1.0)
    reward.add_argument(
        "--step-soft-progress-temperature",
        type=float,
        default=1.0,
    )
    reward.add_argument(
        "--step-soft-cluster-gap-seconds",
        type=float,
        default=0.15,
    )
    reward.add_argument(
        "--step-ankle-high-frequency-cutoff-hz",
        type=float,
        default=4.0,
    )
    reward.add_argument("--step-soft-huber-delta", type=float, default=1.0)
    reward.add_argument("--step-soft-exact-bonus", type=float, default=0.15)
    reward.add_argument(
        "--step-soft-target-scale-floor",
        type=float,
        default=0.25,
    )
    reward.add_argument(
        "--reward-embedding-mode",
        choices=["sample", "mean"],
        default="mean",
        help=(
            "'mean' is the low-variance DDPO default; 'sample' reproduces "
            "RFT_MLD's stochastic embedding draw."
        ),
    )
    reward.add_argument(
        "--fixed-eval-every",
        type=int,
        default=5,
        help="Evaluate a fixed prompt/noise pool every N epochs; 0 disables it.",
    )
    reward.add_argument(
        "--fixed-eval-seed",
        type=int,
        default=20260717,
    )
    reward.add_argument(
        "--fixed-eval-prompts",
        type=int,
        default=128,
        help="Number of deterministic prompts in the fixed evaluation pool.",
    )
    reward.add_argument(
        "--fixed-eval-samples-per-prompt",
        type=int,
        default=4,
        help="Generated motions evaluated for every held-out prompt.",
    )
    reward.add_argument(
        "--fixed-step-eval-samples-per-prompt",
        type=int,
        default=16,
        help=(
            "Generated motions evaluated for every held-out step prompt. "
            "This is independent of --fixed-eval-samples-per-prompt."
        ),
    )
    reward.add_argument(
        "--fixed-eval-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap replicates used for fixed-validation standard errors.",
    )
    reward.add_argument(
        "--fixed-eval-pool-path",
        default="",
        help=(
            "Optional shared fixed_eval_pool.pt. It is created when missing "
            "and copied into every run directory."
        ),
    )
    reward.add_argument(
        "--fixed-step-eval-pool-path",
        default="",
        help="Optional shared fixed_step_eval_pool.pt.",
    )
    reward.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help=(
            "Stop after this many fixed evaluations without a new best reward; "
            "0 disables early stopping."
        ),
    )
    reward.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum fixed-eval reward increase required to reset patience.",
    )
    reward.add_argument(
        "--early-stop-min-delta-mode",
        choices=["fixed", "auto"],
        default="auto",
        help=(
            "In auto mode, the effective minimum is at least the balanced "
            "validation bootstrap standard error."
        ),
    )
    reward.add_argument(
        "--early-stop-se-multiplier",
        type=float,
        default=1.0,
    )
    reward.add_argument(
        "--checkpoint-feasible-se-multiplier",
        type=float,
        default=1.0,
        help=(
            "Allow each component delta down to minus this many paired "
            "bootstrap standard errors when checking balanced feasibility."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument("--save-every", type=int, default=1)
    output.add_argument("--log-every", type=int, default=1)

    tracking = parser.add_argument_group("SwanLab tracking")
    tracking.add_argument(
        "--use-swanlab",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record epoch-level training curves with SwanLab.",
    )
    tracking.add_argument("--swanlab-project", default="mdm-ddpo")
    tracking.add_argument(
        "--swanlab-run-name",
        default="",
        help="Optional SwanLab run name; empty lets SwanLab generate one.",
    )
    tracking.add_argument(
        "--swanlab-workspace",
        default="",
        help="Optional SwanLab workspace/organization.",
    )
    tracking.add_argument(
        "--swanlab-mode",
        choices=["disabled", "online", "local", "offline"],
        default="online",
    )
    tracking.add_argument(
        "--swanlab-log-dir",
        default="",
        help="Local SwanLab log directory; defaults to OUTPUT_DIR/swanlab.",
    )
    output.add_argument(
        "--preflight",
        action="store_true",
        help="Load and validate all components, then exit before rollouts.",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one tiny four-step epoch for integration testing.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> TrainConfig:
    namespace = build_parser().parse_args(argv)
    config = TrainConfig(**vars(namespace))
    if config.dry_run:
        config.epochs = 1
        config.sample_steps = 4
        config.rollout_batch_size = 4 if config.enable_step_reward else 2
        config.rollout_batches_per_epoch = 1
        config.train_batch_size = config.rollout_batch_size
        config.samples_per_prompt = 2
        config.step_samples_per_prompt = 2
        config.fixed_eval_prompts = 1
        config.fixed_eval_samples_per_prompt = 2
        config.fixed_step_eval_samples_per_prompt = 2
        if config.enable_step_reward:
            config.step_data_ratio = 0.5
            config.step_eval_samples_per_target = 1
        config.inner_epochs = 1
        config.data_workers = 0
        config.save_every = 1
    config.validate()
    return config
