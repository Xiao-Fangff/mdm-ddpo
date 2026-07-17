from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MDM_ROOT = "/home/zhiwei/projects/motion-diffusion-model"
DEFAULT_MOTIONRFT_ROOT = "/home/zhiwei/projects/MotionRFT"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CACHE_DIR = str(DEFAULT_PROJECT_ROOT / ".cache" / "mdm")


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
    reward_backbone_path: str = (
        DEFAULT_MOTIONRFT_ROOT
        + "/checkpoints/motionreward/stage1_retrieval_backbone_r128.pth"
    )
    reward_t5_path: str = DEFAULT_MOTIONRFT_ROOT + "/deps/sentence-t5-large"
    output_dir: str = "outputs/mdm_ddpo"
    resume: str = ""

    dataset: str = "humanml"
    split: str = "train"
    data_cache_dir: str = DEFAULT_DATA_CACHE_DIR
    data_workers: int = 4
    pin_memory: bool = True

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
    rollout_batch_size: int = 4
    rollout_batches_per_epoch: int = 4

    train_batch_size: int = 4
    inner_epochs: int = 1
    timestep_fraction: float = 1.0
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1.0e-4
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    clip_range: float = 1.0e-4
    adv_clip_max: float = 5.0
    advantage_epsilon: float = 1.0e-8

    train_mode: str = "lora"
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_target_regex: str = (
        r"(seqTransDecoder|seqTransEncoder|embed_text|output_process)"
    )

    retrieval_weight: float = 1.0
    m2m_weight: float = 1.0
    reward_embedding_mode: str = "sample"

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

    def validate(self) -> None:
        if self.dataset != "humanml":
            raise ValueError(
                "The MotionReward 263-D adapter currently supports dataset='humanml' only."
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
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive.")
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
        missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        "--reward-backbone-path",
        default=TrainConfig.reward_backbone_path,
        help="MotionReward Stage-1 retrieval backbone.",
    )
    paths.add_argument(
        "--reward-t5-path",
        default=TrainConfig.reward_t5_path,
        help="Local sentence-t5-large directory used by MotionReward.",
    )
    paths.add_argument("--output-dir", default=TrainConfig.output_dir)
    paths.add_argument("--resume", default="", help="DDPO checkpoint to resume.")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default="humanml", choices=["humanml"])
    data.add_argument("--split", default="train")
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
    rollout.add_argument("--rollout-batch-size", type=int, default=4)
    rollout.add_argument("--rollout-batches-per-epoch", type=int, default=4)

    train = parser.add_argument_group("DDPO optimization")
    train.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Number of rollout samples per PPO forward batch.",
    )
    train.add_argument("--inner-epochs", type=int, default=1)
    train.add_argument("--timestep-fraction", type=float, default=1.0)
    train.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of sample minibatches to accumulate; every selected "
            "diffusion timestep is included before an optimizer step."
        ),
    )
    train.add_argument("--learning-rate", type=float, default=1.0e-4)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--adam-weight-decay", type=float, default=1.0e-4)
    train.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--clip-range", type=float, default=1.0e-4)
    train.add_argument("--adv-clip-max", type=float, default=5.0)
    train.add_argument("--advantage-epsilon", type=float, default=1.0e-8)

    policy = parser.add_argument_group("policy parameters")
    policy.add_argument("--train-mode", choices=["lora", "full"], default="lora")
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
        "--reward-embedding-mode",
        choices=["sample", "mean"],
        default="sample",
        help="'sample' matches RFT_MLD; 'mean' reduces reward variance.",
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
        config.rollout_batch_size = 2
        config.rollout_batches_per_epoch = 1
        config.train_batch_size = 2
        config.inner_epochs = 1
        config.data_workers = 0
        config.save_every = 1
    config.validate()
    return config
