from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig


@dataclass(frozen=True)
class CachedTextEmbedding:
    kind: str
    values: torch.Tensor
    padding_mask: torch.Tensor | None = None


def bootstrap_external_repositories(config: TrainConfig) -> None:
    """Expose the two reference repositories without modifying them."""

    mdm_root = str(Path(config.mdm_root).resolve())
    motionrft_root = str(Path(config.motionrft_root).resolve())
    # MDM uses top-level imports such as from model.mdm import MDM.
    for path in (motionrft_root, mdm_root):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@contextlib.contextmanager
def working_directory(path: str | os.PathLike[str]) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def resolve_reward_device(config: TrainConfig, policy_device: torch.device) -> torch.device:
    if config.reward_device == "same":
        return policy_device
    return resolve_device(config.reward_device)


def load_model_args(config: TrainConfig) -> SimpleNamespace:
    with open(config.model_args_path, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    values["batch_size"] = config.prompts_per_rollout_batch
    values["dataset"] = config.dataset
    values["device"] = (
        int(config.device.split(":", 1)[1])
        if config.device.startswith("cuda:")
        else -1
    )
    # Compatibility with older checkpoint argument files.
    values.setdefault("pred_len", 0)
    values.setdefault("context_len", 0)
    values.setdefault("emb_policy", "add")
    values.setdefault("multi_target_cond", False)
    values.setdefault("multi_encoder_type", "single")
    values.setdefault("target_enc_layers", 1)
    values.setdefault("lambda_target_loc", 0.0)
    return SimpleNamespace(**values)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _prepare_data_cache(config: TrainConfig) -> Path:
    cache_root = (
        Path(config.data_cache_dir).expanduser().resolve()
        if config.data_cache_dir
        else (Path(config.output_dir).expanduser().resolve() / "cache")
    )
    (cache_root / "dataset").mkdir(parents=True, exist_ok=True)
    glove_source = (Path(config.mdm_root).resolve() / "glove").resolve()
    glove_link = cache_root / "glove"
    if glove_link.is_symlink() and glove_link.resolve() != glove_source:
        raise RuntimeError(
            f"Existing cache glove link points to {glove_link.resolve()}, "
            f"expected {glove_source}."
        )
    if not glove_link.exists():
        glove_link.symlink_to(glove_source, target_is_directory=True)
    return cache_root


def build_dataset(config: TrainConfig, *, split: str) -> Any:
    """Build a HumanML dataset for an explicit split using local cache only."""
    from data_loaders.get_data import get_dataset_class

    cache_root = _prepare_data_cache(config)
    dataset_class = get_dataset_class(config.dataset)
    # Explicit absolute roots avoid relying on the launch directory.
    return dataset_class(
        split=split,
        num_frames=None,
        mode="train",
        abs_path=str(Path(config.mdm_root).resolve()),
        cache_path=str(cache_root),
        device=None,
        autoregressive=False,
    )


def build_data_loader(config: TrainConfig) -> DataLoader:
    from data_loaders.get_data import get_collate_fn

    dataset = build_dataset(config, split=config.split)
    prompt_batch_size = config.prompts_per_rollout_batch
    if len(dataset) < prompt_batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than prompt batch size "
            f"{prompt_batch_size}."
        )
    collate_fn = get_collate_fn(
        config.dataset,
        hml_mode="train",
        batch_size=prompt_batch_size,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=prompt_batch_size,
        shuffle=True,
        num_workers=config.data_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        persistent_workers=config.data_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _build_respaced_diffusion(
    base_diffusion: Any,
    model_args: SimpleNamespace,
    sample_steps: int,
) -> Any:
    if sample_steps == base_diffusion.num_timesteps:
        return base_diffusion

    from diffusion import gaussian_diffusion as gd
    from diffusion.respace import SpacedDiffusion, space_timesteps

    if sample_steps > int(model_args.diffusion_steps):
        raise ValueError(
            f"sample_steps={sample_steps} exceeds checkpoint diffusion_steps="
            f"{model_args.diffusion_steps}."
        )
    betas = gd.get_named_beta_schedule(
        model_args.noise_schedule,
        int(model_args.diffusion_steps),
        1.0,
    )
    return SpacedDiffusion(
        use_timesteps=space_timesteps(int(model_args.diffusion_steps), [sample_steps]),
        betas=betas,
        model_mean_type=base_diffusion.model_mean_type,
        model_var_type=base_diffusion.model_var_type,
        loss_type=base_diffusion.loss_type,
        rescale_timesteps=base_diffusion.rescale_timesteps,
        lambda_rcxyz=base_diffusion.lambda_rcxyz,
        lambda_vel=base_diffusion.lambda_vel,
        lambda_fc=base_diffusion.lambda_fc,
        lambda_target_loc=base_diffusion.lambda_target_loc,
        data_rep=base_diffusion.data_rep,
    )


def build_mdm(
    config: TrainConfig,
    model_args: SimpleNamespace,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, Any, Any, int]:
    from utils.model_util import create_model_and_diffusion, load_saved_model

    # MDM's BERT path is relative in the reference source.
    with working_directory(config.mdm_root):
        model, base_diffusion = create_model_and_diffusion(model_args, data_loader)
        load_saved_model(model, config.model_path, use_avg=True)

    sample_steps = config.sample_steps or int(model_args.diffusion_steps)
    if sample_steps < 2:
        raise ValueError("DDPO needs at least two diffusion steps.")
    diffusion = _build_respaced_diffusion(base_diffusion, model_args, sample_steps)
    model.to(device)
    model.eval()
    return model, diffusion, base_diffusion, sample_steps


class ClassifierFreeGuidance(nn.Module):
    """Lightweight MDM classifier-free guidance wrapper with shared weights."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("Classifier-free guidance requires conditioning.")
        conditional = self.model(x, timesteps, y)
        unconditional_y = dict(y)
        unconditional_y["uncond"] = True
        unconditional = self.model(x, timesteps, unconditional_y)
        scale = y["scale"].reshape(-1, 1, 1, 1)
        return unconditional + scale * (conditional - unconditional)


def build_policy_model(model: nn.Module, guidance_scale: float) -> nn.Module:
    if guidance_scale == 1.0:
        return model
    return ClassifierFreeGuidance(model)


def lengths_to_motion_mask(
    lengths: torch.Tensor,
    num_frames: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    lengths = lengths.to(device=device, dtype=torch.long)
    frames = torch.arange(num_frames, device=device)
    return (frames.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)


def build_model_kwargs(
    model: nn.Module,
    texts: list[str],
    lengths: torch.Tensor,
    num_frames: int,
    *,
    device: torch.device,
    guidance_scale: float,
    cached_text_embeddings: list[CachedTextEmbedding] | None = None,
) -> dict[str, dict[str, Any]]:
    lengths = lengths.to(device=device, dtype=torch.long)
    mask = lengths_to_motion_mask(lengths, num_frames, device=device)
    y: dict[str, Any] = {
        "text": list(texts),
        "lengths": lengths,
        "mask": mask,
        "scale": torch.full(
            (len(texts),),
            float(guidance_scale),
            device=device,
            dtype=torch.float32,
        ),
    }
    if cached_text_embeddings is None:
        with torch.no_grad():
            y["text_embed"] = canonicalize_text_embeddings(
                model.encode_text(y["text"])
            )
    else:
        if len(cached_text_embeddings) != len(texts):
            raise ValueError("Cached text embedding count does not match texts.")
        y["text_embed"] = collate_text_embeddings(
            cached_text_embeddings,
            device=device,
        )
    return {"y": y}


def canonicalize_text_embeddings(value: Any) -> Any:
    """Use the same contiguous layout in rollout and PPO recomputation."""

    if isinstance(value, tuple):
        return tuple(
            item.contiguous() if torch.is_tensor(item) else item
            for item in value
        )
    if torch.is_tensor(value):
        return value.contiguous()
    raise TypeError(f"Unsupported text embedding type: {type(value)!r}")


def split_text_embeddings(value: Any) -> list[CachedTextEmbedding]:
    """Split a batched frozen text-encoder output into per-prompt CPU entries."""

    if isinstance(value, tuple):
        encoded, padding_mask = value
        if encoded.ndim != 3 or padding_mask.ndim != 2:
            raise ValueError("Unexpected BERT text embedding shapes.")
        entries = []
        for index in range(encoded.shape[1]):
            entries.append(
                CachedTextEmbedding(
                    kind="bert",
                    values=encoded[:, index].detach().cpu().contiguous(),
                    padding_mask=padding_mask[index].detach().cpu().contiguous(),
                )
            )
        return entries
    if torch.is_tensor(value):
        if value.ndim < 2:
            raise ValueError("Unexpected tensor text embedding shape.")
        return [
            CachedTextEmbedding(
                kind="tensor",
                values=value[:, index].detach().cpu().contiguous(),
            )
            for index in range(value.shape[1])
        ]
    raise TypeError(f"Unsupported text embedding type: {type(value)!r}")


def collate_text_embeddings(
    entries: list[CachedTextEmbedding],
    *,
    device: torch.device,
) -> Any:
    if not entries:
        raise ValueError("Cannot collate an empty text embedding list.")
    kinds = {entry.kind for entry in entries}
    if len(kinds) != 1:
        raise ValueError(f"Mixed cached text embedding kinds: {sorted(kinds)}")
    kind = entries[0].kind
    if kind == "bert":
        max_length = max(entry.values.shape[0] for entry in entries)
        feature_dim = entries[0].values.shape[1]
        encoded = torch.zeros(
            max_length,
            len(entries),
            feature_dim,
            device=device,
            dtype=entries[0].values.dtype,
        )
        padding_mask = torch.ones(
            len(entries),
            max_length,
            device=device,
            dtype=torch.bool,
        )
        for index, entry in enumerate(entries):
            length = entry.values.shape[0]
            encoded[:length, index] = entry.values.to(device)
            if entry.padding_mask is None:
                raise ValueError("BERT cache entry is missing its padding mask.")
            padding_mask[index, :length] = entry.padding_mask.to(device)
        return encoded, padding_mask
    if kind == "tensor":
        return torch.stack(
            [entry.values.to(device) for entry in entries],
            dim=1,
        )
    raise ValueError(f"Unknown cached text embedding kind: {kind}")


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "no":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)
