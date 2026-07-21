from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .count_conditioning import (
    COUNT_CONDITIONING_VERSION,
    set_count_conditioning_trainable,
    validate_count_conditioning_signature,
)
from .lora import configure_trainable_policy, load_trainable_state_dict
from .runtime import validate_diffusion_runtime_metadata


def load_policy_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Policy checkpoint must contain a mapping: {resolved}.")
    return payload


def policy_count_signature(payload: dict[str, Any]) -> dict[str, Any]:
    signature = payload.get("count_conditioning")
    if signature is None:
        return {
            "enabled": False,
            "version": COUNT_CONDITIONING_VERSION,
        }
    if not isinstance(signature, dict):
        raise TypeError("Policy count-conditioning signature must be a mapping.")
    return dict(signature)


def policy_uses_count_conditioning(payload: dict[str, Any]) -> bool:
    return bool(policy_count_signature(payload).get("enabled", False))


def trainable_policy_state_id(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def policy_checkpoint_id(payload: dict[str, Any]) -> str:
    policy = payload.get("policy")
    if not isinstance(policy, dict) or not all(
        isinstance(value, torch.Tensor) for value in policy.values()
    ):
        raise ValueError("Policy checkpoint has no tensor state mapping.")
    calculated = trainable_policy_state_id(policy)
    stored = payload.get("policy_id")
    if stored is not None and str(stored) != calculated:
        raise ValueError("Policy checkpoint id does not match its tensors.")
    return calculated


def configure_and_load_policy_checkpoint(
    model: nn.Module,
    payload: dict[str, Any],
    *,
    diffusion_metadata: dict[str, Any],
    model_path: str | Path,
    source: str,
) -> dict[str, Any]:
    """Install the checkpoint's trainable structure and load every tensor."""

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{source} has no valid config mapping.")
    train_mode = payload.get("train_mode")
    if train_mode != "lora":
        raise ValueError(
            f"{source} uses unsupported train_mode={train_mode!r}; expected lora."
        )
    required_lora = ("lora_rank", "lora_alpha", "lora_target_regex")
    missing_lora = [name for name in required_lora if name not in config]
    if missing_lora:
        raise KeyError(f"{source} is missing LoRA settings: {missing_lora}.")
    report = configure_trainable_policy(
        model,
        mode="lora",
        lora_rank=int(config["lora_rank"]),
        lora_alpha=float(config["lora_alpha"]),
        lora_target_regex=str(config["lora_target_regex"]),
    )
    count_signature = policy_count_signature(payload)
    if count_signature.get("enabled", False):
        set_count_conditioning_trainable(model, True)
    validate_count_conditioning_signature(
        model,
        count_signature,
        source=source,
    )
    artifact_diffusion = payload.get("mdm_diffusion")
    if not isinstance(artifact_diffusion, dict):
        raise ValueError(f"{source} has no audited MDM diffusion metadata.")
    validate_diffusion_runtime_metadata(
        artifact_diffusion,
        diffusion_metadata,
        source=source,
    )
    artifact_model_path = config.get("model_path")
    if artifact_model_path is not None:
        expected = Path(model_path).expanduser().resolve()
        actual = Path(artifact_model_path).expanduser().resolve()
        if actual != expected:
            raise ValueError(
                f"{source} uses a different base MDM: expected={expected}, "
                f"actual={actual}."
            )
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{source} has no policy state mapping.")
    load_trainable_state_dict(model, policy)
    return {
        "policy_id": policy_checkpoint_id(payload),
        "train_mode": "lora",
        "lora_rank": int(config["lora_rank"]),
        "lora_alpha": float(config["lora_alpha"]),
        "lora_target_regex": str(config["lora_target_regex"]),
        "lora_adapters": report.adapters if report is not None else 0,
        "count_conditioning": count_signature,
    }


__all__ = [
    "configure_and_load_policy_checkpoint",
    "load_policy_checkpoint_payload",
    "policy_checkpoint_id",
    "policy_count_signature",
    "policy_uses_count_conditioning",
    "trainable_policy_state_id",
]
