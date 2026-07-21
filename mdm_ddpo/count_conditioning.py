from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


COUNT_CONDITIONING_VERSION = 1
DEFAULT_COUNT_CLASSES = 8
DEFAULT_NO_COUNT_ID = 7


class ExplicitCountConditioning(nn.Module):
    """Zero-impact count embedding for targets 0..6 and no-count id 7."""

    def __init__(
        self,
        latent_dim: int,
        *,
        num_embeddings: int = DEFAULT_COUNT_CLASSES,
        no_count_id: int = DEFAULT_NO_COUNT_ID,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("Count conditioning latent dimension must be positive.")
        if num_embeddings <= 1:
            raise ValueError("Count conditioning needs at least two embeddings.")
        if no_count_id < 0 or no_count_id >= num_embeddings:
            raise ValueError("No-count id must be inside the embedding table.")
        self.latent_dim = int(latent_dim)
        self.num_embeddings = int(num_embeddings)
        self.no_count_id = int(no_count_id)
        self.embedding = nn.Embedding(self.num_embeddings, self.latent_dim)
        self.projection = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Exact zero initialization preserves every original MDM output.
        nn.init.zeros_(self.projection.weight)

    def count_ids(self, target_steps: torch.Tensor) -> torch.Tensor:
        targets = torch.as_tensor(target_steps, dtype=torch.long)
        valid = (targets >= 0) & (targets < self.no_count_id)
        return torch.where(
            valid,
            targets,
            torch.full_like(targets, self.no_count_id),
        )

    def forward(
        self,
        target_steps: torch.Tensor,
        *,
        force_mask: bool = False,
    ) -> torch.Tensor:
        targets = torch.as_tensor(
            target_steps,
            device=self.embedding.weight.device,
            dtype=torch.long,
        ).reshape(-1)
        ids = self.count_ids(targets).to(self.embedding.weight.device)
        values = self.projection(self.embedding(ids))
        active = ids != self.no_count_id
        if force_mask:
            active = torch.zeros_like(active)
        return values * active.unsqueeze(-1).to(values)


@dataclass(frozen=True)
class CountConditioningInstallation:
    module: ExplicitCountConditioning
    pre_hook: Any
    timestep_hook: Any


def install_count_conditioning(
    model: nn.Module,
    *,
    num_embeddings: int = DEFAULT_COUNT_CLASSES,
    no_count_id: int = DEFAULT_NO_COUNT_ID,
) -> ExplicitCountConditioning:
    """Attach count conditioning to an MDM instance without changing its source."""

    existing = getattr(model, "count_conditioning", None)
    if isinstance(existing, ExplicitCountConditioning):
        if (
            existing.num_embeddings != num_embeddings
            or existing.no_count_id != no_count_id
        ):
            raise ValueError("Existing count conditioning settings do not match.")
        return existing
    if existing is not None:
        raise TypeError("Model already has an incompatible count_conditioning module.")
    if not hasattr(model, "embed_timestep") or not hasattr(model, "latent_dim"):
        raise TypeError("Explicit count conditioning requires an MDM-like model.")

    conditioning = ExplicitCountConditioning(
        int(model.latent_dim),
        num_embeddings=num_embeddings,
        no_count_id=no_count_id,
    )
    model.add_module("count_conditioning", conditioning)
    model._mdm_ddpo_count_context = None

    def capture_context(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del module
        y = kwargs.get("y")
        if y is None and len(args) >= 3:
            y = args[2]
        if not isinstance(y, dict):
            model._mdm_ddpo_count_context = None
            return
        targets = y.get("target_steps")
        model._mdm_ddpo_count_context = (
            targets,
            bool(y.get("uncond", False)),
        )

    def add_count_embedding(
        module: nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, args
        context = getattr(model, "_mdm_ddpo_count_context", None)
        if context is None or context[0] is None:
            return output
        targets, force_mask = context
        addition = conditioning(targets, force_mask=force_mask).to(
            device=output.device,
            dtype=output.dtype,
        )
        if output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError("MDM timestep embedding has an unexpected shape.")
        if addition.shape != (output.shape[1], output.shape[2]):
            raise RuntimeError(
                "Count target batch does not match the MDM timestep embedding."
            )
        return output + addition.unsqueeze(0)

    pre_hook = model.register_forward_pre_hook(capture_context, with_kwargs=True)
    timestep_hook = model.embed_timestep.register_forward_hook(add_count_embedding)
    model._mdm_ddpo_count_installation = CountConditioningInstallation(
        module=conditioning,
        pre_hook=pre_hook,
        timestep_hook=timestep_hook,
    )
    return conditioning


def count_conditioning_metadata(model: nn.Module) -> dict[str, Any]:
    module = getattr(model, "count_conditioning", None)
    if not isinstance(module, ExplicitCountConditioning):
        return {"enabled": False, "version": COUNT_CONDITIONING_VERSION}
    return {
        "enabled": True,
        "version": COUNT_CONDITIONING_VERSION,
        "num_embeddings": module.num_embeddings,
        "no_count_id": module.no_count_id,
        "latent_dim": module.latent_dim,
        "embedding_norm": module.embedding.weight.detach().float().norm().item(),
        "projection_norm": module.projection.weight.detach().float().norm().item(),
    }


def count_conditioning_signature(model: nn.Module) -> dict[str, Any]:
    metadata = count_conditioning_metadata(model)
    metadata.pop("embedding_norm", None)
    metadata.pop("projection_norm", None)
    return metadata


def validate_count_conditioning_signature(
    model: nn.Module,
    expected: dict[str, Any] | None,
    *,
    source: str,
) -> None:
    """Reject policy artifacts built with a different count adapter."""

    actual = count_conditioning_signature(model)
    legacy_disabled = {
        "enabled": False,
        "version": COUNT_CONDITIONING_VERSION,
    }
    resolved = legacy_disabled if expected is None else dict(expected)
    if resolved != actual:
        raise ValueError(
            f"{source} count-conditioning configuration does not match the "
            f"current policy: expected={resolved}, actual={actual}."
        )


def set_count_conditioning_trainable(model: nn.Module, trainable: bool) -> int:
    module = getattr(model, "count_conditioning", None)
    if not isinstance(module, ExplicitCountConditioning):
        if trainable:
            raise RuntimeError("Cannot train count conditioning before installation.")
        return 0
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "COUNT_CONDITIONING_VERSION",
    "DEFAULT_COUNT_CLASSES",
    "DEFAULT_NO_COUNT_ID",
    "ExplicitCountConditioning",
    "count_conditioning_metadata",
    "count_conditioning_signature",
    "install_count_conditioning",
    "set_count_conditioning_trainable",
    "validate_count_conditioning_signature",
]
