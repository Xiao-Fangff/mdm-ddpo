from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRAWeight(nn.Module):
    """Low-rank additive parametrization for a two-dimensional weight."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        alpha: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        delta = torch.matmul(self.lora_b, self.lora_a) * self.scale
        return base_weight + delta


@dataclass(frozen=True)
class LoRAReport:
    adapters: int
    trainable_parameters: int
    target_modules: tuple[str, ...]


def _register_weight_lora(
    module: nn.Module,
    parameter_name: str,
    rank: int,
    alpha: float,
) -> bool:
    weight = getattr(module, parameter_name, None)
    if weight is None or weight.ndim != 2:
        return False
    if parametrize.is_parametrized(module, parameter_name):
        return False
    adapter = LoRAWeight(
        out_features=weight.shape[0],
        in_features=weight.shape[1],
        rank=rank,
        alpha=alpha,
        device=weight.device,
        dtype=weight.dtype,
    )
    parametrize.register_parametrization(module, parameter_name, adapter)
    getattr(module.parametrizations, parameter_name).original.requires_grad_(False)
    return True


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    target_regex: str,
) -> LoRAReport:
    """Freeze the policy and inject LoRA into matching Linear/MHA weights."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    pattern = re.compile(target_regex)
    targets: list[str] = []
    # Snapshot before parametrization adds its own child modules.
    modules = list(model.named_modules())
    for name, module in modules:
        if not name or name.startswith("clip_model.") or not pattern.search(name):
            continue
        if isinstance(module, nn.MultiheadAttention):
            if _register_weight_lora(module, "in_proj_weight", rank, alpha):
                targets.append(name + ".in_proj_weight")
        if isinstance(module, nn.Linear):
            if _register_weight_lora(module, "weight", rank, alpha):
                targets.append(name + ".weight")

    if not targets:
        raise ValueError(
            f"LoRA target regex matched no supported MDM weights: {target_regex!r}"
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return LoRAReport(len(targets), trainable, tuple(targets))


def configure_trainable_policy(
    model: nn.Module,
    *,
    mode: str,
    lora_rank: int,
    lora_alpha: float,
    lora_target_regex: str,
) -> LoRAReport | None:
    if mode == "lora":
        return inject_lora(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_regex=lora_target_regex,
        )
    if mode != "full":
        raise ValueError(f"Unsupported train mode: {mode}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("clip_model."))
    return None


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    known = set(model.state_dict())
    unexpected = sorted(set(state_dict) - known)
    if unexpected:
        raise KeyError(f"Checkpoint has unknown policy tensors: {unexpected[:8]}")
    model.load_state_dict(state_dict, strict=False)


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def merge_lora(model: nn.Module) -> int:
    """Bake all registered LoRA parametrizations into ordinary weights."""

    merged = 0
    for module in list(model.modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for parameter_name in list(parametrizations.keys()):
            parametrize.remove_parametrizations(
                module,
                parameter_name,
                leave_parametrized=True,
            )
            merged += 1
    return merged
