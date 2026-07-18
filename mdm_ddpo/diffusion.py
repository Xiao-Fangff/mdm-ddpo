from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


LOG_TWO_PI = math.log(2.0 * math.pi)


def _extract(values: np.ndarray, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=timesteps.device, dtype=torch.float32)
    result = tensor[timesteps.long()]
    return result.reshape(result.shape + (1,) * (len(shape) - result.ndim))


def _masked_mean_per_sample(
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    reduce_dims = tuple(range(1, values.ndim))
    if mask is None:
        return values.mean(dim=reduce_dims)
    weights = mask.to(device=values.device, dtype=values.dtype).expand_as(values)
    numerator = (values * weights).sum(dim=reduce_dims)
    denominator = weights.sum(dim=reduce_dims).clamp_min(1.0)
    return numerator / denominator


def ddim_step_with_logprob(
    diffusion: Any,
    model: torch.nn.Module,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    *,
    model_kwargs: dict[str, Any],
    eta: float,
    prev_sample: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    clip_denoised: bool = False,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
    min_std: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take one stochastic DDIM step and evaluate its transition log-probability.

    MDM predicts x_0, so the implementation first asks its GaussianDiffusion
    object for pred_xstart and then applies DDIM equation (12).  The returned
    log probability is averaged over valid motion features/frames, matching
    ddpo-pytorch's dimension-normalized likelihood while excluding padding.

    The t=0 transition is deterministic and therefore contributes zero policy
    log-probability.  It is still executed to produce the final clean motion.
    """

    if eta <= 0:
        raise ValueError("eta must be > 0 for a stochastic DDPO policy.")
    if timestep.ndim != 1 or timestep.shape[0] != sample.shape[0]:
        raise ValueError("timestep must have shape [batch].")

    out = diffusion.p_mean_variance(
        model,
        sample,
        timestep,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
    )
    pred_xstart = out["pred_xstart"]
    eps = diffusion._predict_eps_from_xstart(sample, timestep, pred_xstart)

    alpha_bar = _extract(diffusion.alphas_cumprod, timestep, sample.shape)
    alpha_bar_prev = _extract(diffusion.alphas_cumprod_prev, timestep, sample.shape)
    sigma_sq = (
        eta**2
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar).clamp_min(1.0e-12)
        * (1.0 - alpha_bar / alpha_bar_prev.clamp_min(1.0e-12))
    ).clamp_min(0.0)
    sigma = sigma_sq.sqrt()
    direction_scale = (1.0 - alpha_bar_prev - sigma_sq).clamp_min(0.0).sqrt()
    mean = (
        pred_xstart * alpha_bar_prev.sqrt() + direction_scale * eps
    ).contiguous()

    active = (timestep != 0).to(sample.dtype).reshape(
        sample.shape[0], *([1] * (sample.ndim - 1))
    )
    std = sigma.to(sample.dtype) * active

    if prev_sample is None:
        if noise is None:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=generator,
            )
        elif noise.shape != sample.shape:
            raise ValueError("noise must have the same shape as sample.")
        else:
            noise = noise.to(device=sample.device, dtype=sample.dtype)
        prev_sample = (mean + std * noise).contiguous()
    elif prev_sample.shape != sample.shape:
        raise ValueError("prev_sample must have the same shape as sample.")
    else:
        prev_sample = prev_sample.contiguous()

    # Evaluate in float32 even under autocast; gradients still flow to mean.
    mean_fp32 = mean.float()
    prev_fp32 = prev_sample.detach().float()
    safe_std = std.float().clamp_min(min_std)
    elementwise_log_prob = (
        -0.5 * ((prev_fp32 - mean_fp32) / safe_std).square()
        - safe_std.log()
        - 0.5 * LOG_TWO_PI
    )
    log_prob = _masked_mean_per_sample(elementwise_log_prob, mask)
    log_prob = torch.where(timestep != 0, log_prob, torch.zeros_like(log_prob))

    return prev_sample.to(sample.dtype), log_prob, pred_xstart
