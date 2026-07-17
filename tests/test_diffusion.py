from __future__ import annotations

import numpy as np
import torch
from torch import nn
import unittest

from mdm_ddpo.diffusion import ddim_step_with_logprob


class ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, timesteps, **kwargs):
        del timesteps, kwargs
        return self.scale * x


class ToyDiffusion:
    def __init__(self) -> None:
        self.alphas_cumprod = np.asarray([0.82, 0.55, 0.25], dtype=np.float64)
        self.alphas_cumprod_prev = np.asarray([1.0, 0.82, 0.55], dtype=np.float64)

    def p_mean_variance(
        self,
        model,
        sample,
        timestep,
        clip_denoised,
        model_kwargs,
    ):
        del clip_denoised
        prediction = model(sample, timestep, **model_kwargs)
        return {"pred_xstart": prediction}

    def _predict_eps_from_xstart(self, x_t, timestep, pred_xstart):
        alpha = torch.as_tensor(
            self.alphas_cumprod,
            dtype=x_t.dtype,
            device=x_t.device,
        )[timestep].reshape(-1, 1, 1, 1)
        return (x_t / alpha.sqrt() - pred_xstart) / (1.0 / alpha - 1.0).sqrt()


class DiffusionLogProbTest(unittest.TestCase):
    def test_sample_and_recomputed_logprob_match_and_have_gradient(self):
        torch.manual_seed(0)
        diffusion = ToyDiffusion()
        policy = ToyPolicy()
        sample = torch.randn(2, 3, 1, 4)
        timestep = torch.full((2,), 2, dtype=torch.long)
        mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)

        previous, sampled_log_prob, _ = ddim_step_with_logprob(
            diffusion,
            policy,
            sample,
            timestep,
            model_kwargs={},
            eta=1.0,
            mask=mask,
        )
        _, recomputed_log_prob, _ = ddim_step_with_logprob(
            diffusion,
            policy,
            sample,
            timestep,
            model_kwargs={},
            eta=1.0,
            prev_sample=previous,
            mask=mask,
        )

        torch.testing.assert_close(sampled_log_prob, recomputed_log_prob)
        recomputed_log_prob.sum().backward()
        self.assertIsNotNone(policy.scale.grad)
        self.assertGreater(policy.scale.grad.abs().item(), 0)

    def test_padding_is_excluded_from_logprob(self):
        torch.manual_seed(1)
        diffusion = ToyDiffusion()
        policy = ToyPolicy()
        sample = torch.randn(1, 2, 1, 4)
        timestep = torch.tensor([1])
        mask = torch.tensor([[[[True, True, False, False]]]])

        previous, _, _ = ddim_step_with_logprob(
            diffusion,
            policy,
            sample,
            timestep,
            model_kwargs={},
            eta=1.0,
            mask=mask,
        )
        changed_padding = previous.clone()
        changed_padding[..., 2:] += 1000
        _, reference, _ = ddim_step_with_logprob(
            diffusion,
            policy,
            sample,
            timestep,
            model_kwargs={},
            eta=1.0,
            prev_sample=previous,
            mask=mask,
        )
        _, changed, _ = ddim_step_with_logprob(
            diffusion,
            policy,
            sample,
            timestep,
            model_kwargs={},
            eta=1.0,
            prev_sample=changed_padding,
            mask=mask,
        )
        torch.testing.assert_close(reference, changed)

    def test_timestep_zero_is_deterministic_and_has_zero_logprob(self):
        diffusion = ToyDiffusion()
        policy = ToyPolicy()
        sample = torch.randn(2, 2, 1, 3)
        timestep = torch.zeros(2, dtype=torch.long)

        first, log_prob, pred_xstart = ddim_step_with_logprob(
            diffusion,
            policy,
            sample,
            timestep,
            model_kwargs={},
            eta=1.0,
        )
        torch.testing.assert_close(first, pred_xstart)
        torch.testing.assert_close(log_prob, torch.zeros_like(log_prob))


if __name__ == "__main__":
    unittest.main()
