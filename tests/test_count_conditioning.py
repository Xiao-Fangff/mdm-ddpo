from __future__ import annotations

import unittest

import torch
from torch import nn

from mdm_ddpo.count_conditioning import (
    count_conditioning_metadata,
    install_count_conditioning,
    set_count_conditioning_trainable,
)


class ToyTimestep(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, len(timesteps), self.latent_dim)


class ToyMDM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 4
        self.embed_timestep = ToyTimestep(self.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: dict,
    ) -> torch.Tensor:
        del x
        return self.embed_timestep(timesteps)


class CountConditioningTest(unittest.TestCase):
    def test_zero_projection_exactly_preserves_original_output(self):
        model = ToyMDM()
        inputs = torch.zeros(2, 1)
        timesteps = torch.tensor([1, 2])
        baseline = model(inputs, timesteps, {"target_steps": torch.tensor([1, 6])})
        module = install_count_conditioning(model)
        conditioned = model(
            inputs,
            timesteps,
            {"target_steps": torch.tensor([1, 6])},
        )
        torch.testing.assert_close(conditioned, baseline, rtol=0, atol=0)
        self.assertEqual(count_conditioning_metadata(model)["projection_norm"], 0.0)
        self.assertIs(install_count_conditioning(model), module)

    def test_count_changes_output_but_no_count_and_cfg_do_not(self):
        model = ToyMDM()
        module = install_count_conditioning(model)
        with torch.no_grad():
            module.projection.weight.copy_(torch.eye(4))
            module.embedding.weight.zero_()
            module.embedding.weight[2].fill_(0.5)
        inputs = torch.zeros(2, 1)
        timesteps = torch.tensor([1, 2])
        conditioned = model(
            inputs,
            timesteps,
            {"target_steps": torch.tensor([2, -1])},
        )
        torch.testing.assert_close(conditioned[0, 0], torch.full((4,), 0.5))
        torch.testing.assert_close(conditioned[0, 1], torch.zeros(4))
        unconditional = model(
            inputs,
            timesteps,
            {"target_steps": torch.tensor([2, 2]), "uncond": True},
        )
        torch.testing.assert_close(unconditional, torch.zeros_like(unconditional))

    def test_count_parameters_can_be_reenabled_after_lora_freeze(self):
        model = ToyMDM()
        install_count_conditioning(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        count = set_count_conditioning_trainable(model, True)
        self.assertEqual(count, 8 * 4 + 4 * 4)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.count_conditioning.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
