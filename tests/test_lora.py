from __future__ import annotations

import torch
from torch import nn
import unittest

from mdm_ddpo.lora import (
    LoRAWeight,
    inject_lora,
    load_trainable_state_dict,
    merge_lora,
    set_lora_trainable,
    trainable_state_dict,
)


class ToyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(4, 4)
        self.attn = nn.MultiheadAttention(4, 1, batch_first=True)
        self.clip_model = nn.Linear(4, 4)

    def forward(self, x):
        x = self.block(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        return x


class LoRATest(unittest.TestCase):
    def test_lora_preserves_initial_output_and_only_adapters_train(self):
        torch.manual_seed(3)
        model = ToyNetwork()
        inputs = torch.randn(2, 5, 4)
        before = model(inputs).detach()

        report = inject_lora(
            model,
            rank=2,
            alpha=2,
            target_regex=r"(block|attn)",
        )
        after = model(inputs).detach()

        self.assertEqual(report.adapters, 3)
        torch.testing.assert_close(before, after)
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_names)
        self.assertTrue(all("lora_" in name for name in trainable_names))
        self.assertFalse(
            any(name.startswith("clip_model") for name in trainable_names)
        )
        self.assertEqual(set(trainable_state_dict(model)), set(trainable_names))

        for module in model.modules():
            if isinstance(module, LoRAWeight):
                module.lora_b.data.fill_(0.05)
        changed = model(inputs).detach()
        self.assertFalse(torch.allclose(before, changed))

        merged_count = merge_lora(model)
        merged_output = model(inputs).detach()
        self.assertEqual(merged_count, report.adapters)
        torch.testing.assert_close(changed, merged_output)
        self.assertFalse(
            any("parametrizations" in name for name in model.state_dict())
        )

    def test_loading_requires_every_trainable_lora_tensor(self):
        model = ToyNetwork()
        inject_lora(
            model,
            rank=2,
            alpha=2,
            target_regex=r"(block|attn)",
        )
        state = trainable_state_dict(model)
        missing_name = next(iter(state))
        state.pop(missing_name)

        with self.assertRaisesRegex(KeyError, "missing trainable policy"):
            load_trainable_state_dict(model, state)

    def test_loading_accepts_complete_trainable_lora_state(self):
        source = ToyNetwork()
        target = ToyNetwork()
        for model in (source, target):
            inject_lora(
                model,
                rank=2,
                alpha=2,
                target_regex=r"(block|attn)",
            )
        source_state = trainable_state_dict(source)

        load_trainable_state_dict(target, source_state)

        for name, tensor in source_state.items():
            torch.testing.assert_close(target.state_dict()[name], tensor)

    def test_frozen_lora_remains_in_portable_policy_state(self):
        model = ToyNetwork()
        inject_lora(
            model,
            rank=2,
            alpha=2,
            target_regex=r"(block|attn)",
        )

        frozen_count = set_lora_trainable(model, False)
        state = trainable_state_dict(model)

        self.assertGreater(frozen_count, 0)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(state)
        self.assertTrue(all("lora_" in name for name in state))

        missing_name = next(iter(state))
        state.pop(missing_name)
        with self.assertRaisesRegex(KeyError, "missing trainable policy"):
            load_trainable_state_dict(model, state)


if __name__ == "__main__":
    unittest.main()
