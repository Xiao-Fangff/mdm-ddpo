from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import torch

from mdm_ddpo.config import TrainConfig
from mdm_ddpo.lora import (
    merge_lora,
)
from mdm_ddpo.policy_io import (
    configure_and_load_policy_checkpoint,
    policy_uses_count_conditioning,
)
from mdm_ddpo.runtime import (
    bootstrap_external_repositories,
    build_mdm,
    diffusion_runtime_metadata,
    load_model_args,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a DDPO policy into a standard MDM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mdm-root", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-args-path", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    allowed_fields = {field.name for field in fields(TrainConfig)}
    saved_config = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in allowed_fields
    }
    config = TrainConfig(**saved_config)
    config.enable_count_conditioning = policy_uses_count_conditioning(checkpoint)
    config.train_count_conditioning = config.enable_count_conditioning
    config.device = args.device
    config.precision = "no"
    config.sample_steps = 0
    if args.mdm_root:
        config.mdm_root = args.mdm_root
    if args.model_path:
        config.model_path = args.model_path
    if args.model_args_path:
        config.model_args_path = args.model_args_path

    for label, path in {
        "MDM root": config.mdm_root,
        "base MDM checkpoint": config.model_path,
        "MDM args": config.model_args_path,
    }.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    bootstrap_external_repositories(config)
    device = resolve_device(config.device)
    model_args = load_model_args(config)
    dummy_data = SimpleNamespace(
        dataset=SimpleNamespace(num_actions=1)
    )
    model, diffusion, _, _ = build_mdm(
        config,
        model_args,
        dummy_data,
        device,
    )
    configure_and_load_policy_checkpoint(
        model,
        checkpoint,
        diffusion_metadata=diffusion_runtime_metadata(model_args, diffusion),
        model_path=config.model_path,
        source="Export checkpoint",
    )
    merged_adapters = merge_lora(model) if config.train_mode == "lora" else 0

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("clip_model.")
        and not name.startswith("count_conditioning.")
    }
    with open(config.model_args_path, "r", encoding="utf-8") as handle:
        model_args_payload = json.load(handle)
    args_output_path = output_path.parent / "args.json"
    if args_output_path.exists() and not args.overwrite:
        with open(args_output_path, "r", encoding="utf-8") as handle:
            existing_args = json.load(handle)
        if existing_args != model_args_payload:
            raise FileExistsError(
                f"Different args.json already exists: {args_output_path}. "
                "Pass --overwrite to replace it."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state_dict,
            "model_avg": state_dict,
        },
        output_path,
    )
    if not args_output_path.exists() or args.overwrite:
        with open(args_output_path, "w", encoding="utf-8") as handle:
            json.dump(model_args_payload, handle, indent=4, sort_keys=True)

    summary = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "args": str(args_output_path),
        "merged_lora_adapters": merged_adapters,
        "count_conditioning_omitted_for_unconditional_count_export": (
            config.enable_count_conditioning
        ),
        "state_tensors": len(state_dict),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
