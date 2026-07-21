#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm_ddpo.config import TrainConfig  # noqa: E402
from mdm_ddpo.diffusion import ddim_step_with_logprob  # noqa: E402
from mdm_ddpo.lora import (  # noqa: E402
    configure_trainable_policy,
    load_trainable_state_dict,
)
from mdm_ddpo.runtime import (  # noqa: E402
    autocast_context,
    bootstrap_external_repositories,
    build_data_loader,
    build_mdm,
    build_model_kwargs,
    build_policy_model,
    diffusion_runtime_metadata,
    load_model_args,
    resolve_device,
    seed_everything,
)
from mdm_ddpo.step_data import (  # noqa: E402
    load_humanml_stats,
    load_step_manifest,
)
from mdm_ddpo.step_probe import (  # noqa: E402
    CounterfactualNumberPool,
    create_counterfactual_pool,
    load_counterfactual_pool,
    render_counterfactual_step_prompt,
    save_counterfactual_pool,
    summarize_counterfactual_counts,
)
from mdm_ddpo.step_reward import HardStepDetector  # noqa: E402


LOGGER = logging.getLogger("probe_step_number_conditioning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether MDM uses number tokens under identical noise, "
            "motion length, and prompt templates."
        )
    )
    parser.add_argument("--output", default="artifacts/counterfactual_number_probe.json")
    parser.add_argument("--pool-path", default="artifacts/counterfactual_number_pool.pt")
    parser.add_argument("--samples-output", default="")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--conditions", type=int, default=24)
    parser.add_argument("--samples-per-condition", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--step-targets", default="1,2,3,4,5,6")
    parser.add_argument("--number-style", choices=["digits", "words"], default="words")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--prompt-seed", type=int, default=20260719)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--ddim-eta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--step-data-manifest", default=TrainConfig.step_data_manifest)
    parser.add_argument("--step-motion-root", default=TrainConfig.step_motion_root)
    parser.add_argument("--step-detector-root", default=TrainConfig.step_detector_root)
    parser.add_argument("--step-min-frames", type=int, default=40)
    parser.add_argument("--step-max-frames", type=int, default=196)
    parser.add_argument("--step-detector-fps", type=int, default=20)
    parser.add_argument("--step-detector-lead-threshold", type=float, default=0.138)
    parser.add_argument("--step-soft-lead-temperature", type=float, default=1.0)
    parser.add_argument("--step-soft-length-temperature", type=float, default=1.0)
    parser.add_argument("--step-soft-progress-temperature", type=float, default=1.0)
    parser.add_argument("--step-soft-cluster-gap-seconds", type=float, default=0.15)
    parser.add_argument("--step-ankle-high-frequency-cutoff-hz", type=float, default=4.0)
    parser.add_argument("--mdm-root", default=TrainConfig.mdm_root)
    parser.add_argument("--motionrft-root", default=TrainConfig.motionrft_root)
    parser.add_argument("--model-path", default=TrainConfig.model_path)
    parser.add_argument("--model-args-path", default=TrainConfig.model_args_path)
    parser.add_argument(
        "--prediction-type",
        choices=["auto", "x_start", "epsilon"],
        default="auto",
    )
    parser.add_argument("--data-cache-dir", default=TrainConfig.data_cache_dir)
    parser.add_argument("--save-motions", action="store_true")
    return parser


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )


def _pool_for_args(args: argparse.Namespace, config: TrainConfig) -> CounterfactualNumberPool:
    path = Path(args.pool_path).expanduser().resolve()
    targets = config.step_target_values
    if path.exists():
        pool = load_counterfactual_pool(path)
    else:
        records = load_step_manifest(
            args.step_data_manifest,
            motion_root=args.step_motion_root,
            targets=targets,
            min_frames=args.step_min_frames,
            max_frames=args.step_max_frames,
        )
        pool = create_counterfactual_pool(
            records,
            targets=targets,
            condition_count=args.conditions,
            samples_per_condition=args.samples_per_condition,
            max_frames=args.step_max_frames,
            seed=args.seed,
            prompt_seed=args.prompt_seed,
            number_style=args.number_style,
        )
        save_counterfactual_pool(pool, path)
        LOGGER.info("Created counterfactual pool: %s", path)
    expected = (
        args.conditions,
        args.samples_per_condition,
        tuple(targets),
        args.step_max_frames,
        args.number_style,
    )
    actual = (
        pool.condition_count,
        pool.samples_per_condition,
        tuple(pool.targets.tolist()),
        pool.max_frames,
        pool.number_style,
    )
    if actual != expected:
        raise ValueError(
            "Existing counterfactual pool does not match requested settings: "
            f"expected={expected}, actual={actual}."
        )
    return pool


def _pooled_text_embeddings(
    model: torch.nn.Module,
    texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = model.encode_text(texts)
    if isinstance(encoded, tuple):
        raw, padding_mask = encoded
        valid = (~padding_mask).T.unsqueeze(-1).to(raw)
        raw_pooled = (raw * valid).sum(dim=0).div(
            valid.sum(dim=0).clamp_min(1.0)
        )
        projected = model.embed_text(raw)
        projected_pooled = (projected * valid).sum(dim=0).div(
            valid.sum(dim=0).clamp_min(1.0)
        )
        return raw_pooled.float(), projected_pooled.float()
    raw = encoded.squeeze(0)
    return raw.float(), model.embed_text(encoded).squeeze(0).float()


def _embedding_metrics(
    model: torch.nn.Module,
    pool: CounterfactualNumberPool,
) -> dict[str, Any]:
    template_slots = sorted(set(pool.template_slots.tolist()))
    texts = [
        render_counterfactual_step_prompt(
            int(target),
            int(template),
            seed=pool.prompt_seed,
            number_style=pool.number_style,
        )
        for template in template_slots
        for target in pool.targets.tolist()
    ]
    with torch.no_grad():
        raw, projected = _pooled_text_embeddings(model, texts)
    shape = (len(template_slots), len(pool.targets), -1)

    def summarize(values: torch.Tensor) -> dict[str, float]:
        values = values.reshape(shape)
        distances: list[torch.Tensor] = []
        cosine_distances: list[torch.Tensor] = []
        for template_values in values:
            for first, second in itertools.combinations(template_values, 2):
                distances.append((first - second).square().mean().sqrt())
                cosine = torch.nn.functional.cosine_similarity(
                    first.unsqueeze(0),
                    second.unsqueeze(0),
                )[0]
                cosine_distances.append(1.0 - cosine)
        distance = torch.stack(distances).float().cpu()
        cosine_distance = torch.stack(cosine_distances).float().cpu()
        return {
            "pairwise_rms_distance_mean": distance.mean().item(),
            "pairwise_rms_distance_min": distance.min().item(),
            "pairwise_cosine_distance_mean": cosine_distance.mean().item(),
            "pairwise_cosine_distance_min": cosine_distance.min().item(),
            "numerically_distinguishable": float(distance.min() > 1.0e-8),
        }

    return {
        "raw_text_encoder": summarize(raw),
        "projected_embed_text": summarize(projected),
    }


def _motion_pair_metrics(
    generated: torch.Tensor,
    pool: CounterfactualNumberPool,
) -> dict[str, float]:
    # generated: [condition, sample, target, frames, features]
    all_distances: list[torch.Tensor] = []
    adjacent_distances: list[torch.Tensor] = []
    for condition in range(pool.condition_count):
        length = int(pool.lengths[condition])
        for sample in range(pool.samples_per_condition):
            values = generated[condition, sample, :, :length].float()
            for first, second in itertools.combinations(range(len(pool.targets)), 2):
                distance = (
                    values[first].sub(values[second]).square().mean().sqrt()
                )
                all_distances.append(distance)
                if second == first + 1:
                    adjacent_distances.append(distance)
    all_values = torch.stack(all_distances)
    adjacent_values = torch.stack(adjacent_distances)
    return {
        "pairwise_motion_rms_mean": all_values.mean().item(),
        "pairwise_motion_rms_median": all_values.median().item(),
        "adjacent_target_motion_rms_mean": adjacent_values.mean().item(),
    }


@torch.no_grad()
def _evaluate_policy(
    *,
    model: torch.nn.Module,
    policy_model: torch.nn.Module,
    diffusion: Any,
    detector: HardStepDetector,
    pool: CounterfactualNumberPool,
    mean: torch.Tensor,
    std: torch.Tensor,
    config: TrainConfig,
    batch_size: int,
    save_motions: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    target_count = len(pool.targets)
    if batch_size <= 0 or batch_size % target_count != 0:
        raise ValueError("Probe batch size must be divisible by target count.")
    units_per_batch = batch_size // target_count
    units = [
        (condition, sample)
        for condition in range(pool.condition_count)
        for sample in range(pool.samples_per_condition)
    ]
    hard_parts: list[torch.Tensor] = []
    soft_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    spacing_parts: list[torch.Tensor] = []
    jitter_parts: list[torch.Tensor] = []
    motion_parts: list[torch.Tensor] = []
    prompt_shape = (263, 1, pool.max_frames)
    targets = pool.targets.tolist()

    for start in range(0, len(units), units_per_batch):
        batch_units = units[start : start + units_per_batch]
        texts: list[str] = []
        length_values: list[int] = []
        generators: list[torch.Generator] = []
        initial_parts: list[torch.Tensor] = []
        for condition, sample in batch_units:
            length = int(pool.lengths[condition])
            template = int(pool.template_slots[condition])
            texts.extend(
                render_counterfactual_step_prompt(
                    int(target),
                    template,
                    seed=pool.prompt_seed,
                    number_style=pool.number_style,
                )
                for target in targets
            )
            length_values.extend([length] * target_count)
            generator = torch.Generator(device=config.device)
            generator.manual_seed(int(pool.noise_seeds[condition, sample]))
            generators.append(generator)
            shared_initial = torch.randn(
                (1, *prompt_shape),
                device=config.device,
                generator=generator,
            )
            initial_parts.append(shared_initial.repeat(target_count, 1, 1, 1))
        lengths = torch.tensor(length_values, dtype=torch.long)
        current = torch.cat(initial_parts, dim=0)
        kwargs = build_model_kwargs(
            model,
            texts,
            lengths,
            pool.max_frames,
            device=torch.device(config.device),
            guidance_scale=config.guidance_scale,
        )
        for step in range(diffusion.num_timesteps - 1, -1, -1):
            timestep = torch.full(
                (len(current),),
                step,
                device=config.device,
                dtype=torch.long,
            )
            transition = torch.cat(
                [
                    torch.randn(
                        (1, *prompt_shape),
                        device=config.device,
                        generator=generator,
                    ).repeat(target_count, 1, 1, 1)
                    for generator in generators
                ],
                dim=0,
            )
            with autocast_context(torch.device(config.device), config.precision):
                current, _, _ = ddim_step_with_logprob(
                    diffusion,
                    policy_model,
                    current,
                    timestep,
                    model_kwargs=kwargs,
                    eta=config.ddim_eta,
                    mask=kwargs["y"]["mask"],
                    clip_denoised=config.clip_denoised,
                    noise=transition,
                )
        generated = current.squeeze(2).permute(0, 2, 1).contiguous()
        detection = detector.detect_normalized(
            generated,
            lengths,
            mean=mean,
            std=std,
        )
        local_shape = (len(batch_units), target_count)
        hard_parts.append(detection.hard_count.cpu().reshape(local_shape))
        soft_parts.append(detection.soft_count.cpu().reshape(local_shape))
        candidate_parts.append(detection.candidate_count.cpu().reshape(local_shape))
        spacing_parts.append(
            detection.candidate_spacing_mean.cpu().reshape(local_shape)
        )
        jitter_parts.append(
            detection.ankle_high_frequency_ratio.cpu().reshape(local_shape)
        )
        motion_parts.append(
            generated.detach().float().cpu().reshape(
                len(batch_units),
                target_count,
                pool.max_frames,
                263,
            )
        )
        LOGGER.info("Probed %d/%d paired noise units", min(start + units_per_batch, len(units)), len(units))

    unit_shape = (
        pool.condition_count,
        pool.samples_per_condition,
        target_count,
    )
    hard = torch.cat(hard_parts).reshape(unit_shape)
    soft = torch.cat(soft_parts).reshape(unit_shape)
    candidates = torch.cat(candidate_parts).reshape(unit_shape)
    spacing = torch.cat(spacing_parts).reshape(unit_shape)
    jitter = torch.cat(jitter_parts).reshape(unit_shape)
    motions = torch.cat(motion_parts).reshape(
        *unit_shape,
        pool.max_frames,
        263,
    )
    summary = summarize_counterfactual_counts(hard, soft, pool)
    summary.update(
        {
            "motion": _motion_pair_metrics(motions, pool),
            "embedding": _embedding_metrics(model, pool),
            "candidate_count_mean": candidates.float().mean().item(),
            "candidate_spacing_mean": spacing.float().mean().item(),
            "ankle_high_frequency_ratio_mean": jitter.float().mean().item(),
        }
    )
    samples = {
        "hard_count": hard,
        "soft_count": soft,
        "candidate_count": candidates,
        "candidate_spacing": spacing,
        "ankle_high_frequency_ratio": jitter,
    }
    if save_motions:
        samples["generated_motion"] = motions
    return summary, samples


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    if not args.include_original and not args.checkpoints:
        raise ValueError(
            "Probe requires --include-original or at least one --checkpoint."
        )
    checkpoint_payloads = [
        (Path(path).expanduser().resolve(), _load_checkpoint(path))
        for path in args.checkpoints
    ]
    config = TrainConfig(
        mdm_root=args.mdm_root,
        motionrft_root=args.motionrft_root,
        model_path=args.model_path,
        model_args_path=args.model_args_path,
        prediction_type=args.prediction_type,
        data_cache_dir=args.data_cache_dir,
        data_workers=0,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        sample_steps=args.sample_steps,
        guidance_scale=args.guidance_scale,
        ddim_eta=args.ddim_eta,
        fixed_eval_every=0,
        step_targets=args.step_targets,
        step_min_frames=args.step_min_frames,
        step_max_frames=args.step_max_frames,
    )
    bootstrap_external_repositories(config)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    data_loader = build_data_loader(config, prompt_batch_size=1)
    model_args = load_model_args(config)
    model, diffusion, _, _ = build_mdm(
        config,
        model_args,
        data_loader,
        device,
    )
    if checkpoint_payloads:
        first_config = checkpoint_payloads[0][1]["config"]
        configure_trainable_policy(
            model,
            mode=str(checkpoint_payloads[0][1]["train_mode"]),
            lora_rank=int(first_config["lora_rank"]),
            lora_alpha=float(first_config["lora_alpha"]),
            lora_target_regex=str(first_config["lora_target_regex"]),
        )
        for path, payload in checkpoint_payloads[1:]:
            current = payload["config"]
            signature = (
                payload["train_mode"],
                current["lora_rank"],
                current["lora_alpha"],
                current["lora_target_regex"],
            )
            expected = (
                checkpoint_payloads[0][1]["train_mode"],
                first_config["lora_rank"],
                first_config["lora_alpha"],
                first_config["lora_target_regex"],
            )
            if signature != expected:
                raise ValueError(f"Checkpoint LoRA settings differ: {path}")
    model.eval()
    policy_model = build_policy_model(model, config.guidance_scale)
    policy_model.eval()
    pool = _pool_for_args(args, config)
    mean_np, std_np = load_humanml_stats(config.mdm_root)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    detector = HardStepDetector(
        backend="progressive",
        fps=args.step_detector_fps,
        motion_rule_root=args.step_detector_root,
        lead_threshold=args.step_detector_lead_threshold,
        soft_lead_temperature=args.step_soft_lead_temperature,
        soft_length_temperature=args.step_soft_length_temperature,
        soft_progress_temperature=args.step_soft_progress_temperature,
        soft_cluster_gap_seconds=args.step_soft_cluster_gap_seconds,
        ankle_high_frequency_cutoff_hz=(
            args.step_ankle_high_frequency_cutoff_hz
        ),
    )
    records: list[dict[str, Any]] = []
    sample_payload: dict[str, Any] = {
        "pool_id": pool.pool_id,
        "targets": pool.targets,
        "lengths": pool.lengths,
        "template_slots": pool.template_slots,
    }

    def evaluate(label: str, checkpoint: Path | None, epoch: int) -> None:
        metrics, samples = _evaluate_policy(
            model=model,
            policy_model=policy_model,
            diffusion=diffusion,
            detector=detector,
            pool=pool,
            mean=mean,
            std=std,
            config=config,
            batch_size=args.batch_size,
            save_motions=args.save_motions,
        )
        records.append(
            {
                "label": label,
                "checkpoint": str(checkpoint) if checkpoint else "",
                "epoch": epoch,
                "pool_id": pool.pool_id,
                "metrics": metrics,
            }
        )
        sample_payload[label] = samples

    if args.include_original:
        evaluate("original_mdm", None, -1)
    for path, payload in checkpoint_payloads:
        load_trainable_state_dict(model, payload["policy"])
        evaluate(path.stem, path, int(payload.get("epoch", -1)))

    baseline = next(
        (record for record in records if record["label"] == "original_mdm"),
        None,
    )
    if baseline is not None:
        baseline_metrics = baseline["metrics"]
        for record in records:
            metrics = record["metrics"]
            record["delta_from_original"] = {
                "hard_mae": metrics["hard_mae"] - baseline_metrics["hard_mae"],
                "hard_exact_fraction": (
                    metrics["hard_exact_fraction"]
                    - baseline_metrics["hard_exact_fraction"]
                ),
                "hard_within_one_fraction": (
                    metrics["hard_within_one_fraction"]
                    - baseline_metrics["hard_within_one_fraction"]
                ),
                "hard_target_count_spearman": (
                    metrics["hard"]["target_count_spearman"]
                    - baseline_metrics["hard"]["target_count_spearman"]
                ),
                "soft_target_count_spearman": (
                    metrics["soft"]["target_count_spearman"]
                    - baseline_metrics["soft"]["target_count_spearman"]
                ),
                "ankle_high_frequency_ratio": (
                    metrics["ankle_high_frequency_ratio_mean"]
                    - baseline_metrics["ankle_high_frequency_ratio_mean"]
                ),
            }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pool_id": pool.pool_id,
                "conditions": pool.condition_count,
                "samples_per_condition": pool.samples_per_condition,
                "targets": pool.targets.tolist(),
                "number_style": pool.number_style,
                "mdm_diffusion": diffusion_runtime_metadata(
                    model_args,
                    diffusion,
                ),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    if args.samples_output:
        samples_output = Path(args.samples_output).expanduser().resolve()
        samples_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample_payload, samples_output)
    LOGGER.info("Saved counterfactual number probe: %s", output)
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
