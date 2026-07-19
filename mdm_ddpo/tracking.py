from __future__ import annotations

import importlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from .config import TrainConfig


LOGGER = logging.getLogger(__name__)


# These names are accepted by scripts/train_humanml.sh as convenience aliases,
# but SwanLab 0.9 also treats SWANLAB_PROJECT and SWANLAB_RUN_NAME as nested
# Pydantic settings that must contain JSON. The tracker passes the corresponding
# values explicitly to swanlab.init(), so hide the aliases while SwanLab builds
# its settings. Credential variables such as SWANLAB_API_KEY remain untouched.
_SWANLAB_INIT_ENV_ALIASES = (
    "SWANLAB_PROJECT",
    "SWANLAB_RUN_NAME",
    "SWANLAB_WORKSPACE",
    "SWANLAB_MODE",
    "SWANLAB_LOG_DIR",
)


@contextmanager
def _without_swanlab_init_aliases() -> Iterator[None]:
    saved = {
        name: os.environ[name]
        for name in _SWANLAB_INIT_ENV_ALIASES
        if name in os.environ
    }
    for name in saved:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.update(saved)


TRAINING_METRIC_NAMES = {
    "reward": "reward/total",
    "reward_std": "reward/std",
    "reward_retrieval": "reward/retrieval",
    "reward_m2m": "reward/m2m",
    "reward_step": "reward/step",
    "step_exact_fraction": "step/exact_fraction",
    "step_within_one_fraction": "step/within_one_fraction",
    "step_mae": "step/mae",
    "step_detected_mean": "step/detected_mean",
    "step_target_mean": "step/target_mean",
    "step_soft_count_mean": "step/soft_count_mean",
    "step_soft_count_error_mean": "step/soft_count_error_mean",
    "step_soft_count_mae": "step/soft_count_mae",
    "step_soft_hard_count_difference_mean": (
        "step/soft_hard_count_difference_mean"
    ),
    "step_raw_candidate_count_mean": "step/raw_candidate_count_mean",
    "step_candidate_count_mean": "step/candidate_count_mean",
    "step_candidate_spacing_mean": "step/candidate_spacing_mean",
    "step_candidate_spacing_min_mean": "step/candidate_spacing_min_mean",
    "step_ankle_high_frequency_ratio": (
        "step/ankle_high_frequency_ratio"
    ),
    "step_reward_candidate_count_correlation": (
        "step/reward_candidate_count_correlation"
    ),
    "step_reward_ankle_high_frequency_correlation": (
        "step/reward_ankle_high_frequency_correlation"
    ),
    "step_rollout_samples": "step/rollout_samples",
    "step_use_m2m_reward": "step/use_m2m_reward",
    "step_target_1_samples": "step/target_1_samples",
    "step_target_2_samples": "step/target_2_samples",
    "step_target_3_samples": "step/target_3_samples",
    "step_target_4_samples": "step/target_4_samples",
    "step_target_5_samples": "step/target_5_samples",
    "step_target_6_samples": "step/target_6_samples",
    "step_target_1_prompt_groups": "step/target_1_prompt_groups",
    "step_target_2_prompt_groups": "step/target_2_prompt_groups",
    "step_target_3_prompt_groups": "step/target_3_prompt_groups",
    "step_target_4_prompt_groups": "step/target_4_prompt_groups",
    "step_target_5_prompt_groups": "step/target_5_prompt_groups",
    "step_target_6_prompt_groups": "step/target_6_prompt_groups",
    "reward_within_prompt_std": "reward/within_prompt_std",
    "reward_between_prompt_std": "reward/between_prompt_std",
    "reward_centered_std": "reward/centered_std",
    "reward_group_std_min": "reward/group_std_min",
    "reward_group_std_median": "reward/group_std_median",
    "reward_group_std_max": "reward/group_std_max",
    "potential_group_whiten_scale_max": (
        "reward/potential_group_whiten_scale_max"
    ),
    "advantage_std_floor": "advantage/std_floor",
    "effective_shrink_scale_max": "advantage/effective_shrink_scale_max",
    "zero_variance_prompt_fraction": "reward/zero_variance_prompt_fraction",
    "component_advantage_correlation": "advantage/component_correlation",
    "component_advantage_conflict_fraction": (
        "advantage/component_conflict_fraction"
    ),
    "component_advantage_retrieval_weight": (
        "advantage/retrieval_weight"
    ),
    "component_advantage_m2m_weight": "advantage/m2m_weight",
    "component_advantage_step_m2m_enabled": (
        "advantage/step_m2m_enabled"
    ),
    "component_advantage_step_retrieval_weight": (
        "advantage/step_retrieval_weight"
    ),
    "component_advantage_step_m2m_weight": (
        "advantage/step_m2m_weight"
    ),
    "component_advantage_step_retrieval_contribution_mean_abs": (
        "advantage/step_retrieval_contribution_mean_abs"
    ),
    "component_advantage_step_m2m_contribution_mean_abs": (
        "advantage/step_m2m_contribution_mean_abs"
    ),
    "component_advantage_retrieval_std_floor": (
        "advantage/retrieval_std_floor"
    ),
    "component_advantage_m2m_std_floor": "advantage/m2m_std_floor",
    "component_advantage_retrieval_std": "advantage/retrieval_std",
    "component_advantage_m2m_std": "advantage/m2m_std",
    "component_advantage_retrieval_contribution_mean_abs": (
        "advantage/retrieval_contribution_mean_abs"
    ),
    "component_advantage_m2m_contribution_mean_abs": (
        "advantage/m2m_contribution_mean_abs"
    ),
    "component_advantage_retrieval_group_std_median": (
        "advantage/retrieval_group_std_median"
    ),
    "component_advantage_m2m_group_std_median": (
        "advantage/m2m_group_std_median"
    ),
    "component_advantage_retrieval_effective_scale_max": (
        "advantage/retrieval_effective_scale_max"
    ),
    "component_advantage_m2m_effective_scale_max": (
        "advantage/m2m_effective_scale_max"
    ),
    "component_advantage_step_weight": "advantage/step_weight",
    "component_advantage_step_std_floor": "advantage/step_std_floor",
    "component_advantage_step_std": "advantage/step_std",
    "component_advantage_step_contribution_mean_abs": (
        "advantage/step_contribution_mean_abs"
    ),
    "component_advantage_step_group_std_median": (
        "advantage/step_group_std_median"
    ),
    "component_advantage_step_zero_variance_prompt_fraction": (
        "advantage/step_zero_variance_prompt_fraction"
    ),
    "component_advantage_step_effective_scale_max": (
        "advantage/step_effective_scale_max"
    ),
    "component_advantage_retrieval_step_correlation": (
        "advantage/retrieval_step_correlation"
    ),
    "component_advantage_m2m_step_correlation": (
        "advantage/m2m_step_correlation"
    ),
    "component_advantage_retrieval_step_conflict_fraction": (
        "advantage/retrieval_step_conflict_fraction"
    ),
    "component_advantage_m2m_step_conflict_fraction": (
        "advantage/m2m_step_conflict_fraction"
    ),
    "component_advantage_step_samples": "advantage/step_samples",
    "component_advantage_step_unique_reward_levels_mean": (
        "advantage/step_unique_reward_levels_mean"
    ),
    "component_advantage_step_unique_reward_levels_median": (
        "advantage/step_unique_reward_levels_median"
    ),
    "component_advantage_step_unique_reward_levels_min": (
        "advantage/step_unique_reward_levels_min"
    ),
    "component_advantage_step_pairwise_reward_tie_fraction": (
        "advantage/step_pairwise_reward_tie_fraction"
    ),
    "component_advantage_step_nonzero_advantage_sample_fraction": (
        "advantage/step_nonzero_advantage_sample_fraction"
    ),
    "component_advantage_step_top1_advantage_concentration": (
        "advantage/step_top1_advantage_concentration"
    ),
    "advantage_mean": "ppo/advantage_mean",
    "advantage_std": "ppo/advantage_std",
    "loss": "ppo/loss",
    "approx_kl": "ppo/approx_kl",
    "clip_fraction": "ppo/clip_fraction",
    "ratio": "ppo/ratio",
    "ratio_std": "ppo/ratio_std",
    "log_ratio_mean": "ppo/log_ratio_mean",
    "log_ratio_std": "ppo/log_ratio_std",
    "log_ratio_max": "ppo/log_ratio_abs_max",
    "initial_log_prob_abs_diff_mean": "audit/log_prob_abs_diff_mean",
    "initial_log_prob_abs_diff_max": "audit/log_prob_abs_diff_max",
    "initial_log_ratio_mean": "audit/log_ratio_mean",
    "initial_log_ratio_std": "audit/log_ratio_std",
    "initial_log_ratio_max": "audit/log_ratio_abs_max",
    "initial_ratio_mean": "audit/ratio_mean",
    "initial_ratio_std": "audit/ratio_std",
    "initial_ratio_abs_deviation_max": "audit/ratio_abs_deviation_max",
    "grad_norm": "optimization/grad_norm",
    "lora_norm": "optimization/lora_norm",
    "update_norm": "optimization/update_norm",
    "anchor_loss": "anchor/loss",
    "anchor_weighted_loss": "anchor/weighted_loss",
    "anchor_grad_norm": "anchor/grad_norm",
    "anchor_weighted_grad_norm": "anchor/weighted_grad_norm",
    "ppo_grad_norm": "anchor/ppo_grad_norm",
    "anchor_grad_ratio": "anchor/grad_ratio",
    "anchor_lambda": "anchor/lambda",
    "anchor_batch_samples": "anchor/batch_samples",
    "anchor_calls": "anchor/calls",
    "skipped_updates": "optimization/skipped_updates",
    "rollout_samples": "progress/rollout_samples",
    "unique_prompts": "progress/unique_prompts",
    "samples_per_prompt": "progress/samples_per_prompt",
    "humanml_samples_per_prompt": "progress/humanml_samples_per_prompt",
    "step_samples_per_prompt": "progress/step_samples_per_prompt",
    "humanml_rollout_samples": "progress/humanml_rollout_samples",
    "step_motion_ratio": "progress/step_motion_ratio",
    "epoch": "progress/epoch",
    "global_step": "progress/global_step",
    "elapsed_seconds": "time/epoch_seconds",
    "eval_reward": "eval/reward_total",
    "eval_reward_std": "eval/reward_std",
    "eval_reward_median": "eval/reward_total_median",
    "eval_reward_bootstrap_se": "eval/reward_total_bootstrap_se",
    "eval_reward_retrieval": "eval/reward_retrieval",
    "eval_reward_retrieval_median": "eval/reward_retrieval_median",
    "eval_reward_retrieval_bootstrap_se": (
        "eval/reward_retrieval_bootstrap_se"
    ),
    "eval_reward_m2m": "eval/reward_m2m",
    "eval_reward_m2m_median": "eval/reward_m2m_median",
    "eval_reward_m2m_bootstrap_se": "eval/reward_m2m_bootstrap_se",
    "eval_reward_delta": "eval/reward_total_delta",
    "eval_reward_delta_median": "eval/reward_total_delta_median",
    "eval_reward_improvement_fraction": (
        "eval/reward_total_improvement_fraction"
    ),
    "eval_reward_delta_bootstrap_se": "eval/reward_total_delta_bootstrap_se",
    "eval_reward_retrieval_delta": "eval/reward_retrieval_delta",
    "eval_reward_retrieval_delta_median": (
        "eval/reward_retrieval_delta_median"
    ),
    "eval_reward_retrieval_improvement_fraction": (
        "eval/reward_retrieval_improvement_fraction"
    ),
    "eval_reward_retrieval_delta_bootstrap_se": (
        "eval/reward_retrieval_delta_bootstrap_se"
    ),
    "eval_reward_m2m_delta": "eval/reward_m2m_delta",
    "eval_reward_m2m_delta_median": "eval/reward_m2m_delta_median",
    "eval_reward_m2m_improvement_fraction": (
        "eval/reward_m2m_improvement_fraction"
    ),
    "eval_reward_m2m_delta_bootstrap_se": (
        "eval/reward_m2m_delta_bootstrap_se"
    ),
    "eval_normalized_retrieval_delta": (
        "eval/normalized_retrieval_delta"
    ),
    "eval_normalized_m2m_delta": "eval/normalized_m2m_delta",
    "eval_balanced_score": "eval/balanced_score",
    "eval_balanced_score_median": "eval/balanced_score_median",
    "eval_balanced_score_bootstrap_se": (
        "eval/balanced_score_bootstrap_se"
    ),
    "eval_feasible": "eval/feasible",
    "eval_retrieval_tolerance": "eval/retrieval_tolerance",
    "eval_m2m_tolerance": "eval/m2m_tolerance",
    "eval_effective_min_delta": "eval/effective_min_delta",
    "eval_reward_baseline": "eval/reward_total_baseline",
    "eval_reward_retrieval_baseline": "eval/reward_retrieval_baseline",
    "eval_reward_m2m_baseline": "eval/reward_m2m_baseline",
    "eval_samples": "eval/samples",
    "eval_prompts": "eval/prompts",
    "eval_samples_per_prompt": "eval/samples_per_prompt",
    "eval_seed": "eval/seed",
    "eval_batch_size": "eval/batch_size",
    "eval_diffusion_steps": "eval/diffusion_steps",
    "eval_is_best": "eval/is_best",
    "eval_is_best_balanced": "eval/is_best_balanced",
    "eval_is_best_retrieval": "eval/is_best_retrieval",
    "eval_is_best_m2m": "eval/is_best_m2m",
    "eval_is_best_step": "step_eval/is_best_step",
    "eval_step_acceptance": "step_eval/acceptance",
    "eval_step_acceptance_score": "step_eval/acceptance_score",
    "eval_is_best_step_acceptance": (
        "step_eval/is_best_acceptance"
    ),
    "eval_best_step_acceptance_score": (
        "step_eval/best_acceptance_score"
    ),
    "eval_best_step_acceptance_epoch": (
        "step_eval/best_acceptance_epoch"
    ),
    "eval_best_balanced_score": "eval/best_balanced_score",
    "eval_best_balanced_epoch": "eval/best_balanced_epoch",
    "eval_best_retrieval_delta": "eval/best_retrieval_delta",
    "eval_best_retrieval_epoch": "eval/best_retrieval_epoch",
    "eval_best_m2m_delta": "eval/best_m2m_delta",
    "eval_best_m2m_epoch": "eval/best_m2m_epoch",
    "eval_best_step_delta": "step_eval/best_reward_delta",
    "eval_best_step_epoch": "step_eval/best_epoch",
    "eval_best_reward": "eval/best_reward_total",
    "eval_best_reward_delta": "eval/best_reward_total_delta",
    "eval_best_epoch": "eval/best_epoch",
    "eval_evals_without_improvement": "eval/evals_without_improvement",
    "step_eval_reward_std": "step_eval/reward_std",
    "step_eval_detected_mean": "step_eval/detected_mean",
    "step_eval_target_mean": "step_eval/target_mean",
    "step_eval_soft_count_mean": "step_eval/soft_count_mean",
    "step_eval_soft_count_error_mean": "step_eval/soft_count_error_mean",
    "step_eval_soft_count_mae": "step_eval/soft_count_mae",
    "step_eval_soft_hard_count_difference_mean": (
        "step_eval/soft_hard_count_difference_mean"
    ),
    "step_eval_candidate_count_mean": "step_eval/candidate_count_mean",
    "step_eval_candidate_spacing_mean": "step_eval/candidate_spacing_mean",
    "step_eval_ankle_high_frequency_ratio": (
        "step_eval/ankle_high_frequency_ratio"
    ),
    "step_eval_samples": "step_eval/samples",
    "step_eval_prompts": "step_eval/prompts",
    "step_eval_samples_per_prompt": "step_eval/samples_per_prompt",
    "step_eval_use_m2m_reward": "step_eval/use_m2m_reward",
    "step_eval_seed": "step_eval/seed",
    "eval_step_total": "step_eval/total",
    "eval_step_total_delta": "step_eval/total_delta",
    "eval_step_retrieval": "step_eval/retrieval",
    "eval_step_retrieval_delta": "step_eval/retrieval_delta",
    "eval_step_m2m": "step_eval/m2m",
    "eval_step_m2m_delta": "step_eval/m2m_delta",
    "eval_step_reward": "step_eval/reward",
    "eval_step_reward_baseline": "step_eval/reward_baseline",
    "eval_step_reward_delta": "step_eval/reward_delta",
    "eval_step_reward_delta_median": "step_eval/reward_delta_median",
    "eval_step_reward_improvement_fraction": (
        "step_eval/reward_improvement_fraction"
    ),
    "eval_step_reward_delta_bootstrap_se": (
        "step_eval/reward_delta_bootstrap_se"
    ),
    "eval_step_exact_fraction": "step_eval/exact_fraction",
    "eval_step_exact_fraction_delta": "step_eval/exact_fraction_delta",
    "eval_step_within_one_fraction": "step_eval/within_one_fraction",
    "eval_step_within_one_fraction_delta": (
        "step_eval/within_one_fraction_delta"
    ),
    "eval_step_mae": "step_eval/mae",
    "eval_step_mae_baseline": "step_eval/mae_baseline",
    "eval_step_mae_delta": "step_eval/mae_delta",
    "eval_step_mae_improvement_fraction": (
        "step_eval/mae_improvement_fraction"
    ),
    "eval_step_detected_mean": "step_eval/detected_mean_per_prompt",
    "eval_step_detected_mean_delta": "step_eval/detected_mean_delta",
    "eval_step_soft_count": "step_eval/soft_count",
    "eval_step_soft_count_delta": "step_eval/soft_count_delta",
    "eval_step_soft_error_mean": "step_eval/soft_error_mean",
    "eval_step_soft_error_mean_delta": "step_eval/soft_error_mean_delta",
    "eval_step_soft_mae": "step_eval/soft_mae",
    "eval_step_soft_mae_delta": "step_eval/soft_mae_delta",
    "eval_step_candidate_count_delta": "step_eval/candidate_count_delta",
    "eval_step_candidate_spacing_delta": (
        "step_eval/candidate_spacing_delta"
    ),
    "eval_step_ankle_high_frequency_ratio_delta": (
        "step_eval/ankle_high_frequency_ratio_delta"
    ),
    "eval_normalized_step_delta": "step_eval/normalized_reward_delta",
}


def format_training_metrics(
    record: Mapping[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    """Map the JSONL epoch record to grouped SwanLab curve names."""
    metrics = {
        target: record[source]
        for source, target in TRAINING_METRIC_NAMES.items()
        if source in record
    }
    metrics["optimization/learning_rate"] = learning_rate
    return metrics


def _as_scalar(value: Any) -> int | float | bool | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (bool, int, float)):
        return value
    return None


class SwanLabTracker:
    """Optional, lazily imported SwanLab run scoped to one training call."""

    def __init__(
        self,
        config: TrainConfig,
        output_dir: Path,
        swanlab_module: Any | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self._swanlab = swanlab_module
        self._run: Any | None = None
        self._finished = False

    @property
    def enabled(self) -> bool:
        return self.config.use_swanlab

    def start(self) -> "SwanLabTracker":
        if not self.enabled or self._run is not None:
            return self

        log_dir = (
            Path(self.config.swanlab_log_dir).expanduser().resolve()
            if self.config.swanlab_log_dir
            else self.output_dir / "swanlab"
        )
        init_kwargs: dict[str, Any] = {
            "project": self.config.swanlab_project,
            "mode": self.config.swanlab_mode,
            "log_dir": str(log_dir),
            "config": self.config.to_dict(),
        }
        if self.config.swanlab_run_name:
            init_kwargs["name"] = self.config.swanlab_run_name
        if self.config.swanlab_workspace:
            init_kwargs["workspace"] = self.config.swanlab_workspace

        with _without_swanlab_init_aliases():
            if self._swanlab is None:
                try:
                    self._swanlab = importlib.import_module("swanlab")
                except ImportError as exc:
                    raise RuntimeError(
                        "SwanLab logging was enabled, but the 'swanlab' package "
                        "is not installed. Install the project requirements or "
                        "run 'pip install swanlab>=0.9'."
                    ) from exc
            self._run = self._swanlab.init(**init_kwargs)
        if self._run is None or not callable(getattr(self._run, "log", None)):
            raise RuntimeError("swanlab.init() did not return a usable run object.")
        LOGGER.info(
            "SwanLab tracking enabled: project=%s, mode=%s, log_dir=%s",
            self.config.swanlab_project,
            self.config.swanlab_mode,
            log_dir,
        )
        return self

    def log(self, values: Mapping[str, Any], step: int) -> None:
        if self._run is None:
            return
        scalars = {
            name: scalar
            for name, value in values.items()
            if (scalar := _as_scalar(value)) is not None
        }
        if scalars:
            self._run.log(scalars, step=step)

    def finish(
        self,
        state: str = "success",
        error: str | None = None,
    ) -> None:
        if self._run is None or self._finished:
            return
        self._finished = True
        try:
            self._swanlab.finish(state=state, error=error)
        except Exception:
            LOGGER.exception("Failed to finish the SwanLab run cleanly.")

    def __enter__(self) -> "SwanLabTracker":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        if exc_type is None:
            self.finish(state="success")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.finish(state="aborted", error=str(exc_value))
        else:
            self.finish(state="crashed", error=str(exc_value))
        return False
