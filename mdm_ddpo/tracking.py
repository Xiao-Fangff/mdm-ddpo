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
    "skipped_updates": "optimization/skipped_updates",
    "rollout_samples": "progress/rollout_samples",
    "unique_prompts": "progress/unique_prompts",
    "samples_per_prompt": "progress/samples_per_prompt",
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
    "eval_best_reward": "eval/best_reward_total",
    "eval_best_reward_delta": "eval/best_reward_total_delta",
    "eval_best_epoch": "eval/best_epoch",
    "eval_evals_without_improvement": "eval/evals_without_improvement",
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
