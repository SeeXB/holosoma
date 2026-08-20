"""Reward manager for computing reward signals."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import torch

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.semantic_keyframes import get_semantic_keyframe_runtime

from .base import RewardTermBase


def _is_semantic_term(term_cfg: RewardTermCfg) -> bool:
    return "semantic" in term_cfg.tags


def compute_base_positive_reward_budget(cfg: RewardManagerCfg) -> float:
    """Sum original positive bounded tracking weights in a semantic preset.

    Semantic presets retain the original WBT term set, whose positive terms are
    normalized exponential tracking rewards. Semantic-tagged terms are excluded
    defensively so they can never inflate the inherited budget.
    """
    budget = sum(
        float(term_cfg.weight)
        for term_cfg in cfg.terms.values()
        if term_cfg.weight > 0.0 and not _is_semantic_term(term_cfg)
    )
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError(f"Base positive reward budget must be finite and positive, got {budget}")
    return budget


@dataclass(frozen=True)
class FixedBudgetAllocation:
    """Per-environment decomposition before the manager's common ``dt`` scale."""

    base_positive: torch.Tensor
    penalty: torch.Tensor
    activity: torch.Tensor
    semantic_quality: torch.Tensor
    alpha: torch.Tensor
    base_contribution: torch.Tensor
    semantic_contribution: torch.Tensor
    positive_total: torch.Tensor
    total: torch.Tensor


def allocate_fixed_positive_budget(
    base_positive: torch.Tensor,
    penalty: torch.Tensor,
    activity: torch.Tensor,
    semantic_quality: torch.Tensor,
    positive_budget: float,
) -> FixedBudgetAllocation:
    """Redistribute, rather than enlarge, the original positive reward budget."""
    if not math.isfinite(positive_budget) or positive_budget <= 0.0:
        raise ValueError("positive_budget must be finite and positive")
    if not (
        base_positive.shape == penalty.shape == activity.shape == semantic_quality.shape
    ):
        raise ValueError("All fixed-budget inputs must have identical shapes")
    if torch.any((activity < 0.0) | (activity > 1.0)):
        raise ValueError("semantic activity must lie in [0, 1]")

    scale = positive_budget / (positive_budget + activity)
    alpha = activity / (positive_budget + activity)
    base_contribution = scale * base_positive
    semantic_contribution = scale * activity * semantic_quality
    positive_total = base_contribution + semantic_contribution
    return FixedBudgetAllocation(
        base_positive=base_positive,
        penalty=penalty,
        activity=activity,
        semantic_quality=semantic_quality,
        alpha=alpha,
        base_contribution=base_contribution,
        semantic_contribution=semantic_contribution,
        positive_total=positive_total,
        total=penalty + positive_total,
    )


class RewardManager:
    """Manages reward computation as a weighted sum of individual terms.

    The reward manager computes the total reward by evaluating each configured
    reward term, multiplying by its weight and the environment's time step (dt),
    and summing the results. It tracks episodic sums for logging and supports
    both stateless (function) and stateful (class) reward terms.

    Parameters
    ----------
    cfg : RewardManagerCfg
        Configuration specifying reward terms and settings.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device where tensors should be allocated.
    """

    def __init__(self, cfg: RewardManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)
        self.latest_metrics: dict[str, torch.Tensor] = {}
        self.base_positive_reward_budget: float | None = None
        if self.cfg.semantic_keyframe is not None and self.cfg.semantic_keyframe.enabled:
            self.base_positive_reward_budget = compute_base_positive_reward_budget(self.cfg)

        # Storage for resolved functions and stateful terms
        self._term_funcs: dict[str, Any] = {}
        self._term_instances: dict[str, RewardTermBase] = {}
        self._term_names: list[str] = []
        self._term_cfgs: list[RewardTermCfg] = []

        # Initialize terms
        self._initialize_terms()

        # Buffers for reward tracking
        self._reward_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Episode sums for each term (for logging)
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._episode_sums_raw: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums_raw[term_name] = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def _initialize_terms(self) -> None:
        """Initialize reward terms and resolve their functions/classes."""
        for term_name, term_cfg in self.cfg.terms.items():
            # Skip terms with zero weight
            if term_cfg.weight == 0.0:
                continue

            # Resolve function or class
            func = self._resolve_function(term_cfg.func)

            # Check if it's a class (stateful) or function (stateless)
            if isinstance(func, type) and issubclass(func, RewardTermBase):
                # Stateful term - instantiate
                instance = func(term_cfg, self.env)
                self._term_instances[term_name] = instance
            else:
                # Stateless function
                self._term_funcs[term_name] = func

            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)

    def _resolve_function(self, func: Any | str) -> Any:
        """Resolve a reward callable or class from a string specification.

        Parameters
        ----------
        func : Any or str
            Function or class reference, or a string like ``"module:object_name"``.

        Returns
        -------
        Any
            Resolved callable or class.

        Raises
        ------
        ValueError
            If the string path is malformed or the target cannot be imported.
        """
        if isinstance(func, str):
            # Parse string like "module.path:function_name"
            if ":" not in func:
                raise ValueError(f"Function string must be in format 'module:function', got: {func}")

            module_path, func_name = func.split(":", 1)
            try:
                module = importlib.import_module(module_path)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import function '{func}': {e}") from e
        return func

    @property
    def active_terms(self) -> list[str]:
        """Names of active reward terms."""
        return self._term_names

    @property
    def episode_sums(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (scaled)."""
        return self._episode_sums

    @property
    def episode_sums_raw(self) -> dict[str, torch.Tensor]:
        """Episodic sums for each reward term (raw, unscaled)."""
        return self._episode_sums_raw

    def compute(self, dt: float) -> torch.Tensor:
        """Compute the total reward as a weighted sum of individual terms.

        Each reward term is evaluated, scaled by its configured weight and the
        environment time step, and accumulated into the total reward. Episodic
        sums are updated for logging purposes.

        Notes
        -----
        Curriculum scaling is handled by directly modifying term weights via
        :meth:`set_term_cfg`, rather than through extra scaling parameters.

        Parameters
        ----------
        dt : float
            Environment time-step interval.

        Returns
        -------
        torch.Tensor
            Net reward tensor with shape ``[num_envs]``.
        """
        if self.cfg.semantic_keyframe is not None and self.cfg.semantic_keyframe.enabled:
            return self._compute_fixed_budget_semantic(dt)

        # Reset computation
        self._reward_buf[:] = 0.0
        self.latest_metrics = {}

        # Iterate over all reward terms
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            # Compute raw reward value
            if term_name in self._term_instances:
                # Stateful term
                instance = self._term_instances[term_name]
                rew_raw = instance(self.env, **term_cfg.params)
            else:
                # Stateless function
                func = self._term_funcs[term_name]
                rew_raw = func(self.env, **term_cfg.params)

            # Validate shape
            if rew_raw.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. "
                    f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )

            # Scale by weight and dt
            rew_scaled = rew_raw * term_cfg.weight * dt

            # Accumulate
            self._reward_buf += rew_scaled

            # Track episodic sums
            self._episode_sums[term_name] += rew_scaled
            self._episode_sums_raw[term_name] += rew_raw

        # Optionally clip to positive
        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)

        return self._reward_buf

    def _compute_fixed_budget_semantic(self, dt: float) -> torch.Tensor:
        """Apply task-agnostic semantic preference inside the inherited budget.

        Negative terms are evaluated and accumulated exactly as in the baseline.
        Only original positive bounded tracking terms receive the common budget
        factor. Consequently ``A=0`` is numerically identical to the baseline,
        while a perfect semantic/base state never exceeds ``W_pos``.
        """
        semantic_cfg = self.cfg.semantic_keyframe
        if semantic_cfg is None or not semantic_cfg.enabled:
            raise RuntimeError("Fixed-budget semantic path requires an enabled semantic config")
        if self.base_positive_reward_budget is None:
            raise RuntimeError("Fixed-budget semantic path has no inherited positive budget")

        evaluated: dict[str, torch.Tensor] = {}
        base_positive = torch.zeros_like(self._reward_buf)
        penalty = torch.zeros_like(self._reward_buf)
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_name in self._term_instances:
                rew_raw = self._term_instances[term_name](self.env, **term_cfg.params)
            else:
                rew_raw = self._term_funcs[term_name](self.env, **term_cfg.params)
            if rew_raw.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. "
                    f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )
            evaluated[term_name] = rew_raw
            self._episode_sums_raw[term_name] += rew_raw
            if _is_semantic_term(term_cfg):
                continue
            weighted = rew_raw * term_cfg.weight
            if term_cfg.weight > 0.0:
                base_positive += weighted
            else:
                penalty += weighted

        runtime_output = get_semantic_keyframe_runtime(self.env, semantic_cfg).evaluate()
        # An empty valid-objective set cannot express a semantic preference. In
        # that degenerate case use zero effective activity and recover baseline.
        effective_activity = runtime_output.active_gate * (runtime_output.valid_objective_count > 0).to(
            runtime_output.active_gate.dtype
        )
        allocation = allocate_fixed_positive_budget(
            base_positive=base_positive,
            penalty=penalty,
            activity=effective_activity,
            semantic_quality=runtime_output.combined_reward,
            positive_budget=self.base_positive_reward_budget,
        )
        common_scale = self.base_positive_reward_budget / (
            self.base_positive_reward_budget + effective_activity
        )
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if _is_semantic_term(term_cfg):
                # Kept only for backward-compatible custom configs. Semantic
                # magnitude is determined by equal valid-objective averaging,
                # never by a RewardTerm weight.
                continue
            actual = evaluated[term_name] * term_cfg.weight
            if term_cfg.weight > 0.0:
                actual = actual * common_scale
            self._episode_sums[term_name] += actual * dt

        self._reward_buf[:] = allocation.total * dt
        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)
        self.latest_metrics = {
            "semantic/activity": allocation.activity,
            "semantic/alpha": allocation.alpha,
            "semantic/part": runtime_output.part_reward,
            "semantic/rel": runtime_output.rel_reward,
            "semantic/dyn": runtime_output.dyn_reward,
            "semantic/combined": runtime_output.combined_reward,
            "semantic/valid_objective_count": runtime_output.valid_objective_count,
            "reward/base_positive": allocation.base_positive,
            "reward/base_contribution": allocation.base_contribution,
            "reward/semantic_contribution": allocation.semantic_contribution,
            "reward/positive_total": allocation.positive_total,
            "reward/penalty": allocation.penalty,
            "reward/total": allocation.total,
        }
        return self._reward_buf

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        """Reset reward tracking and return episodic sums for logging.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Dictionary mirroring the direct reward path structure::

                {
                    "episode": {term_name: tensor_per_reset_env},
                    "episode_all": {term_name: tensor_per_all_envs},
                    "raw_episode": {...},
                    "raw_episode_all": {...},
                }
        """
        extras: dict[str, dict[str, torch.Tensor]] = {
            "episode": {},
            "episode_all": {},
            "raw_episode": {},
            "raw_episode_all": {},
        }

        # Resolve environment ids to operate on
        if env_ids is None:
            env_ids_tensor: torch.Tensor | None = None
            env_ids_slice: slice | torch.Tensor = slice(None)
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

            env_ids_slice = env_ids_tensor

        # Helper to detach values before zeroing internal buffers
        def _clone(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().clone()

        # Populate scaled reward statistics
        for term_name in self._term_names:
            rew_all = self._episode_sums[term_name] / self.env.max_episode_length_s
            extras["episode_all"][f"rew_{term_name}"] = _clone(rew_all)
            if env_ids_tensor is None:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all)
            else:
                extras["episode"][f"rew_{term_name}"] = _clone(rew_all[env_ids_slice])

            # Reset episodic sums for the completed environments
            self._episode_sums[term_name][env_ids_slice] = 0.0

        # Populate raw (unscaled) reward statistics
        for term_name in self._term_names:
            rew_raw_all = self._episode_sums_raw[term_name] / self.env.max_episode_length_s
            extras["raw_episode_all"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            if env_ids_tensor is None:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all)
            else:
                extras["raw_episode"][f"raw_rew_{term_name}"] = _clone(rew_raw_all[env_ids_slice])

            self._episode_sums_raw[term_name][env_ids_slice] = 0.0

        # Reset stateful reward terms
        for instance in self._term_instances.values():
            instance.reset(env_ids=env_ids_tensor)

        return extras

    def get_term(self, name: str) -> Any:
        """Get reward term function or instance by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        Any
            Reward term function or instance.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        if name in self._term_instances:
            return self._term_instances[name]
        if name in self._term_funcs:
            return self._term_funcs[name]
        raise KeyError(f"Reward term '{name}' not found")

    def get_term_cfg(self, name: str) -> RewardTermCfg:
        """Get reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.

        Returns
        -------
        RewardTermCfg
            Configuration for the specified reward term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            return self._term_cfgs[idx]
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def set_term_cfg(self, name: str, cfg: RewardTermCfg) -> None:
        """Set reward term configuration by name.

        Parameters
        ----------
        name : str
            Name of the reward term.
        cfg : RewardTermCfg
            New configuration for the term.

        Raises
        ------
        KeyError
            If the term name is not found.
        """
        try:
            idx = self._term_names.index(name)
            self._term_cfgs[idx] = cfg
        except ValueError:
            raise KeyError(f"Reward term '{name}' not found")

    def __str__(self) -> str:
        """String representation of reward manager."""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
        msg += "Terms:\n"
        for name, cfg in zip(self._term_names, self._term_cfgs):
            msg += f"  - {name}: weight={cfg.weight}\n"
        return msg
