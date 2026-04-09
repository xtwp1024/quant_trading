"""SuiteTrading v2 — Optuna-based Bayesian optimiser for backtesting.

Adapts ``suitetrading.optimization.optuna_optimizer`` for standalone use
within ``quant_trading.backtester`` without requiring the full SuiteTrading
package.

Supports single-objective (TPE) and multi-objective (NSGA-II) search
with optional SQLite persistence and study resume.

Samplers
--------
- **TPE** (Tree-structured Parzen Estimator): default, Bayesian.
- **Random**: uniform random search baseline.
- **NSGA-II**: multi-objective genetic algorithm.
- **CMA-ES**: covariance matrix adaptation evolution strategy.

Pruners
-------
- **Median**: prune trials whose intermediate value exceeds median.
- **Nop**: no pruning (default).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Summary of an Optuna study run."""

    study_name: str
    n_trials: int
    n_completed: int
    n_pruned: int
    best_value: float
    best_params: dict[str, Any]
    best_run_id: str
    wall_time_sec: float
    trials_per_sec: float


# ── Sampler / Pruner registries ──────────────────────────────────────────────

_SAMPLERS: dict[str, type] = {}
_PRUNERS: dict[str, type] = {}

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler, CmaEsSampler
    from optuna.pruners import MedianPruner, NopPruner

    _SAMPLERS = {
        "tpe": TPESampler,
        "random": RandomSampler,
        "nsga2": NSGAIISampler,
        "cmaes": CmaEsSampler,
    }
    _PRUNERS = {
        "median": MedianPruner,
        "none": NopPruner,
    }
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# ── Optimizer ────────────────────────────────────────────────────────────────

class OptunaOptimizer:
    """Wrapper around Optuna study creation, execution and result extraction.

    Parameters
    ----------
    objective
        Callable compatible with ``optuna.Study.optimize()``.
        Receives a trial and returns a scalar value to maximise/minimise.
    study_name
        Unique name for the study (used for persistence/resume).
    storage
        Optuna storage URL.  ``None`` for in-memory, or
        ``"sqlite:///path/to/studies.db"`` for persistent SQLite.
    sampler
        Sampler key: ``"tpe"`` | ``"random"`` | ``"nsga2"`` | ``"cmaes"``.
    pruner
        Pruner key: ``"median"`` | ``"none"``.
    direction
        ``"maximize"`` or ``"minimize"``.
    n_startup_trials
        Random trials before TPE kicks in.
    seed
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        *,
        objective: Callable[..., float],
        study_name: str,
        storage: str | None = None,
        sampler: str = "tpe",
        pruner: str = "none",
        direction: str = "maximize",
        n_startup_trials: int = 20,
        seed: int | None = None,
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise RuntimeError(
                "optuna is not installed.  Install it with: pip install optuna"
            )

        self._objective = objective
        self._study_name = study_name
        self._direction = direction

        import optuna
        sampler_cls = _SAMPLERS.get(sampler)
        if sampler_cls is None:
            raise ValueError(
                f"Unknown sampler {sampler!r}. Available: {sorted(_SAMPLERS)}"
            )
        pruner_cls = _PRUNERS.get(pruner)
        if pruner_cls is None:
            raise ValueError(
                f"Unknown pruner {pruner!r}. Available: {sorted(_PRUNERS)}"
            )

        if sampler == "tpe":
            sampler_obj = sampler_cls(n_startup_trials=n_startup_trials, seed=seed)
        elif sampler == "random":
            sampler_obj = sampler_cls(seed=seed)
        elif sampler == "nsga2":
            sampler_obj = sampler_cls(seed=seed)
        elif sampler == "cmaes":
            sampler_obj = sampler_cls(seed=seed)
        else:
            sampler_obj = sampler_cls()

        pruner_obj = pruner_cls()

        self._study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler_obj,
            pruner=pruner_obj,
            direction=direction,
            load_if_exists=True,
        )

    def optimize(
        self,
        n_trials: int,
        timeout: float | None = None,
        show_progress_bar: bool = False,
    ) -> OptimizationResult:
        """Run the optimisation loop.

        Parameters
        ----------
        n_trials
            Number of trials to execute.
        timeout
            Maximum wall-clock seconds (``None`` for unlimited).
        show_progress_bar
            Show Optuna progress bar (default False).
        """
        import optuna

        t0 = time.perf_counter()

        self._study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )

        elapsed = time.perf_counter() - t0
        completed = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]

        n_completed = len(completed)
        n_pruned = len(pruned)

        if n_completed == 0:
            return OptimizationResult(
                study_name=self._study_name,
                n_trials=len(self._study.trials),
                n_completed=0,
                n_pruned=n_pruned,
                best_value=float("nan"),
                best_params={},
                best_run_id="",
                wall_time_sec=elapsed,
                trials_per_sec=0.0,
            )

        best = self._study.best_trial
        result = OptimizationResult(
            study_name=self._study_name,
            n_trials=len(self._study.trials),
            n_completed=n_completed,
            n_pruned=n_pruned,
            best_value=best.value,
            best_params=best.params,
            best_run_id=best.user_attrs.get("run_id", ""),
            wall_time_sec=elapsed,
            trials_per_sec=n_completed / elapsed if elapsed > 0 else 0.0,
        )
        return result

    def get_top_n(self, n: int = 50) -> list[dict[str, Any]]:
        """Extract the top *n* completed trials sorted by objective value.

        Returns a list of dicts with keys: ``trial_number``, ``value``,
        ``params``, and all user_attrs.
        """
        import optuna

        completed = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        reverse = self._direction == "maximize"
        completed.sort(key=lambda t: t.value, reverse=reverse)

        top: list[dict[str, Any]] = []
        for t in completed[:n]:
            entry: dict[str, Any] = {
                "trial_number": t.number,
                "value": t.value,
                "params": t.params,
            }
            entry.update(t.user_attrs)
            top.append(entry)
        return top

    def get_study(self) -> Any:
        """Access the underlying Optuna study directly."""
        return self._study


# ── Grid-search runner (no Optuna dependency) ────────────────────────────────

def grid_search(
    objective: Callable[..., dict[str, Any]],
    param_grid: dict[str, list[Any]],
    *,
    direction: str = "maximize",
) -> tuple[OptimizationResult, list[dict[str, Any]]]:
    """Brute-force grid search over a parameter grid.

    Unlike ``OptunaOptimizer`` this requires no external dependencies.
    Evaluates every combination in *param_grid* sequentially.

    Parameters
    ----------
    objective
        Callable(params: dict) → dict with at least keys ``value`` (float)
        and optionally ``run_id`` (str) and ``user_attrs`` (dict).
    param_grid
        Mapping ``param_name → list of values`` to iterate.
    direction
        ``"maximize"`` or ``"minimize"``.

    Returns
    -------
    tuple[OptimizationResult, list[dict[str, Any]]]
        Study summary and list of all trial results.
    """
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    best_value = float("-inf") if direction == "maximize" else float("inf")
    best_params: dict[str, Any] = {}
    best_run_id = ""

    for combo in combinations:
        params = dict(zip(keys, combo))
        try:
            res = objective(params)
            value = float(res.get("value", float("-inf") if direction == "maximize" else float("inf")))
            run_id = str(res.get("run_id", ""))
        except Exception as exc:
            value = float("-inf") if direction == "maximize" else float("inf")
            run_id = f"error_{len(results)}"
            res = {"error": str(exc)}

        results.append({
            "trial_number": len(results),
            "value": value,
            "params": params,
            "run_id": run_id,
            **res,
        })

        if direction == "maximize":
            if value > best_value:
                best_value = value
                best_params = params
                best_run_id = run_id
        else:
            if value < best_value:
                best_value = value
                best_params = params
                best_run_id = run_id

    elapsed = time.perf_counter() - t0
    n_completed = len([r for r in results if "error" not in r])

    study_result = OptimizationResult(
        study_name="grid_search",
        n_trials=len(results),
        n_completed=n_completed,
        n_pruned=0,
        best_value=best_value,
        best_params=best_params,
        best_run_id=best_run_id,
        wall_time_sec=elapsed,
        trials_per_sec=len(results) / elapsed if elapsed > 0 else 0.0,
    )

    return study_result, results


# ── Suggest helpers (for building Optuna objectives) ────────────────────────

def suggest_int(trial: Any, name: str, low: int, high: int, step: int = 1) -> int:
    """Suggest an integer parameter."""
    return trial.suggest_int(name, low, high, step=step)


def suggest_float(
    trial: Any,
    name: str,
    low: float,
    high: float,
    *,
    log: bool = False,
) -> float:
    """Suggest a float parameter."""
    return trial.suggest_float(name, low, high, log=log)


def suggest_categorical(trial: Any, name: str, choices: list[Any]) -> Any:
    """Suggest a categorical parameter."""
    return trial.suggest_categorical(name, choices)
