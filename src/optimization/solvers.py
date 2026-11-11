"""Constrained portfolio weight solvers with explicit infeasibility reporting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize


class OptimizationError(RuntimeError):
    """Raised when no feasible portfolio solution can be verified."""


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    objective_value: float
    solver: str
    selected_assets: tuple[int, ...]
    diagnostics: dict[str, Any]


def _validate_inputs(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    minimum_weight: float,
    maximum_weight: float,
    maximum_holdings: int | None,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(expected_returns, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if mean.ndim != 1 or cov.shape != (len(mean), len(mean)):
        raise ValueError("Expected returns and covariance dimensions do not align.")
    if not len(mean) or not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError("Expected returns and covariance must be finite and non-empty.")
    if minimum_weight < 0 or maximum_weight <= 0 or minimum_weight > maximum_weight:
        raise ValueError("Invalid asset weight bounds.")
    if maximum_holdings is not None and maximum_holdings < 1:
        raise ValueError("maximum_holdings must be positive.")
    if not np.allclose(cov, cov.T, atol=tolerance, rtol=0.0):
        raise ValueError("Covariance matrix must be symmetric.")
    if np.linalg.eigvalsh(cov).min() < -tolerance:
        raise ValueError("Covariance matrix must be positive semidefinite.")
    return mean, (cov + cov.T) / 2.0


def optimize_weights(
    method: str,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    *,
    historical_returns: np.ndarray | None = None,
    annual_risk_free_rate: float = 0.0,
    cvar_confidence: float = 0.95,
    minimum_weight: float = 0.0,
    maximum_weight: float = 1.0,
    maximum_holdings: int | None = None,
    tolerance: float = 1e-8,
) -> OptimizationResult:
    """Solve minimum variance, maximum Sharpe, or historical CVaR portfolios."""
    mean, cov = _validate_inputs(
        expected_returns,
        covariance,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        maximum_holdings=maximum_holdings,
        tolerance=tolerance,
    )
    if method not in {"minimum_variance", "maximum_sharpe", "cvar_minimizing"}:
        raise ValueError(f"Unknown optimization method {method!r}.")
    if annual_risk_free_rate <= -1.0 or not np.isfinite(annual_risk_free_rate):
        raise ValueError("annual_risk_free_rate must be finite and greater than -1.")
    if not 0.0 < cvar_confidence < 1.0:
        raise ValueError("cvar_confidence must be between zero and one.")
    scenarios = None
    if method == "cvar_minimizing":
        if historical_returns is None:
            raise ValueError("Historical returns are required for CVaR optimization.")
        scenarios = np.asarray(historical_returns, dtype=np.float64)
        if scenarios.ndim != 2 or scenarios.shape[1] != len(mean):
            raise ValueError("Historical return scenarios do not match assets.")
        if len(scenarios) < 2 or not np.isfinite(scenarios).all():
            raise ValueError("CVaR scenarios must contain finite observations.")

    asset_count = len(mean)
    if maximum_holdings is None:
        candidates = (
            [tuple(range(asset_count))]
            if asset_count * minimum_weight - tolerance
            <= 1.0
            <= asset_count * maximum_weight + tolerance
            else []
        )
    else:
        limit = min(maximum_holdings, asset_count)
        candidates = [
            selected
            for count in range(1, limit + 1)
            if count * minimum_weight - tolerance
            <= 1.0
            <= count * maximum_weight + tolerance
            for selected in combinations(range(asset_count), count)
        ]
    failures: list[dict[str, Any]] = []
    successful: list[OptimizationResult] = []
    for selected in candidates:
        subset_mean = mean[list(selected)]
        subset_cov = cov[np.ix_(selected, selected)]
        bounds = [(minimum_weight, maximum_weight)] * len(selected)
        if method == "cvar_minimizing":
            assert scenarios is not None
            subset_returns = scenarios[:, list(selected)]
            result = _solve_cvar(
                subset_returns,
                confidence=cvar_confidence,
                bounds=bounds,
            )
        else:
            result = _solve_smooth(
                method,
                subset_mean,
                subset_cov,
                bounds=bounds,
                annual_risk_free_rate=annual_risk_free_rate,
                tolerance=tolerance,
            )
        if result is None:
            failures.append({"selected_assets": list(selected), "status": "solver_failed"})
            continue
        subset_weights, objective, solver, message = result
        if (
            not np.isfinite(subset_weights).all()
            or abs(float(subset_weights.sum()) - 1.0) > max(tolerance * 10, 1e-7)
            or (subset_weights < minimum_weight - tolerance).any()
            or (subset_weights > maximum_weight + tolerance).any()
        ):
            failures.append(
                {
                    "selected_assets": list(selected),
                    "status": "solution_failed_verification",
                    "weights": subset_weights.tolist(),
                    "solver_message": message,
                }
            )
            continue
        full_weights = np.zeros(asset_count, dtype=np.float64)
        full_weights[list(selected)] = subset_weights
        successful.append(
            OptimizationResult(
                full_weights,
                float(objective),
                solver,
                tuple(selected),
                {
                    "solver_message": message,
                    "candidate_subset_count": len(candidates),
                    "failed_candidate_count": len(failures),
                    "candidate_failures": failures,
                },
            )
        )
    if not successful:
        if not candidates:
            reason = (
                "No holdings subset satisfies the sum-to-one and weight bounds."
            )
        else:
            reason = f"No feasible solver result among {len(candidates)} candidates."
        raise OptimizationError(f"{reason} Details: {failures}")
    best = min(successful, key=lambda result: result.objective_value)
    return OptimizationResult(
        best.weights,
        best.objective_value,
        best.solver,
        best.selected_assets,
        {
            **best.diagnostics,
            "successful_candidate_count": len(successful),
            "selected_assets": list(best.selected_assets),
        },
    )


def _solve_smooth(
    method: str,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    bounds: list[tuple[float, float]],
    annual_risk_free_rate: float,
    tolerance: float,
) -> tuple[np.ndarray, float, str, str] | None:
    count = len(mean)
    lower = np.array([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.array([bound[1] for bound in bounds], dtype=np.float64)
    remaining = 1.0 - lower.sum()
    capacity = upper - lower
    initial = (
        lower.copy()
        if capacity.sum() <= tolerance
        else lower + remaining * capacity / capacity.sum()
    )
    constraints = ({"type": "eq", "fun": lambda weights: weights.sum() - 1.0},)
    if method == "minimum_variance":
        objective = lambda weights: float(weights @ covariance @ weights)
    else:
        objective = lambda weights: -float(
            (weights @ mean - annual_risk_free_rate)
            / max(float(np.sqrt(weights @ covariance @ weights)), 1e-12)
        )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tolerance, "maxiter": 1000, "disp": False},
    )
    if not result.success or result.x is None:
        return None
    return np.asarray(result.x), float(result.fun), "SLSQP", str(result.message)


def _solve_cvar(
    scenarios: np.ndarray,
    *,
    confidence: float,
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, float, str, str] | None:
    scenario_count, asset_count = scenarios.shape
    variable_count = asset_count + 1 + scenario_count
    objective = np.concatenate(
        (
            np.zeros(asset_count),
            [1.0],
            np.full(scenario_count, 1.0 / ((1.0 - confidence) * scenario_count)),
        )
    )
    inequalities = np.zeros((scenario_count, variable_count), dtype=np.float64)
    inequalities[:, :asset_count] = -scenarios
    inequalities[:, asset_count] = -1.0
    inequalities[:, asset_count + 1 :] = -np.eye(scenario_count)
    equalities = np.zeros((1, variable_count), dtype=np.float64)
    equalities[0, :asset_count] = 1.0
    variable_bounds = [
        *bounds,
        (None, None),
        *((0.0, None) for _ in range(scenario_count)),
    ]
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(scenario_count),
        A_eq=equalities,
        b_eq=np.ones(1),
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        return None
    return (
        np.asarray(result.x[:asset_count]),
        float(result.fun),
        "HiGHS linear programming",
        str(result.message),
    )
