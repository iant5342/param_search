"""Core solver and scoring backend for the endogenous-Gamma bang-bang search."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

PRIMITIVES_TO_VARY: List[str] = [
    "a", "Delta", "lam", "theta", "k", "omega_bar"
]
PARAMETER_ORDER: List[str] = [
    "theta", "a", "Delta", "omega_bar", "k", "lam"
]
PRIMITIVE_IMPORTANCE: Dict[str, float] = {
    "a": 3.0,
    "Delta": 3.0,
    "lam": 3.0,
    "theta": 2.0,
    "k": 2.0,
    "omega_bar": 1.0,
}

E_LOWER, E_UPPER = 0.0, 10.0
SW_LOWER, SW_UPPER = 0.0, 1.0 - 1e-6
NEG_INF = -1e12


def validate_params(params: Dict[str, float]) -> bool:
    """Return whether all required parameters are present and admissible."""
    required = {"theta", "a", "Delta", "omega_bar", "k", "lam"}
    return required.issubset(params) and (
        params["theta"] > 1.0
        and params["a"] > 0.0
        and params["Delta"] > 0.0
        and params["omega_bar"] > 0.0
        and 0.0 < params["k"] < 1.0
        and params["lam"] >= 0.0
    )


def cost(e: float, a: float) -> float:
    """Manager effort cost: c(e) = a e^2 / 2."""
    return 0.5 * a * e**2


def output_with_workers(e: float, Delta: float, lam: float) -> float:
    """Total output when workers remain."""
    return e + Delta + lam * e * Delta


def worker_increment(e: float, Delta: float, lam: float) -> float:
    """Incremental output created by retaining workers."""
    return Delta + lam * e * Delta


def F(Gamma: float, omega_bar: float) -> float:
    """Uniform outside-option CDF on [0, omega_bar]."""
    return Gamma / omega_bar


def Pi(e: float, Gamma: float, params: Dict[str, float]) -> float:
    """Expected surplus extracted from workers."""
    Delta, lam = params["Delta"], params["lam"]
    return (worker_increment(e, Delta, lam) - Gamma) * F(
        Gamma, params["omega_bar"]
    )


def Gamma_of_contract(
    e: float, psi: float, sW: float, params: Dict[str, float]
) -> float:
    """Worker expected compensation under a contract."""
    return psi + sW * (
        output_with_workers(e, params["Delta"], params["lam"]) - psi
    )


def is_feasible_contract_outcome(
    e: float,
    psi: float,
    sW: float,
    params: Dict[str, float],
    tol: float = 1e-8,
) -> bool:
    """Check contract and compensation feasibility."""
    Gamma_e = Gamma_of_contract(e, psi, sW, params)
    return (
        psi >= -tol
        and -tol <= sW <= 1.0 + tol
        and -tol <= Gamma_e <= params["omega_bar"] + tol
    )


def _manager_objective(
    e: float,
    psi: float,
    sW: float,
    params: Dict[str, float],
    coefficient: float,
) -> float:
    Gamma_e = Gamma_of_contract(e, psi, sW, params)
    if not -1e-8 <= Gamma_e <= params["omega_bar"] + 1e-8:
        return NEG_INF
    return coefficient * (e + Pi(e, Gamma_e, params)) - cost(e, params["a"])


def manager_stage3_objective(
    e: float, psi: float, sW: float, params: Dict[str, float]
) -> float:
    """Manager's stage-3 effort objective."""
    return _manager_objective(e, psi, sW, params, params["k"])


def manager_exante_objective(
    e: float, psi: float, sW: float, params: Dict[str, float]
) -> float:
    """Manager's ex-ante payoff component."""
    theta, k = params["theta"], params["k"]
    coefficient = theta + (1.0 - theta) * k
    return _manager_objective(e, psi, sW, params, coefficient)


def maximize_scalar_on_bounds(
    objective, lower: float, upper: float
) -> Tuple[float, float]:
    """Robustly maximize a scalar objective on [lower, upper]."""
    if not np.isfinite(lower) or not np.isfinite(upper):
        return np.nan, NEG_INF
    if upper <= lower:
        return lower, float(objective(lower))

    result = minimize_scalar(
        lambda x: -objective(x),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-8},
    )
    candidates = [(lower, objective(lower)), (upper, objective(upper))]
    if result.success and np.isfinite(result.x):
        x = float(result.x)
        candidates.append((x, objective(x)))
    x_best, val_best = max(candidates, key=lambda pair: pair[1])
    return float(x_best), float(val_best)


def solve_effort(psi: float, sW: float, params: Dict[str, float]) -> float:
    """Solve the manager's stage-3 effort choice for a contract."""
    effort_upper = E_UPPER
    if sW > 1e-12:
        Delta, lam = params["Delta"], params["lam"]
        denominator = sW * (1.0 + lam * Delta)
        if denominator > 0.0:
            feasible = (
                params["omega_bar"] - psi - sW * (Delta - psi)
            ) / denominator
            effort_upper = min(E_UPPER, feasible)

    if effort_upper < E_LOWER:
        return np.nan

    objective = lambda e: manager_stage3_objective(e, psi, sW, params)
    e_star, val_star = maximize_scalar_on_bounds(
        objective, E_LOWER, effort_upper
    )
    return np.nan if val_star <= NEG_INF / 10 else float(e_star)


def _contract_result(
    psi: float, sW: float, e: float, Gamma: float, payoff: float
) -> Dict[str, float]:
    return {"psi": psi, "sW": sW, "e": e, "Gamma": Gamma, "payoff": payoff}


def evaluate_contract(
    psi: float, sW: float, params: Dict[str, float]
) -> Dict[str, float]:
    """Solve effort and evaluate ex-ante payoff for a contract."""
    e_star = solve_effort(psi, sW, params)
    if not np.isfinite(e_star):
        return _contract_result(psi, sW, np.nan, np.nan, NEG_INF)

    Gamma_star = Gamma_of_contract(e_star, psi, sW, params)
    if not is_feasible_contract_outcome(e_star, psi, sW, params):
        return _contract_result(psi, sW, e_star, Gamma_star, NEG_INF)

    payoff = manager_exante_objective(e_star, psi, sW, params)
    return _contract_result(psi, sW, e_star, Gamma_star, payoff)


def solve_pure_equity(params: Dict[str, float]) -> Dict[str, float]:
    """Solve the pure-equity regime: psi = 0 and choose sW."""
    objective = lambda sW: evaluate_contract(0.0, sW, params)["payoff"]
    sW_star, _ = maximize_scalar_on_bounds(objective, SW_LOWER, SW_UPPER)
    return evaluate_contract(0.0, sW_star, params)


def solve_fixed_wage(params: Dict[str, float]) -> Dict[str, float]:
    """Solve the fixed-wage regime: sW = 0 and choose psi."""
    objective = lambda psi: evaluate_contract(psi, 0.0, params)["payoff"]
    psi_star, _ = maximize_scalar_on_bounds(
        objective, 0.0, params["omega_bar"]
    )
    return evaluate_contract(psi_star, 0.0, params)


def solve_option_B(params: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Solve both restricted regimes and select the better one."""
    if not validate_params(params):
        bad = _contract_result(np.nan, np.nan, np.nan, np.nan, NEG_INF)
        return {
            "equity": bad.copy(),
            "fixed": bad.copy(),
            "selected": {**bad, "regime": "Invalid"},
        }

    equity, fixed = solve_pure_equity(params), solve_fixed_wage(params)
    if equity["payoff"] >= fixed["payoff"]:
        selected = {**equity, "regime": "Equity"}
    else:
        selected = {**fixed, "regime": "Fixed wage"}
    return {"equity": equity, "fixed": fixed, "selected": selected}


def make_grid(
    primitive: str,
    grid_size: int,
    grid_ranges: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Construct a one-dimensional comparative-static grid."""
    if primitive not in grid_ranges:
        raise KeyError(f"No grid range specified for primitive: {primitive}")
    return np.linspace(*grid_ranges[primitive], grid_size)


def run_comparative_static(
    param_name: str,
    grid: Sequence[float],
    baseline: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Vary one primitive while holding all others fixed."""
    records: List[Dict[str, Any]] = []
    for x in grid:
        params = {**baseline, param_name: float(x)}
        result = solve_option_B(params)
        selected, equity, fixed = (
            result["selected"],
            result["equity"],
            result["fixed"],
        )
        record: Dict[str, Any] = {
            "x": float(x),
            "selected_regime": selected["regime"],
        }
        for key in ("sW", "psi", "Gamma", "e", "payoff"):
            record[f"selected_{key}"] = selected[key]
        for regime, values in (("equity", equity), ("fixed", fixed)):
            for key in ("payoff", "sW", "psi", "Gamma", "e"):
                record[f"{regime}_{key}"] = values[key]
        records.append(record)
    return records


def records_to_arrays(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Convert comparative-static records into NumPy arrays."""
    numeric_keys = [
        "x",
        "selected_sW",
        "selected_psi",
        "selected_Gamma",
        "selected_e",
        "selected_payoff",
        "equity_payoff",
        "fixed_payoff",
        "equity_sW",
        "equity_psi",
        "equity_Gamma",
        "equity_e",
        "fixed_sW",
        "fixed_psi",
        "fixed_Gamma",
        "fixed_e",
    ]
    arrays = {
        key: np.array([record[key] for record in records], dtype=float)
        for key in numeric_keys
    }
    arrays["selected_regime"] = np.array(
        [record["selected_regime"] for record in records], dtype=object
    )
    return arrays


def _safe_median_abs(values: np.ndarray, default: float = 1.0) -> float:
    """Compute a safe median absolute value for normalization."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    median = float(np.nanmedian(np.abs(finite)))
    return median if np.isfinite(median) else default


def find_payoff_gap_crossings(
    records: Sequence[Dict[str, Any]],
    omega_bar_for_normalization: Optional[float] = None,
    boundary_fraction: float = 0.10,
) -> List[Dict[str, float]]:
    """Find payoff-gap sign changes and calculate crossing diagnostics."""
    if len(records) < 2:
        return []

    arrays = records_to_arrays(records)
    x = arrays["x"]
    gap = arrays["equity_payoff"] - arrays["fixed_payoff"]
    selected_sW = arrays["selected_sW"]
    selected_psi = arrays["selected_psi"]
    selected_Gamma = arrays["selected_Gamma"]
    selected_e = arrays["selected_e"]
    selected_payoff = arrays["selected_payoff"]

    n = len(records)
    boundary_n = max(1, math.ceil(boundary_fraction * n))
    median_e = _safe_median_abs(selected_e)
    payoff_scale = 1.0 + _safe_median_abs(selected_payoff)

    if omega_bar_for_normalization is None or not np.isfinite(
        omega_bar_for_normalization
    ):
        omega_scale = max(1e-8, _safe_median_abs(selected_Gamma))
    else:
        omega_scale = max(1e-8, float(omega_bar_for_normalization))

    crossings: List[Dict[str, float]] = []
    for i in range(n - 1):
        g0, g1 = float(gap[i]), float(gap[i + 1])
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue
        if not (
            (g0 == 0.0 and g1 != 0.0)
            or (g1 == 0.0 and g0 != 0.0)
            or g0 * g1 < 0.0
        ):
            continue

        x0, x1 = float(x[i]), float(x[i + 1])
        dx = abs(x1 - x0)
        if dx <= 0.0 or not np.isfinite(dx):
            continue

        jump_sW = abs(float(selected_sW[i + 1]) - float(selected_sW[i]))
        jump_psi = abs(float(selected_psi[i + 1]) - float(selected_psi[i]))
        jump_Gamma = abs(
            float(selected_Gamma[i + 1]) - float(selected_Gamma[i])
        )
        jump_e = abs(float(selected_e[i + 1]) - float(selected_e[i]))
        contract_jump = (
            jump_sW
            + jump_psi / omega_scale
            + jump_Gamma / omega_scale
            + jump_e / (1.0 + median_e)
        )
        raw_slope = abs(g1 - g0) / dx
        gap_change = abs(g1 - g0) / payoff_scale
        interior = float(
            i >= boundary_n and i + 1 <= n - boundary_n - 1
        )
        crossings.append(
            {
                "index_left": float(i),
                "x_left": x0,
                "x_right": x1,
                "x_mid": 0.5 * (x0 + x1),
                "gap_left": g0,
                "gap_right": g1,
                "jump_sW": jump_sW,
                "jump_psi": jump_psi,
                "jump_Gamma": jump_Gamma,
                "jump_e": jump_e,
                "normalized_contract_jump": float(contract_jump),
                "raw_gap_slope": float(raw_slope),
                "normalized_gap_slope": float(raw_slope / payoff_scale),
                "normalized_gap_change": float(gap_change),
                "interior": interior,
            }
        )
    return crossings


def score_comparative_static(
    records: Sequence[Dict[str, Any]],
    baseline: Optional[Dict[str, float]] = None,
    jump_threshold: float = 0.05,
    clean_gap_change_threshold: float = 1e-5,
    boundary_fraction: float = 0.10,
) -> Dict[str, Any]:
    """Return the unweighted score and diagnostics for one primitive."""
    omega_bar = None if baseline is None else baseline.get("omega_bar")
    crossings = find_payoff_gap_crossings(
        records, omega_bar, boundary_fraction
    )
    num_crossings = len(crossings)
    if not num_crossings:
        return {
            "base_score": 0.0,
            "num_crossings": 0,
            "has_crossing": 0,
            "has_meaningful_jump": 0,
            "interior_crossing": 0,
            "clean_gap_slope": 0,
            "extra_crossings": 0,
            "max_jump": 0.0,
            "max_gap_slope": 0.0,
            "max_gap_change": 0.0,
            "best_crossing_x": np.nan,
            "best_crossing_index_left": np.nan,
        }

    jump_key = "normalized_contract_jump"
    best = max(crossings, key=lambda crossing: crossing[jump_key])
    max_jump = max(crossing[jump_key] for crossing in crossings)
    max_gap_slope = max(
        crossing["normalized_gap_slope"] for crossing in crossings
    )
    max_gap_change = max(
        crossing["normalized_gap_change"] for crossing in crossings
    )
    has_meaningful_jump = int(max_jump > jump_threshold)
    interior_crossing = int(
        any(crossing["interior"] >= 0.5 for crossing in crossings)
    )
    clean_gap_slope = int(max_gap_change > clean_gap_change_threshold)
    extra_crossings = max(0, num_crossings - 1)
    base_score = (
        10
        + 5 * has_meaningful_jump
        + 5 * interior_crossing
        + 2 * clean_gap_slope
        - 5 * extra_crossings
    )
    return {
        "base_score": float(base_score),
        "num_crossings": num_crossings,
        "has_crossing": 1,
        "has_meaningful_jump": has_meaningful_jump,
        "interior_crossing": interior_crossing,
        "clean_gap_slope": clean_gap_slope,
        "extra_crossings": extra_crossings,
        "max_jump": float(max_jump),
        "max_gap_slope": float(max_gap_slope),
        "max_gap_change": float(max_gap_change),
        "best_crossing_x": float(best["x_mid"]),
        "best_crossing_index_left": int(best["index_left"]),
    }


def score_parameter_vector(
    baseline: Dict[str, float],
    primitives_to_vary: Sequence[str] = PRIMITIVES_TO_VARY,
    grid_size: int = 50,
    grid_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    jump_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Score a baseline parameter vector across all requested primitives."""
    if grid_ranges is None:
        raise ValueError(
            "grid_ranges must be supplied to score_parameter_vector()."
        )

    row: Dict[str, Any] = {
        parameter: float(baseline[parameter]) for parameter in PARAMETER_ORDER
    }
    total_score = 0.0
    crossings_total = 0
    primitives_with_crossing = 0
    primitives_with_jump = 0

    for primitive in primitives_to_vary:
        if primitive not in PRIMITIVE_IMPORTANCE:
            raise KeyError(
                f"Missing importance weight for primitive: {primitive}"
            )

        weight = float(PRIMITIVE_IMPORTANCE[primitive])
        records = run_comparative_static(
            primitive,
            make_grid(primitive, grid_size, grid_ranges),
            baseline,
        )
        diag = score_comparative_static(
            records, baseline, jump_threshold=jump_threshold
        )
        base_score = float(diag["base_score"])
        weighted_score = base_score * weight
        total_score += weighted_score
        crossings_total += int(diag["num_crossings"])
        primitives_with_crossing += int(diag["has_crossing"])
        primitives_with_jump += int(diag["has_meaningful_jump"])

        row[f"{primitive}_base_score"] = base_score
        row[f"{primitive}_importance_weight"] = weight
        row[f"{primitive}_weighted_score"] = weighted_score
        row.update(
            {
                f"{primitive}_{key}": value
                for key, value in diag.items()
                if key != "base_score"
            }
        )

    row.update(
        {
            "total_score": float(total_score),
            "num_crossings_total": crossings_total,
            "num_primitives_with_crossing": primitives_with_crossing,
            "num_primitives_with_meaningful_jump": primitives_with_jump,
        }
    )
    return row