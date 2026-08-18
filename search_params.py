"""
search_params.py

Coarse Latin-hypercube parameter search for bang-bang regime switches.

Outputs, in the same folder as this script:
    - results.csv

Adjust N_SAMPLES and GRID_SIZE below as runtime permits.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

import bang_bang as bb


# ============================================================
# User-editable coarse-search settings
# ============================================================

N_SAMPLES = 2000
GRID_SIZE = 60
N_TOP = 100
RANDOM_SEED = 123
N_WORKERS = max(1, int(0.8 * ((os.cpu_count() or 2) - 2)))

JUMP_THRESHOLD = 0.05

OUTPUT_ALL = "results.csv"


# If True, the script prints progress every PROGRESS_EVERY completed candidates.
PRINT_PROGRESS = True
PROGRESS_EVERY = 100


# ============================================================
# Search ranges
# ============================================================

# Ranges for sampled baseline parameter vectors.
SEARCH_RANGES: Dict[str, Tuple[float, float]] = {
    "theta": (1.05, 2.50),
    "k": (0.10, 0.90),
    "a": (0.075, 4.00),
    "Delta": (0.05, 2.00),
    "omega_bar": (0.50, 2.50),
    "lam": (0.00, 1.00),
}

# Ranges for one-dimensional comparative-static grids.
# For the coarse search, use the same ranges as the sampled baseline range.
GRID_RANGES: Dict[str, Tuple[float, float]] = SEARCH_RANGES.copy()

LOG_SAMPLE_PRIMITIVES = {"a", "Delta", "omega_bar"}
LINEAR_SAMPLE_PRIMITIVES = {"theta", "k", "lam"}

PARAMETER_ORDER: List[str] = ["theta", "a", "Delta", "omega_bar", "k", "lam"]
PRIMITIVES_TO_VARY: List[str] = ["a", "Delta", "lam", "theta", "k", "omega_bar"]

HIGH_IMPORTANCE = {"a", "Delta", "lam"}
MEDIUM_IMPORTANCE = {"theta", "k"}
LOW_IMPORTANCE = {"omega_bar"}

IMPORTANCE_WEIGHTS = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


# ============================================================
# Latin-hypercube sampler
# ============================================================

def transform_unit_to_range(u: float, low: float, high: float, log_scale: bool) -> float:
    """Map u in [0, 1] to [low, high], optionally log-uniformly."""
    if log_scale:
        if low <= 0.0 or high <= 0.0:
            raise ValueError("Log-scale sampling requires positive bounds.")
        return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
    return float(low + u * (high - low))


def latin_hypercube_sample(
    n_samples: int,
    seed: int,
    parameter_order: Sequence[str] = PARAMETER_ORDER,
    search_ranges: Dict[str, Tuple[float, float]] = SEARCH_RANGES,
) -> List[Dict[str, float]]:
    """Generate baseline parameter vectors using Latin-hypercube sampling."""
    sampler = qmc.LatinHypercube(d=len(parameter_order), seed=seed)
    unit = sampler.random(n=n_samples)

    samples: List[Dict[str, float]] = []
    for row in unit:
        params: Dict[str, float] = {}
        for j, name in enumerate(parameter_order):
            low, high = search_ranges[name]
            params[name] = transform_unit_to_range(
                float(row[j]),
                low,
                high,
                log_scale=(name in LOG_SAMPLE_PRIMITIVES),
            )
        samples.append(params)

    return samples


# ============================================================
# Candidate evaluation
# ============================================================

def evaluate_candidate(args: Tuple[int, Dict[str, float]]) -> Dict[str, Any]:
    """
    Evaluate one sampled baseline candidate.

    This function is top-level so it can be pickled by ProcessPoolExecutor.
    """
    candidate_id, params = args
    try:
        row = bb.score_parameter_vector(
            baseline=params,
            primitives_to_vary=PRIMITIVES_TO_VARY,
            grid_size=GRID_SIZE,
            grid_ranges=GRID_RANGES,
            jump_threshold=JUMP_THRESHOLD,
        )
        row["candidate_id"] = int(candidate_id)
        row["status"] = "ok"
        row["error"] = ""
        return row
    except Exception as exc:  # noqa: BLE001: preserve errors in output CSV.
        row = {name: params.get(name, np.nan) for name in PARAMETER_ORDER}
        row.update(
            {
                "candidate_id": int(candidate_id),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "total_score": -9999.0,
                "num_crossings_total": 0,
                "num_primitives_with_crossing": 0,
                "num_primitives_with_meaningful_jump": 0,
            }
        )
        return row


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    """Sort candidates by the main diagnostics."""
    sort_cols = [
        "total_score",
        "num_primitives_with_meaningful_jump",
        "num_primitives_with_crossing",
        "num_crossings_total",
    ]
    existing = [c for c in sort_cols if c in df.columns]
    return df.sort_values(existing, ascending=[False] * len(existing)).reset_index(drop=True)


def run_search() -> pd.DataFrame:
    """Run the full coarse Latin-hypercube search and write CSV outputs."""
    start = time.time()

    print("Starting coarse Latin-hypercube search")
    print(f"  N_SAMPLES      = {N_SAMPLES}")
    print(f"  GRID_SIZE      = {GRID_SIZE}")
    print(f"  N_TOP          = {N_TOP}")
    print(f"  RANDOM_SEED    = {RANDOM_SEED}")
    print(f"  N_WORKERS      = {N_WORKERS}")
    print(f"  JUMP_THRESHOLD = {JUMP_THRESHOLD}")
    print()

    samples = latin_hypercube_sample(N_SAMPLES, RANDOM_SEED)
    tasks = list(enumerate(samples))

    rows: List[Dict[str, Any]] = []

    if N_WORKERS == 1:
        for idx, task in enumerate(tasks, start=1):
            rows.append(evaluate_candidate(task))
            if PRINT_PROGRESS and (idx % PROGRESS_EVERY == 0 or idx == len(tasks)):
                elapsed = time.time() - start
                print(f"Completed {idx}/{len(tasks)} candidates after {elapsed / 60:.1f} minutes")
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(evaluate_candidate, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if PRINT_PROGRESS and (idx % PROGRESS_EVERY == 0 or idx == len(futures)):
                    elapsed = time.time() - start
                    print(f"Completed {idx}/{len(futures)} candidates after {elapsed / 60:.1f} minutes")

    df = pd.DataFrame(rows)
    df = sort_results(df)

    # Put the most important columns first when present.
    priority_cols = [
        "candidate_id",
        "status",
        "total_score",
        "num_crossings_total",
        "num_primitives_with_crossing",
        "num_primitives_with_meaningful_jump",
        "theta",
        "a",
        "Delta",
        "omega_bar",
        "k",
        "lam",
    ]
    priority_cols = [c for c in priority_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + remaining_cols]

    df.to_csv(OUTPUT_ALL, index=False)

    elapsed = time.time() - start
    print()
    print("Search complete")
    print(f"  Runtime: {elapsed / 60:.1f} minutes")
    print(f"  Wrote:   {OUTPUT_ALL}")
    print()

    if len(df) > 0:
        cols_to_show = [
            "candidate_id",
            "total_score",
            "num_primitives_with_meaningful_jump",
            "num_primitives_with_crossing",
            "num_crossings_total",
            "theta",
            "a",
            "Delta",
            "omega_bar",
            "k",
            "lam",
        ]
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        print("Top candidates:")
        print(df[cols_to_show].head(min(10, len(df))).to_string(index=False))

    return df


if __name__ == "__main__":
    run_search()
