"""
Node: Problem State Initialization (live yfinance data, with Woerner
benchmark fallback via USE_LIVE_DATA=False).
"""

from __future__ import annotations

import numpy as np
from qiskit_algorithms.utils import algorithm_globals

from config.settings import (
    USE_LIVE_DATA, TICKERS, BUDGET, RISK_FACTOR, LOOKBACK_DAYS,
    RETURN_CONVENTION, END_DATE, PENALTY, N_SHOTS, REPS, SEED,
)
from data_sources.finance_data import get_problem_data


def problem_init_node(state: dict) -> dict:
    algorithm_globals.random_seed = SEED

    if USE_LIVE_DATA:
        problem = get_problem_data(
            TICKERS, lookback_days=LOOKBACK_DAYS, return_convention=RETURN_CONVENTION,
            end_date=END_DATE, budget=BUDGET, risk_factor=RISK_FACTOR,
            # allow_synthetic_fallback stays False (default) — a failed live fetch
            # should stop the run, never silently substitute fake market data.
        )
        mu, sigma    = problem.mu, problem.sigma
        n_assets     = problem.n_assets
        tickers_used = problem.tickers

        print(f"Live data: {n_assets} assets, {problem.n_trading_days_used} trading days, "
              f"{problem.date_range[0]} to {problem.date_range[1]}")
        if tickers_used != TICKERS:
            print(f"  Note: yfinance reordered columns — using {tickers_used} "
                  f"(mu/sigma are correctly aligned to this order, not TICKERS in config)")

        problem_input = (
            f"Portfolio optimization: {tickers_used}, budget={BUDGET}, "
            f"risk_aversion={RISK_FACTOR}, live data {problem.date_range[0]} to {problem.date_range[1]}"
        )

    else:
        n_assets = 6
        if not (1 <= BUDGET <= n_assets):
            raise ValueError(f"BUDGET={BUDGET} must satisfy 1 <= BUDGET <= n_assets={n_assets}")
        if not (0.0 <= RISK_FACTOR <= 1.0):
            raise ValueError(f"RISK_FACTOR={RISK_FACTOR} must be in [0, 1]")

        # Woerner's exact instance — Barkoutsos et al. 2020, Appendix B
        mu = np.array([0.7313, 0.9893, 0.2725, 0.8750, 0.7667, 0.3622])
        sigma = np.array([
            [ 0.7312, -0.6233,  0.4689, -0.5452, -0.0082, -0.3809],
            [-0.6233,  2.4732, -0.7538,  2.4659, -0.0733,  0.8945],
            [ 0.4689, -0.7538,  1.1543, -1.4095,  0.0007, -0.4301],
            [-0.5452,  2.4659, -1.4095,  3.5067,  0.2012,  1.0922],
            [-0.0082, -0.0733,  0.0007,  0.2012,  0.6231,  0.1509],
            [-0.3809,  0.8945, -0.4301,  1.0922,  0.1509,  0.8992]
        ])
        tickers_used = None

        problem_input = (
            f"Portfolio optimization: {n_assets} assets, budget={BUDGET}, "
            f"risk_aversion={RISK_FACTOR} (Woerner benchmark)"
        )

    print(f"Assets: {n_assets}, Budget: {BUDGET}, Risk factor: {RISK_FACTOR}")
    print(f"Covariance matrix shape: {sigma.shape}")
    print("\n✓ State initialized.")
    print("  Problem:", problem_input)

    return {
        "problem_input":  problem_input,
        "retry_count":    0,
        "n_assets":       n_assets,
        "tickers_used":   tickers_used,
        "mu":             mu,
        "sigma":          sigma,
        "risk_factor":    RISK_FACTOR,
        "budget":         BUDGET,
        "penalty":        PENALTY,
        "n_shots":        N_SHOTS,
        "reps":           REPS,
        "p_layers_recommended": None,
        "truncation_fraction":  None,
    }