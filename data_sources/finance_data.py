"""
finance_data.py

Deterministic (no-LLM) fetch-and-compute pipeline for portfolio problem
data: given a list of tickers, produces the annualized expected-return
vector (mu) and covariance matrix (sigma) that Node 1's formulation
prompt currently receives as hardcoded constants.

Convention: daily log returns, annualized by the standard 252-trading-
day-per-year convention — matching Ganguly (SSRN, Mar 2026), which
states: "Daily log-returns were computed as r_t = log(p_t / p_{t-1})
and annualized using the standard convention of 252 trading days."
Log returns are used specifically because they are time-additive
(multi-period log return = sum of single-period log returns), which
simple returns are not — the standard justification for this convention
in quantitative finance generally, independent of the Ganguly citation.

Both `lookback_days` and `return_convention` are overridable parameters,
not hardcoded — 252/"log" are defaults, not the only supported values.

Design choice, stated explicitly rather than left implicit:
fetch_price_history() ALWAYS raises on failure — it has no fallback
logic at all, by design, so it has one simple contract regardless of
caller. The synthetic-data fallback lives ONLY in get_problem_data(),
the one place that decides whether that's acceptable — never silent,
never split across two functions coordinating through shared state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

SUPPORTED_RETURN_CONVENTIONS = ("log", "simple")


class InsufficientHistoryError(RuntimeError):
    """Raised when a ticker doesn't have enough trading history for the requested lookback."""


class TickerFetchError(RuntimeError):
    """Raised when one or more tickers can't be resolved / return no data."""


@dataclass
class ProblemData:
    mu: np.ndarray
    sigma: np.ndarray
    n_assets: int
    tickers: list[str]
    date_range: tuple[str, str]      # (first date used, last date used), as ISO strings
    n_trading_days_used: int
    return_convention: str
    lookback_days_requested: int
    is_synthetic: bool
    budget: int | None = None
    risk_factor: float | None = None


# ------------------------------------------------------------------ #
# Step 1: fetch — pure I/O, no computation, ALWAYS raises on failure.
# No fallback logic here, by design — see module docstring.
# ------------------------------------------------------------------ #

def fetch_price_history(
    tickers: list[str],
    lookback_days: int = 252,
    end_date: str | datetime | None = None,
    calendar_buffer_multiplier: float = 1.6,
) -> pd.DataFrame:
    """
    Returns a DataFrame of adjusted close prices, one column per ticker,
    trimmed to exactly the most recent `lookback_days` trading days
    ending at `end_date` (or today, if not given).

    end_date: optional, e.g. "2026-02-15" or a datetime. Use this to
    deliberately reproduce a specific historical window — e.g. a prior
    paper's reported instance — rather than always pulling "as of today."
    Note yfinance's `end` is exclusive of that calendar date's own bar;
    pass the day AFTER your intended last trading day if you need that
    day's price included.

    Raises TickerFetchError or InsufficientHistoryError on any failure —
    never returns partial or fake data.

    calendar_buffer_multiplier: trading days are ~5/7 of calendar days
    once weekends are excluded, and holidays reduce that further — 1.6x
    is a generous buffer so a `lookback_days=252` request reliably yields
    at least 252 rows before trimming, without over-fetching by a lot.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be positive, got {lookback_days}")

    calendar_days = int(lookback_days * calendar_buffer_multiplier) + 10
    end = pd.Timestamp(end_date) if end_date is not None else datetime.today()
    start = end - timedelta(days=calendar_days)

    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        raise TickerFetchError(f"yfinance download failed for {tickers}: {e}") from e

    if raw is None or raw.empty:
        raise TickerFetchError(
            f"yfinance returned no data for {tickers} — check ticker symbols "
            f"are correct and currently listed."
        )

    # yfinance returns a single-level frame for one ticker, a DataFrame
    # with a MultiIndex column (field, ticker) for multiple — normalize
    # to a single-level DataFrame with one column per ticker either way.
    if len(tickers) == 1:
        prices = raw[["Close"]].copy()
        prices.columns = tickers
    else:
        prices = raw["Close"].copy()

    # A ticker that doesn't exist / is delisted often comes back as an
    # all-NaN column rather than raising — catch that explicitly and name
    # the offending ticker(s), rather than letting NaNs silently propagate
    # into mu/sigma downstream.
    all_nan_tickers = [t for t in prices.columns if prices[t].isna().all()]
    if all_nan_tickers:
        raise TickerFetchError(
            f"No data returned for ticker(s): {all_nan_tickers}. "
            f"Check these are valid, currently-listed symbols."
        )

    prices = prices.dropna(how="any")  # any date missing ANY ticker's price is dropped

    if len(prices) < lookback_days:
        raise InsufficientHistoryError(
            f"Only {len(prices)} trading days of overlapping history available "
            f"for {tickers} in the fetched window, but lookback_days={lookback_days} "
            f"was requested. This usually means one of these tickers IPO'd or started "
            f"trading more recently than the requested lookback window — try a shorter "
            f"lookback_days, or check for a newly-listed ticker in this set."
        )

    return prices.iloc[-lookback_days:]


def _synthetic_price_history(tickers: list[str], lookback_days: int, seed: int) -> pd.DataFrame:
    """
    Only ever called from get_problem_data(), and only when
    allow_synthetic_fallback=True was explicitly passed — never reached
    silently. Correlated synthetic prices, loosely modeled on the shape
    of Dr. Yu's lab fallback (common market factor + idiosyncratic noise
    per ticker), for offline development only.

    Generates a generous date range and slices to exactly the last
    `lookback_days` entries — deliberately NOT relying on
    pd.bdate_range(end=X, periods=N) to return exactly N rows, since
    that call silently returns N-1 rows whenever X falls on a weekend
    (confirmed: today's date is a weekend at time of writing, which is
    exactly what surfaced this during testing). Slicing the tail of an
    over-generated range is correct regardless of what day it's run on.
    """
    rng = np.random.default_rng(seed)
    buffer_days = lookback_days + 10
    dates = pd.bdate_range(end=datetime.today(), periods=buffer_days)[-lookback_days:]
    market_factor = rng.normal(0, 0.01, lookback_days)
    returns = {t: 0.5 * market_factor + 0.01 * rng.normal(0, 1, lookback_days) for t in tickers}
    returns_df = pd.DataFrame(returns, index=dates)
    prices = (1 + returns_df).cumprod() * 100
    return prices


# ------------------------------------------------------------------ #
# Step 2: compute — pure math, fully testable without any network
# ------------------------------------------------------------------ #

def compute_mu_sigma(
    prices: pd.DataFrame,
    return_convention: str = "log",
    trading_days_per_year: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """
    mu: annualized expected return per asset, shape (n_assets,)
    sigma: annualized covariance matrix, shape (n_assets, n_assets)

    Column order of the input DataFrame determines the order of both
    outputs — caller is responsible for keeping tickers/mu/sigma aligned.
    """
    if return_convention not in SUPPORTED_RETURN_CONVENTIONS:
        raise ValueError(
            f"return_convention must be one of {SUPPORTED_RETURN_CONVENTIONS}, "
            f"got {return_convention!r}"
        )

    if return_convention == "log":
        returns = np.log(prices / prices.shift(1)).dropna(how="any")
    else:  # "simple"
        returns = prices.pct_change().dropna(how="any")

    mu = returns.mean().to_numpy() * trading_days_per_year
    sigma = returns.cov().to_numpy() * trading_days_per_year
    return mu, sigma


# ------------------------------------------------------------------ #
# Step 3: convenience wrapper — the function Node 0/problem-init calls.
# This is the ONE place the synthetic-fallback decision is made.
# ------------------------------------------------------------------ #

def get_problem_data(
    tickers: list[str],
    lookback_days: int = 252,
    return_convention: str = "log",
    end_date: str | datetime | None = None,
    budget: int | None = None,
    risk_factor: float | None = None,
    allow_synthetic_fallback: bool = False,
    synthetic_seed: int = 42,
) -> ProblemData:
    n_assets = len(tickers)
    if budget is not None and not (1 <= budget <= n_assets):
        raise ValueError(f"budget={budget} must satisfy 1 <= budget <= n_assets={n_assets}")
    if risk_factor is not None and not (0.0 <= risk_factor <= 1.0):
        raise ValueError(f"risk_factor={risk_factor} is outside the conventional [0, 1] range "
                          f"— double-check this is intentional.")

    is_synthetic = False
    try:
        prices = fetch_price_history(tickers, lookback_days, end_date=end_date)
    except (TickerFetchError, InsufficientHistoryError):
        if not allow_synthetic_fallback:
            raise
        prices = _synthetic_price_history(tickers, lookback_days, synthetic_seed)
        is_synthetic = True

    mu, sigma = compute_mu_sigma(prices, return_convention)

    return ProblemData(
        mu=mu, sigma=sigma, n_assets=n_assets, tickers=list(prices.columns),
        date_range=(str(prices.index[0].date()), str(prices.index[-1].date())),
        n_trading_days_used=len(prices),
        return_convention=return_convention,
        lookback_days_requested=lookback_days,
        is_synthetic=is_synthetic,
        budget=budget, risk_factor=risk_factor,
    )
