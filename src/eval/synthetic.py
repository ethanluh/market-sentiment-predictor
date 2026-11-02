"""
Seeded synthetic market data with a planted sentiment signal.

Used for offline tests and ``python -m src.eval.run --synthetic``. With
``signal=0`` the series has no predictable component, so any edge a model shows
there is a harness or model bug, not a real effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.prediction.features import FEATURE_NAMES


@dataclass(frozen=True)
class SyntheticData:
    features: pd.DataFrame  # columns FEATURE_NAMES, rows with full history only
    log_returns: pd.Series  # full series, first value NaN
    vix: pd.Series  # VIX-like level aligned to ``log_returns``


def make_synthetic(n_days: int = 1500, signal: float = 0.5, seed: int = 0) -> SyntheticData:
    """
    Bar *t*'s return is ``signal * 0.01 * sentiment[t-1] + sigma[t] * z[t]``, with
    stochastic volatility and fat-tailed noise. Features at *t* use data up to *t*,
    so ``sentiment_agg`` at *t* predicts the return of bar *t+1*.
    """
    if n_days < 60:
        raise ValueError(f"n_days must be >= 60, got {n_days}")
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n_days, tz="UTC")

    raw = np.zeros(n_days)
    log_vol = np.zeros(n_days)
    shocks = rng.normal(size=(2, n_days))
    for t in range(1, n_days):
        raw[t] = 0.5 * raw[t - 1] + 0.8 * shocks[0, t]
        log_vol[t] = 0.97 * log_vol[t - 1] + 0.1 * shocks[1, t]
    sentiment = np.tanh(raw)
    sigma = 0.01 * np.exp(log_vol)
    noise = rng.standard_t(df=5, size=n_days) / np.sqrt(5.0 / 3.0)

    returns = np.full(n_days, np.nan)
    returns[1:] = signal * 0.01 * sentiment[:-1] + sigma[1:] * noise[1:]
    log_returns = pd.Series(returns, index=index, name="log_return")

    feats = pd.DataFrame(index=index, columns=FEATURE_NAMES, dtype=float)
    feats["sentiment_agg"] = sentiment
    feats["estimated_lag_hours"] = 6.0
    feats["sector_proximity"] = rng.uniform(0.0, 1.0, n_days)
    feats["momentum_5d"] = log_returns.rolling(5).sum()
    feats["momentum_20d"] = log_returns.rolling(20).sum()
    feats["realized_vol_20d"] = log_returns.rolling(20).std(ddof=0)
    feats["volume_zscore"] = rng.normal(size=n_days)
    feats["search_interest_zscore"] = rng.normal(size=n_days)
    feats["insider_flow_npr"] = rng.uniform(-1.0, 1.0, n_days)
    feats["return_lag_1"] = log_returns
    feats["regime_label"] = -1.0  # filled per fold from ``vix`` by the runner

    vix = pd.Series(sigma * 100.0 * np.sqrt(252.0), index=index, name="vix")
    return SyntheticData(features=feats.dropna(), log_returns=log_returns, vix=vix)
