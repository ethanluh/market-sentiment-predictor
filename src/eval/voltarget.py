"""
Volatility targeting as defined in ``docs/protocol_h3.md``.

A position's weight at the close of bar *t* is ``min(1, target / annualised_vol_t)``. It earns the
return of bar *t+1*, so a forecast never sees the return it is sized against. The portfolio is the
equal-weight mean over available tickers; buy-and-hold is the same mean with weight 1.

Every constant below is fixed by the protocol; changing one after the first development run is an
amendment, not an edit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.significance import block_bootstrap_ci
from src.eval.trading import ANNUALISATION, annualized_sharpe
from src.prediction.volatility import ewma_vol

DEV_START = "2020-01-02"
DEV_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
FORECASTERS = ("ewma94", "ewma97", "rolling63")  # registered priority order for ties
COST_BPS = 10.0
MIN_TICKERS = 20
RV_WINDOW = 21
BLOCK = 21
N_BOOT = 2000
SEED = 0
MAX_VV_RATIO = 0.80
MIN_SHARPE_DIFF = -0.10
TIE_TOLERANCE = 0.02
BARS_PER_YEAR = ANNUALISATION["1d"]


def forecast_vol(log_returns: pd.Series, name: str) -> pd.Series:
    """Per-bar volatility known at the close of each bar (daily, not annualised)."""
    if name == "ewma94":
        return ewma_vol(log_returns, lam=0.94)
    if name == "ewma97":
        return ewma_vol(log_returns, lam=0.97)
    if name == "rolling63":
        r2 = log_returns.dropna().pow(2)  # missing returns are skipped, as in ewma_vol
        return r2.rolling(63, min_periods=63).mean().pow(0.5).reindex(log_returns.index)
    raise ValueError(f"unknown forecaster {name!r}; the protocol fixes {FORECASTERS}")


def annualised_forecasts(log_returns: pd.DataFrame, name: str) -> pd.DataFrame:
    return log_returns.apply(lambda col: forecast_vol(col, name)) * np.sqrt(BARS_PER_YEAR)


def calibrate_target(sigma_ann: pd.DataFrame, start: str, end: str) -> float:
    """Median annualised forecast over all tickers and bars in ``[start, end]``."""
    window = sigma_ann.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
    values = window.to_numpy().ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"no finite forecasts between {start} and {end}")
    return float(np.median(values))


def target_weights(sigma_ann: pd.DataFrame, target: float) -> pd.DataFrame:
    """``min(1, target / sigma)``; NaN where the forecast is NaN. No leverage, no shorting."""
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"target must be finite and positive, got {target}")
    return (target / sigma_ann).clip(upper=1.0)


@dataclass(frozen=True)
class Backtest:
    """Per-bar series, indexed by return date, plus the mean weight and turnover per bar."""

    net: pd.Series
    buyhold: pd.Series
    exposure: pd.Series
    turnover: pd.Series

    def period(self, start: str, end: str | None = None) -> Backtest:
        """Bars whose UTC date is in ``[start, end]``, both inclusive."""
        day = self.net.index.tz_convert("UTC").normalize()
        keep = np.asarray(day >= pd.Timestamp(start, tz="UTC"))
        if end is not None:
            keep &= np.asarray(day <= pd.Timestamp(end, tz="UTC"))
        return Backtest(
            self.net[keep], self.buyhold[keep], self.exposure[keep], self.turnover[keep]
        )


def backtest(
    log_returns: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = COST_BPS,
    min_tickers: int = MIN_TICKERS,
    scored_from: str = DEV_START,
) -> Backtest:
    """
    The weight set at the close of bar *t* earns bar *t+1*. A ticker is available on a bar when that
    weight and the bar's return are both finite. From ``scored_from`` on, every bar needs at least
    ``min_tickers`` available (a bar with none raises too); earlier bars are warm-up and are kept
    only if some ticker is available. An unavailable ticker is flat: its first available bar, and
    any re-entry after a gap, is charged from a previous weight of 0; an exit costs nothing because
    there is no bar to trade it on.
    """
    if not log_returns.index.equals(weights.index) or list(log_returns.columns) != list(
        weights.columns
    ):
        raise ValueError("returns and weights must share an index and columns")
    r = log_returns.transform(np.expm1)
    applied = weights.shift(1)
    ok = applied.notna() & r.notna()
    counts = ok.sum(axis=1)
    scored = np.asarray(
        counts.index.tz_convert("UTC").normalize() >= pd.Timestamp(scored_from, tz="UTC")
    )
    bad = counts[scored & np.asarray(counts < min_tickers)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} scored bars have fewer than {min_tickers} available tickers "
            f"(first {bad.index[0].date()}: {int(bad.iloc[0])})"
        )
    live = counts > 0
    w = applied.where(ok)
    prev = w.shift(1).fillna(0.0)
    turn = (w - prev).abs().where(ok)
    cost = turn * cost_bps / 1e4
    net = (w * r - cost).where(ok).sum(axis=1)[live] / counts[live]
    buyhold = r.where(ok).sum(axis=1)[live] / counts[live]
    exposure = w.sum(axis=1)[live] / counts[live]
    turnover = turn.sum(axis=1)[live] / counts[live]
    if net.empty:
        raise ValueError("no bar has an available ticker")
    return Backtest(net, buyhold, exposure, turnover)


def rv21(returns: pd.Series) -> pd.Series:
    """Square root of the mean squared return over the 21 bars ending at each bar; NaN before 21."""
    return returns.pow(2).rolling(RV_WINDOW, min_periods=RV_WINDOW).mean().pow(0.5)


def dispersion(rv: np.ndarray) -> float:
    """Relative dispersion ``std / mean`` (ddof 1) of a realised-volatility series."""
    rv = np.asarray(rv, dtype=float)
    if len(rv) < 2 or not np.isfinite(rv).all() or rv.mean() <= 0.0:
        raise ValueError("dispersion needs at least two finite, positive-mean values")
    return float(rv.std(ddof=1) / rv.mean())


def net_sharpe(returns: np.ndarray) -> float:
    return annualized_sharpe(np.asarray(returns, dtype=float), BARS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def worst_window(returns: pd.Series, window: int = RV_WINDOW) -> float:
    growth = (1.0 + returns).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
    return float(growth.min())


def _sharpe_diff(a: np.ndarray, b: np.ndarray) -> float:
    diff = net_sharpe(a) - net_sharpe(b)
    if not np.isfinite(diff):
        raise ValueError("Sharpe is undefined for a zero-variance series")
    return diff


def evaluate(bt: Backtest, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, object]:
    """H3a and H3b on one period, with the protocol's paired block bootstrap (90% interval, so
    ``ratio_hi`` is the 95th percentile and ``sharpe_diff_lo`` the 5th)."""
    rv_s = rv21(bt.net).dropna()
    rv_b = rv21(bt.buyhold).dropna()
    if len(rv_s) < 2 * BLOCK:
        raise ValueError(f"{len(rv_s)} usable RV windows; need at least {2 * BLOCK}")
    rv_s_a, rv_b_a = rv_s.to_numpy(), rv_b.to_numpy()
    ratio = dispersion(rv_s_a) / dispersion(rv_b_a)
    _, ratio_hi = block_bootstrap_ci(
        lambda a, b: dispersion(a) / dispersion(b),
        rv_s_a,
        rv_b_a,
        block=BLOCK,
        n_boot=n_boot,
        seed=seed,
        level=0.90,
    )
    net_a, bh_a = bt.net.to_numpy(), bt.buyhold.to_numpy()
    sharpe_diff = _sharpe_diff(net_a, bh_a)
    sharpe_lo, _ = block_bootstrap_ci(
        _sharpe_diff, net_a, bh_a, block=BLOCK, n_boot=n_boot, seed=seed, level=0.90
    )
    matched = bt.buyhold * float(bt.exposure.mean())  # exposure-matched, zero cost
    return {
        "n_bars": len(bt.net),
        "vv_ratio": ratio,
        "ratio_hi": ratio_hi,
        "sharpe": net_sharpe(net_a),
        "sharpe_buyhold": net_sharpe(bh_a),
        "sharpe_diff": sharpe_diff,
        "sharpe_diff_lo": sharpe_lo,
        "mean_exposure": float(bt.exposure.mean()),
        "turnover": float(bt.turnover.mean()),
        "max_drawdown": max_drawdown(bt.net),
        "max_drawdown_buyhold": max_drawdown(bt.buyhold),
        "max_drawdown_matched": max_drawdown(matched),
        "worst_21": worst_window(bt.net),
        "worst_21_buyhold": worst_window(bt.buyhold),
        "worst_21_matched": worst_window(matched),
        "h3a": bool(ratio <= MAX_VV_RATIO and ratio_hi < 1.0),
        "h3b": bool(sharpe_lo >= MIN_SHARPE_DIFF),
    }


@dataclass(frozen=True)
class Selection:
    winner: str
    proceed: bool
    reasons: list[str]


def select(dev: dict[str, dict[str, object]]) -> Selection:
    """Lowest ratio wins; within ``TIE_TOLERANCE`` of it the first registered forecaster wins. The
    holdout opens only if the winner meets both H3a and H3b on development."""
    if set(dev) != set(FORECASTERS):
        raise ValueError(f"dev results must cover exactly {FORECASTERS}, got {sorted(dev)}")
    best = min(float(dev[n]["vv_ratio"]) for n in FORECASTERS)  # type: ignore[arg-type]
    winner = next(
        n
        for n in FORECASTERS
        if float(dev[n]["vv_ratio"]) <= best + TIE_TOLERANCE + 1e-12  # type: ignore[arg-type]
    )
    reasons = [f"{winner} fails {k}" for k in ("h3a", "h3b") if not dev[winner][k]]
    return Selection(winner, not reasons, reasons)
