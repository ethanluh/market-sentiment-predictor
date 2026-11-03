"""
Pooled walk-forward evaluation of a vol-scaled candidate (``docs/protocol_h2.md``).

One model per fold is fit on the stacked rows of all tickers over a rolling window of calendar
dates, then scored on the next block of dates. Output is one ``EvalResult`` per ticker in return
units, with the standard baselines on the same rows, so the panel test, Benjamini-Hochberg and the
gate apply unchanged.

Row sets are decided by dates only, never by the candidate, so every candidate is scored on identical
rows. A row is scored when it is at or after ``score_start`` and its label's last bar is on or before
``label_end_max``; rows outside may still train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.baselines import make_baselines
from src.eval.runner import EvalResult, summarize
from src.eval.splits import Fold, purged_walk_forward
from src.eval.targets import forward_log_return
from src.prediction.baselines import horizon_to_steps
from src.prediction.regime import RegimeClassifier
from src.prediction.vol_scaled import CANDIDATES, PooledQuantileModel, normalised_features
from src.prediction.volatility import ewma_vol, standardize

TRAIN_DATES = 504
REFIT_EVERY = 21
POOL_START = "2020-01-02"
BASELINE_WINDOW = 252
MAX_VIX_STALENESS = pd.Timedelta(days=5)  # the same bound the dataset builder applies


def fold_schedule(n: int, train_dates: int, refit_every: int, steps: int) -> list[Fold]:
    """
    Rolling folds over ``n`` calendar dates with a ``steps``-date embargo, plus a final partial block
    so the newest dates are scored too (the regular schedule drops a tail shorter than ``refit_every``).
    """
    folds = purged_walk_forward(n, train_dates, refit_every, step=refit_every, embargo=steps)
    next_start = int(folds[-1].train[0]) + refit_every
    test_start = next_start + train_dates + steps
    if int(folds[-1].test[-1]) + 1 < n:
        folds.append(
            Fold(
                train=np.arange(next_start, next_start + train_dates),
                test=np.arange(test_start, n),
            )
        )
    return folds


def pooled_walk_forward(
    features: dict[str, pd.DataFrame],
    log_returns: dict[str, pd.Series],
    vix: pd.Series | None,
    candidate_id: str,
    horizon: str = "5d",
    *,
    train_dates: int = TRAIN_DATES,
    refit_every: int = REFIT_EVERY,
    pool_start: str = POOL_START,
    score_start: str | None = None,
    label_end_max: str | None = None,
    cost_bps: float = 1.0,
    n_boot: int = 500,
    seed: int = 0,
    baseline_cache: dict | None = None,
) -> dict[str, EvalResult]:
    """
    Per-ticker results for one candidate. Raises when a ticker has no scored rows, when feature rows
    are not consecutive bars on the shared calendar, or when VIX is missing or stale for a regime
    feature. ``baseline_cache`` (any dict) lets several candidates share identical baseline forecasts.
    """
    if candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate_id!r}; choose from {sorted(CANDIDATES)}")
    cand = CANDIDATES[candidate_id]
    if set(features) != set(log_returns):
        raise ValueError("features and log_returns must cover the same tickers")
    steps = horizon_to_steps(horizon)
    tz = "UTC"
    start = pd.Timestamp(pool_start, tz=tz)
    s_from = pd.Timestamp(score_start, tz=tz) if score_start else None
    l_to = pd.Timestamp(label_end_max, tz=tz) if label_end_max else None

    # Per-ticker frames of normalised features, standardised target, raw target, label end date.
    frames: dict[str, pd.DataFrame] = {}
    raw_y: dict[str, pd.Series] = {}
    sigma_all: dict[str, pd.Series] = {}
    label_end: dict[str, pd.Series] = {}
    for t, feats in features.items():
        lr = log_returns[t]
        feats = feats.loc[feats.index >= start]
        if feats.empty or feats.isna().any().any():
            raise ValueError(f"{t}: features are empty or contain NaN after {pool_start}")
        pos = lr.index.get_indexer(feats.index)
        if (pos < 0).any() or not (np.diff(pos) == 1).all():
            raise ValueError(f"{t}: feature rows must be consecutive bars of its returns")
        sigma = ewma_vol(lr)
        target = forward_log_return(lr, steps)
        y = target.reindex(feats.index)
        sig_rows = sigma.reindex(feats.index)
        x = normalised_features(feats, sig_rows, cand.feature_set)
        ends = pd.Series(
            [lr.index[p + steps] if p + steps < len(lr) else pd.NaT for p in pos],
            index=feats.index,
        )
        frames[t] = x.assign(
            _y_std=standardize(y.to_numpy(), sig_rows.to_numpy(), steps),
        )
        raw_y[t], sigma_all[t], label_end[t] = y, sig_rows, ends

    calendar = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
    cal_pos = {d: i for i, d in enumerate(calendar)}
    for t, f in frames.items():
        p = np.array([cal_pos[d] for d in f.index])
        if not (np.diff(p) == 1).all():
            raise ValueError(f"{t}: rows are not contiguous on the shared trading calendar")

    use_regime = "regime" in frames[next(iter(frames))].columns
    vix_dates: np.ndarray | None = None
    if use_regime:
        if vix is None:
            raise ValueError("the candidate uses the regime feature; vix is required")
        v = vix.sort_index()
        at = v.index.searchsorted(calendar, side="right") - 1
        if (at < 0).any():
            raise ValueError("vix has no value at or before some calendar dates")
        stale = calendar - v.index[at]
        if (stale > MAX_VIX_STALENESS).any():
            worst = calendar[np.argmax(stale)]
            raise ValueError(f"vix is stale by more than {MAX_VIX_STALENESS} at {worst:%Y-%m-%d}")
        vix_dates = v.to_numpy(dtype=float)[at]
        if np.isnan(vix_dates).any():
            raise ValueError("vix has NaN values on the calendar")

    folds = fold_schedule(len(calendar), train_dates, refit_every, steps)

    out_rows: dict[str, list[tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]]] = {
        t: [] for t in frames
    }
    for fold in folds:
        tr_dates, te_dates = calendar[fold.train], calendar[fold.test]
        regime_tr = regime_te = None
        if vix_dates is not None:
            clf = RegimeClassifier().fit(vix_dates[fold.train])
            regime_tr = dict(zip(tr_dates, clf.predict(vix_dates[fold.train]).astype(float)))
            regime_te = dict(zip(te_dates, clf.predict(vix_dates[fold.test]).astype(float)))

        def stack(dates: pd.DatetimeIndex, regimes: dict | None) -> pd.DataFrame:
            parts = []
            for t, f in frames.items():
                part = f.loc[f.index.isin(dates)].copy()
                if regimes is not None:
                    part["regime"] = [regimes[d] for d in part.index]
                part["_ticker"] = t
                parts.append(part)
            return pd.concat(parts)

        train, test = stack(tr_dates, regime_tr), stack(te_dates, regime_te)
        train = train[np.isfinite(train["_y_std"])]
        cols = [c for c in train.columns if not c.startswith("_")]
        model = PooledQuantileModel(cand).fit(train[cols], train["_y_std"].to_numpy())
        std_q = model.predict(test[cols])
        for t in frames:
            mask = (test["_ticker"] == t).to_numpy()
            rows = test.index[mask]
            sig = sigma_all[t].loc[rows].to_numpy()
            if not (np.isfinite(sig).all() and (sig > 0).all()):
                raise ValueError(f"{t}: volatility is not finite and positive on a test row")
            q_raw = std_q[mask] * (sig * np.sqrt(steps))[:, None]
            y = raw_y[t].loc[rows].to_numpy()
            keep = np.isfinite(y)
            if s_from is not None:
                keep &= rows >= s_from
            if l_to is not None:
                keep &= (label_end[t].loc[rows] <= l_to).to_numpy()
            if keep.any():
                out_rows[t].append((rows[keep], y[keep], q_raw[keep]))

    results: dict[str, EvalResult] = {}
    for t, parts in out_rows.items():
        if not parts:
            raise ValueError(f"{t}: no scored rows for this candidate and window")
        idx = parts[0][0].append([p[0] for p in parts[1:]]) if len(parts) > 1 else parts[0][0]
        y = np.concatenate([p[1] for p in parts])
        model_pred = np.concatenate([p[2] for p in parts])
        lr = log_returns[t]
        key = (t, horizon, idx[0], idx[-1], len(idx))
        if baseline_cache is not None and key in baseline_cache:
            base_preds = baseline_cache[key]
        else:
            baselines = make_baselines(
                lr, forward_log_return(lr, steps), horizon, steps, BASELINE_WINDOW
            )
            positions = lr.index.get_indexer(idx)
            base_preds = {
                name: np.array([forecast(int(p)) for p in positions])
                for name, forecast in baselines.items()
            }
            if baseline_cache is not None:
                baseline_cache[key] = base_preds
        preds = {"model": model_pred, **base_preds}
        res = EvalResult(horizon, steps, len(folds), idx, y, preds)
        res.summary = summarize(res, cost_bps=cost_bps, n_boot=n_boot, seed=seed)
        results[t] = res
    return results
