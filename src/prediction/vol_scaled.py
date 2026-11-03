"""
Pooled, volatility-normalised quantile model (hypothesis H2, ``docs/protocol_h2.md``).

The model predicts quantiles of ``y = forward return / (sigma * sqrt(steps))`` from vol-normalised
features, pooled over tickers, and scales back with the known EWMA volatility. With no features it
is the pooled filtered-historical-simulation control (``fhs-pooled``).

The candidate list and their order are fixed by the protocol; do not add to them here without a new
hypothesis id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.prediction.model import QUANTILES
from src.prediction.volatility import fhs_quantiles

SEED = 0

PRICE_FEATURES = ("mom5", "mom20", "ret1", "volratio", "volz")
NEWS_FEATURES = (*PRICE_FEATURES, "sent", "sectprox", "regime")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "price": PRICE_FEATURES,
    "news": NEWS_FEATURES,
}


@dataclass(frozen=True)
class Candidate:
    id: str
    estimator: str  # "fhs", "lq" (L1 linear quantile regression) or "gb" (gradient boosting)
    feature_set: str  # a key of FEATURE_SETS
    alpha: float = 0.0  # lq only
    n_estimators: int = 0  # gb only


def _lq(alpha: float, feature_set: str) -> Candidate:
    return Candidate(f"lq-a{alpha}-{feature_set}", "lq", feature_set, alpha=alpha)


def _gb(n: int, feature_set: str) -> Candidate:
    return Candidate(f"gb-{n}-{feature_set}", "gb", feature_set, n_estimators=n)


CONTROL = Candidate("fhs-pooled", "fhs", "none")

# Simplest first: the order is the protocol's tie-break.
COMPLEXITY_ORDER: tuple[Candidate, ...] = (
    _lq(0.01, "price"),
    _lq(0.01, "news"),
    _lq(0.001, "price"),
    _lq(0.001, "news"),
    _gb(50, "price"),
    _gb(50, "news"),
    _gb(100, "price"),
    _gb(100, "news"),
)
CANDIDATES: dict[str, Candidate] = {c.id: c for c in (CONTROL, *COMPLEXITY_ORDER)}

_REQUIRED_RAW = (
    "momentum_5d",
    "momentum_20d",
    "return_lag_1",
    "realized_vol_20d",
    "volume_zscore",
    "sentiment_agg",
    "sector_proximity",
    "regime_label",
)


def normalised_features(features: pd.DataFrame, sigma: pd.Series, feature_set: str) -> pd.DataFrame:
    """
    The protocol's vol-normalised features for ``feature_set`` (columns ordered as in the protocol).

    ``sigma`` is the EWMA volatility known at each row's bar close. A missing column or a
    non-positive or missing volatility raises.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    missing = [c for c in _REQUIRED_RAW if c not in features.columns]
    if missing:
        raise ValueError(f"features are missing columns {missing}")
    s = sigma.reindex(features.index).to_numpy(dtype=float)
    if not (np.isfinite(s).all() and (s > 0).all()):
        raise ValueError("sigma must be finite and positive at every feature row")
    f = features
    cols = {
        "mom5": f["momentum_5d"].to_numpy() / (s * np.sqrt(5)),
        "mom20": f["momentum_20d"].to_numpy() / (s * np.sqrt(20)),
        "ret1": f["return_lag_1"].to_numpy() / s,
        "volratio": f["realized_vol_20d"].to_numpy() / s,
        "volz": f["volume_zscore"].to_numpy(),
        "sent": f["sentiment_agg"].to_numpy(),
        "sectprox": f["sector_proximity"].to_numpy(),
        "regime": f["regime_label"].to_numpy(),
    }
    return pd.DataFrame({k: cols[k] for k in FEATURE_SETS[feature_set]}, index=features.index)


class PooledQuantileModel:
    """Quantiles of the standardised return from standardised features; trains in standardised units."""

    def __init__(self, candidate: Candidate, seed: int = SEED) -> None:
        self.candidate = candidate
        self.seed = seed
        self._scaler: object | None = None
        self._models: list[object] = []
        self._fixed: np.ndarray | None = None

    def _estimator(self, q: float) -> object:
        c = self.candidate
        if c.estimator == "lq":
            from sklearn.linear_model import QuantileRegressor

            return QuantileRegressor(quantile=q, alpha=c.alpha, solver="highs")
        if c.estimator == "gb":
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=c.n_estimators,
                max_depth=2,
                learning_rate=0.05,
                min_samples_leaf=200,
                subsample=0.7,
                random_state=self.seed,
            )
        raise ValueError(f"no estimator for {c.estimator!r}")

    def fit(self, x: pd.DataFrame, y_std: np.ndarray) -> "PooledQuantileModel":
        y = np.asarray(y_std, dtype=float)
        if len(x) != len(y) or len(y) == 0 or not np.isfinite(y).all():
            raise ValueError("fit needs matching, non-empty, finite rows")
        if self.candidate.estimator == "fhs":
            self._fixed = fhs_quantiles(y)
            return self
        if x.shape[1] != len(FEATURE_SETS[self.candidate.feature_set]):
            raise ValueError("feature columns do not match the candidate's feature set")
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("features contain NaN or infinity")
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x.to_numpy())  # training window only
        z = scaler.transform(x.to_numpy())
        self._scaler = scaler
        self._models = [self._estimator(q).fit(z, y) for q in QUANTILES]  # type: ignore[attr-defined]
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """``(n, 3)`` standardised P10, P50, P90, sorted per row to repair quantile crossing."""
        if self.candidate.estimator == "fhs":
            if self._fixed is None:
                raise RuntimeError("model is not fitted")
            return np.tile(self._fixed, (len(x), 1))
        if self._scaler is None:
            raise RuntimeError("model is not fitted")
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("features contain NaN or infinity")
        z = self._scaler.transform(x.to_numpy())  # type: ignore[attr-defined]
        out = np.column_stack([m.predict(z) for m in self._models])  # type: ignore[attr-defined]
        return np.sort(out, axis=1)
