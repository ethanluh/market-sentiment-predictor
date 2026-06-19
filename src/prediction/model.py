"""
Quantile regression over log-returns.

Predicts the P10 / P50 / P90 of the forward log-return at each horizon
(``1h``, ``1d``, ``5d``) rather than a single point estimate, which supports
position sizing and threshold-based entry. One estimator is trained per
``(horizon, quantile)`` pair. Predicted quantiles are sorted per row so the
output never exhibits quantile crossing (p10 <= p50 <= p90).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.prediction.features import FEATURE_NAMES, FeatureVector

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)
HORIZONS: tuple[str, ...] = ("1h", "1d", "5d")


@dataclass
class QuantilePrediction:
    p10: float
    p50: float
    p90: float

    def as_dict(self) -> dict[str, float]:
        return {"p10": self.p10, "p50": self.p50, "p90": self.p90}

    @classmethod
    def from_sorted(cls, values: list[float]) -> "QuantilePrediction":
        """Build from three quantile estimates, enforcing monotonicity."""
        lo, mid, hi = sorted(values)
        return cls(p10=lo, p50=mid, p90=hi)


def _make_estimator(backend: str, quantile: float):  # type: ignore[no-untyped-def]
    """Construct a single-quantile sklearn estimator."""
    if backend == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            loss="quantile", alpha=quantile, n_estimators=200, max_depth=3, random_state=0
        )
    if backend == "linear":
        from sklearn.linear_model import QuantileRegressor

        return QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
    raise ValueError(f"Unknown backend {backend!r}; use 'gbr' or 'linear'")


@dataclass
class QuantileReturnModel:
    horizons: tuple[str, ...] = HORIZONS
    quantiles: tuple[float, ...] = QUANTILES
    backend: str = "gbr"
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    _models: dict[tuple[str, float], object] = field(default_factory=dict, repr=False)
    _available: set[str] = field(default_factory=set, repr=False)

    def _prepare_x(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reindex to the trained feature order, raising on missing columns."""
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Feature matrix missing columns: {missing}")
        return X[self.feature_names]

    def fit(self, X: pd.DataFrame, y: dict[str, pd.Series]) -> "QuantileReturnModel":
        """
        Train one estimator per (horizon, quantile).

        ``y`` maps each horizon to a target log-return series aligned to
        ``X.index``. Rows with NaN features or target are dropped per horizon.
        Horizons with no usable rows are skipped (with a warning) and omitted
        from predictions.
        """
        Xp = self._prepare_x(X)
        self._models.clear()
        self._available.clear()

        for horizon in self.horizons:
            if horizon not in y:
                continue
            target = y[horizon].reindex(Xp.index)
            mask = Xp.notna().all(axis=1) & target.notna()
            if mask.sum() == 0:
                warnings.warn(f"No usable training rows for horizon {horizon!r}; skipping")
                continue
            X_fit = Xp[mask].to_numpy()
            y_fit = target[mask].to_numpy()
            for q in self.quantiles:
                est = _make_estimator(self.backend, q)
                est.fit(X_fit, y_fit)
                self._models[(horizon, q)] = est
            self._available.add(horizon)
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, list[QuantilePrediction]]:
        """Predict quantiles for each row; output is monotonic per row."""
        Xp = self._prepare_x(X).to_numpy()
        out: dict[str, list[QuantilePrediction]] = {}
        for horizon in self._available:
            cols = np.column_stack(
                [self._models[(horizon, q)].predict(Xp) for q in self.quantiles]  # type: ignore[attr-defined]
            )
            cols = np.sort(cols, axis=1)  # remove quantile crossing
            out[horizon] = [QuantilePrediction(*row) for row in cols]
        return out

    def predict_one(self, fv: FeatureVector) -> dict[str, QuantilePrediction]:
        """Predict quantiles for a single :class:`FeatureVector`."""
        X = fv.to_series().to_frame().T
        per_horizon = self.predict(X)
        return {h: preds[0] for h, preds in per_horizon.items()}

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "horizons": self.horizons,
                "quantiles": self.quantiles,
                "backend": self.backend,
                "feature_names": self.feature_names,
                "models": self._models,
                "available": self._available,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "QuantileReturnModel":
        import joblib

        state = joblib.load(path)
        obj = cls(
            horizons=tuple(state["horizons"]),
            quantiles=tuple(state["quantiles"]),
            backend=state["backend"],
            feature_names=list(state["feature_names"]),
        )
        obj._models = state["models"]
        obj._available = set(state["available"])
        return obj
