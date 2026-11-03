"""
Development-set selection for H2 (``docs/protocol_h2.md``, "Selection").

Everything here reads development results only. The statistic S(c, ref) is the mean over all
development test dates of the panel-averaged relative pinball difference of candidate ``c`` against
``ref`` (each ticker's differential divided by its own mean reference loss); negative is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.eval.live import with_baseline
from src.eval.panel import relative_loss_differential
from src.eval.runner import EvalResult
from src.prediction.vol_scaled import CANDIDATES, COMPLEXITY_ORDER, CONTROL

TIE_TOLERANCE = 0.001  # candidates within this of the best S(c, fhs) tie; the simpler one wins
MATERIALITY = -0.005  # the winner must be at least 0.5% better than every featureless forecaster
REFERENCES = ("fhs", "garch", CONTROL.id)


def relative_score(cand: dict[str, EvalResult], baseline: str) -> float:
    """S for one candidate against a baseline present in each ticker's ``preds``."""
    frame = pd.DataFrame(
        {t: relative_loss_differential(r, baseline) for t, r in cand.items()}
    ).sort_index()
    return float(frame.mean(axis=1).mean())


def score_candidate(
    cand: dict[str, EvalResult], control: dict[str, EvalResult]
) -> dict[str, float]:
    """S against ``fhs``, ``garch`` and the pooled control, on the same rows."""
    if set(cand) != set(control):
        raise ValueError("candidate and control must cover the same tickers")
    paired = {t: with_baseline(cand[t], CONTROL.id, control[t]) for t in cand}
    return {ref: relative_score(paired, ref) for ref in REFERENCES}


@dataclass(frozen=True)
class Selection:
    scores: dict[str, dict[str, float]]  # candidate id -> reference -> S
    best_price: str
    best_news: str
    overall: str
    proceed: bool
    reasons: list[str] = field(default_factory=list)


def _pick(ids: list[str], scores: dict[str, dict[str, float]]) -> str:
    best = min(scores[i]["fhs"] for i in ids)
    tied = {i for i in ids if scores[i]["fhs"] <= best + TIE_TOLERANCE}
    return next(c.id for c in COMPLEXITY_ORDER if c.id in tied)


def select(dev: dict[str, dict[str, EvalResult]]) -> Selection:
    """``dev`` maps every candidate id (the control included) to its per-ticker development results."""
    expected = set(CANDIDATES)
    if set(dev) != expected:
        raise ValueError(f"need results for exactly {sorted(expected)}; got {sorted(dev)}")
    control = dev[CONTROL.id]
    scores = {c.id: score_candidate(dev[c.id], control) for c in COMPLEXITY_ORDER}
    best_price = _pick([c.id for c in COMPLEXITY_ORDER if c.feature_set == "price"], scores)
    best_news = _pick([c.id for c in COMPLEXITY_ORDER if c.feature_set == "news"], scores)
    overall = _pick(
        [c.id for c in COMPLEXITY_ORDER], scores
    )  # lowest S over all eight, ties simplest
    reasons = [
        f"{overall}: S vs {ref} = {scores[overall][ref]:+.4%}, need <= {MATERIALITY:+.2%}"
        for ref in REFERENCES
        if not scores[overall][ref] <= MATERIALITY
    ]
    return Selection(scores, best_price, best_news, overall, not reasons, reasons)
