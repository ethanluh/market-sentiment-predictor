"""Generate a one-page visual case study for the market-sentiment-predictor.

Produces ``case_study.png`` and ``case_study.pdf`` combining three panels that
tell the whole story of the project at a glance:

  1. Architecture  -- the five-stage pipeline as boxes + arrows.
  2. Sample forecast -- the P10 / P50 / P90 log-return "fan" the model emits at
     the 1h / 1d / 5d horizons, rendered as a price cone around the spot price.
  3. Backtest      -- mean pinball loss of the model vs. the momentum / ARIMA /
     GARCH baselines, plus the empirical [P10, P90] coverage.

Data is *injectable*. Drop real output next to this file and it is used
automatically; otherwise clearly-labelled illustrative numbers are shown:

  portfolio/data/prediction.json   -- see PredictionSpec below
  portfolio/data/backtest.json     -- mirrors src.prediction.backtest.BacktestResult

Regenerate with::

    pip install -r portfolio/requirements.txt
    python portfolio/generate_case_study.py

See portfolio/README.md for how to produce the JSON from a real train+backtest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
HORIZONS: tuple[str, ...] = ("1h", "1d", "5d")

# --- palette -----------------------------------------------------------------
INK = "#0f172a"
MUTED = "#64748b"
ACCENT = "#2563eb"
ACCENT_SOFT = "#bfdbfe"
GOOD = "#059669"
GRID = "#e2e8f0"
STAGE_FILL = "#f1f5f9"


@dataclass
class PredictionSpec:
    """Format of portfolio/data/prediction.json."""

    ticker: str = "AAPL"
    as_of: str = "2024-01-05"
    spot: float = 185.00
    # horizon -> {p10, p50, p90} of the forward LOG-return
    quantiles: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "1h": {"p10": -0.004, "p50": 0.0006, "p90": 0.005},
            "1d": {"p10": -0.014, "p50": 0.002, "p90": 0.017},
            "5d": {"p10": -0.031, "p50": 0.004, "p90": 0.038},
        }
    )
    illustrative: bool = True


@dataclass
class BacktestSpec:
    """Subset of src.prediction.backtest.BacktestResult we visualise."""

    horizon: str = "1d"
    pinball_loss: dict[str, float] = field(
        default_factory=lambda: {"p10": 0.0061, "p50": 0.0074, "p90": 0.0059}
    )
    coverage: dict[str, float] = field(default_factory=lambda: {"p10_p90": 0.81})
    sharpe: float = 1.34
    n_predictions: int = 248
    baseline_pinball: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "momentum": {"p10": 0.0079, "p50": 0.0091, "p90": 0.0077},
            "arima": {"p10": 0.0074, "p50": 0.0086, "p90": 0.0072},
            "garch": {"p10": 0.0070, "p50": 0.0083, "p90": 0.0069},
        }
    )
    illustrative: bool = True


def _load(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_prediction() -> PredictionSpec:
    raw = _load(DATA_DIR / "prediction.json")
    if raw is None:
        return PredictionSpec()
    raw.setdefault("illustrative", False)
    return PredictionSpec(**raw)


def load_backtest() -> BacktestSpec:
    raw = _load(DATA_DIR / "backtest.json")
    if raw is None:
        return BacktestSpec()
    raw.setdefault("illustrative", False)
    return BacktestSpec(**raw)


def _mean_pinball(d: dict[str, float]) -> float:
    vals = [d[k] for k in ("p10", "p50", "p90") if k in d]
    return float(np.mean(vals)) if vals else float("nan")


# --- panel 1: architecture ----------------------------------------------------
def draw_architecture(ax: plt.Axes) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")
    ax.set_title(
        "Five-stage pipeline  ·  multi-source ingestion → FinBERT → graph diffusion → quantile regression",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        loc="left",
        pad=8,
    )

    stages = [
        ("1  Ingestion", "prices · news · SEC\nForm 4 · social · trends"),
        ("2  Sentiment", "FinBERT score in [-1,1]\ncredibility + recency wt."),
        ("3  Graph diffusion", "heat kernel on Laplacian\n→ impact-lag, sector prox."),
        ("4  Quantile model", "P10/P50/P90 log-return\n1h · 1d · 5d, VIX regime"),
        ("5  Backtest + serve", "walk-forward harness\nFastAPI /predict"),
    ]
    n = len(stages)
    gap = 2.2
    w = (100 - gap * (n - 1)) / n
    h = 15
    y = 5
    centers = []
    for i, (title, body) in enumerate(stages):
        x = i * (w + gap)
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.6,
            edgecolor=ACCENT,
            facecolor=STAGE_FILL,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h - 3.2,
            title,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=ACCENT,
        )
        ax.text(
            x + w / 2,
            y + h - 6.4,
            body,
            ha="center",
            va="top",
            fontsize=8.2,
            color=MUTED,
            linespacing=1.35,
        )
        centers.append((x + w, x, y + h / 2))

    for i in range(n - 1):
        right = centers[i][0]
        nxt_left = centers[i + 1][1]
        ymid = centers[i][2]
        ax.add_patch(
            FancyArrowPatch(
                (right + 0.1, ymid),
                (nxt_left - 0.1, ymid),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color=INK,
            )
        )


# --- panel 2: forecast cone ----------------------------------------------------
def draw_forecast(ax: plt.Axes, pred: PredictionSpec) -> None:
    spot = pred.spot
    xs = [0]
    p10 = [spot]
    p50 = [spot]
    p90 = [spot]
    labels = ["spot"]
    for i, hz in enumerate(HORIZONS, start=1):
        q = pred.quantiles[hz]
        xs.append(i)
        p10.append(spot * np.exp(q["p10"]))
        p50.append(spot * np.exp(q["p50"]))
        p90.append(spot * np.exp(q["p90"]))
        labels.append(hz)

    ax.fill_between(xs, p10, p90, color=ACCENT_SOFT, alpha=0.85, label="P10–P90 band")
    ax.plot(xs, p50, color=ACCENT, linewidth=2.2, marker="o", markersize=4, label="P50 (median)")
    ax.plot(xs, p10, color=ACCENT, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.plot(xs, p90, color=ACCENT, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(spot, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.8)

    # annotate widening uncertainty at the final horizon
    ax.annotate(
        f"P90  {p90[-1]:.2f}",
        (xs[-1], p90[-1]),
        textcoords="offset points",
        xytext=(6, 2),
        fontsize=8,
        color=ACCENT,
    )
    ax.annotate(
        f"P10  {p10[-1]:.2f}",
        (xs[-1], p10[-1]),
        textcoords="offset points",
        xytext=(6, -8),
        fontsize=8,
        color=ACCENT,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("price ($)", fontsize=9, color=MUTED)
    ax.set_title(
        f"Sample forecast  ·  {pred.ticker}  ·  as of {pred.as_of}",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        loc="left",
    )
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


# --- panel 3: backtest vs baselines -------------------------------------------
def draw_backtest(ax: plt.Axes, bt: BacktestSpec) -> None:
    model = _mean_pinball(bt.pinball_loss)
    names = ["model"] + list(bt.baseline_pinball.keys())
    vals = [model] + [_mean_pinball(v) for v in bt.baseline_pinball.values()]
    colors = [GOOD] + [MUTED] * (len(names) - 1)

    bars = ax.bar(names, vals, color=colors, width=0.6, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )

    ax.set_ylabel("mean pinball loss  (lower = better)", fontsize=9, color=MUTED)
    ax.set_title(
        f"Backtest  ·  {bt.horizon} horizon  ·  n={bt.n_predictions} walk-forward preds",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        loc="left",
    )
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_ylim(0, max(vals) * 1.25)

    cov = bt.coverage.get("p10_p90") or next(iter(bt.coverage.values()), float("nan"))
    ax.text(
        0.99,
        0.97,
        f"[P10,P90] coverage  {cov:.0%}\nSharpe  {bt.sharpe:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=STAGE_FILL, edgecolor=GRID),
    )


def build_figure() -> plt.Figure:
    pred = load_prediction()
    bt = load_backtest()

    fig = plt.figure(figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.25], hspace=0.34, wspace=0.22,
        left=0.06, right=0.95, top=0.86, bottom=0.10,
    )

    fig.text(
        0.06, 0.955, "market-sentiment-predictor",
        fontsize=20, fontweight="bold", color=INK,
    )
    fig.text(
        0.06, 0.915,
        "News-driven equity forecasting: financial NLP + graph-theoretic information diffusion + quantile regression over returns",
        fontsize=10.5, color=MUTED,
    )

    ax_arch = fig.add_subplot(gs[0, :])
    draw_architecture(ax_arch)
    ax_fc = fig.add_subplot(gs[1, 0])
    draw_forecast(ax_fc, pred)
    ax_bt = fig.add_subplot(gs[1, 1])
    draw_backtest(ax_bt, bt)

    note_bits = []
    if pred.illustrative or bt.illustrative:
        which = []
        if pred.illustrative:
            which.append("forecast")
        if bt.illustrative:
            which.append("backtest")
        note_bits.append(
            "Illustrative numbers (" + " & ".join(which) + "): drop real train/backtest "
            "output into portfolio/data/*.json to regenerate with live results."
        )
    note_bits.append("Python 3.11 · scikit-learn · FinBERT · FastAPI · typed (mypy) · CI")
    fig.text(0.06, 0.025, "   ·   ".join(note_bits), fontsize=8, color=MUTED)

    return fig


def main() -> None:
    fig = build_figure()
    png = HERE / "case_study.png"
    pdf = HERE / "case_study.pdf"
    fig.savefig(png, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
