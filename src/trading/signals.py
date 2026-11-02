"""Turn per-ticker quantile forecasts into target portfolio weights. Pure; no I/O."""

from __future__ import annotations

from dataclasses import dataclass

from src.trading.config import TradingConfig


@dataclass(frozen=True)
class Forecast:
    ticker: str
    p10: float
    p50: float
    p90: float


def _validate(forecasts: list[Forecast], cfg: TradingConfig) -> None:
    seen: set[str] = set()
    for f in forecasts:
        if f.ticker not in cfg.universe:
            raise ValueError(f"forecast for {f.ticker!r} is outside the universe")
        if f.ticker in seen:
            raise ValueError(f"duplicate forecast for {f.ticker!r}")
        seen.add(f.ticker)
        if not f.p10 <= f.p50 <= f.p90:
            raise ValueError(f"{f.ticker}: quantiles not ordered ({f.p10}, {f.p50}, {f.p90})")


def decide_targets(forecasts: list[Forecast], cfg: TradingConfig) -> dict[str, float]:
    """
    Equal-weight long-only targets over names whose forecast clears the edge threshold.

    Weight per name is ``min(max_position_weight, gross / n_selected)`` where ``gross`` is
    ``max_gross_exposure / (1 + buy_slippage_buffer)``, so a full plan still passes the cash
    check and a thin selection leaves cash.
    Names not selected get no entry (target weight 0 is implied by absence).
    """
    _validate(forecasts, cfg)
    selected = sorted(
        f.ticker for f in forecasts if f.p50 > cfg.edge_threshold and f.p10 >= cfg.min_p10
    )
    if not selected:
        return {}
    gross = cfg.max_gross_exposure / (1.0 + cfg.buy_slippage_buffer)  # keep cash for slippage
    weight = min(cfg.max_position_weight, gross / len(selected))
    return {ticker: weight for ticker in selected}
