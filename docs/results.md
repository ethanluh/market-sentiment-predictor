# Backtest Results — Predictive Edge Validation

> **Status: NOT YET POPULATED.** This is the template/runbook for the ship-gating
> validation. It must be filled in with real walk-forward backtest numbers (which
> require live market/news data and the full runtime) **before** the model is
> shipped. If the model does not beat the baselines, that outcome is the ship
> decision — record it here rather than shipping.

See `docs/testing_the_thesis.md` for the underlying methodology.

## How to reproduce

```bash
pip install -r requirements.txt          # full runtime (FinBERT/torch)

# (optional, for sentiment-driven features) build a point-in-time news archive
python scripts/build_news_archive.py --ticker AAPL --source alphavantage \
    --start 2022-01-01 --end 2024-01-01

# walk-forward backtest, model vs. baselines (pinball loss)
python -m src.prediction.backtest --ticker AAPL --start 2023-01-01 --end 2024-01-01 \
    --use-trends --use-insider-flow --use-vix
```

Repeat per ticker in the private-beta universe.

## Acceptance criteria (ship gate)

For each ticker × horizon the model should:

- **Pinball loss ≤** the best of momentum / ARIMA / EWMA-GARCH baselines.
- **Interval coverage** of the P10–P90 band close to nominal 80% (e.g. 70–90%).
- **Directional hit-rate** > 50% (P50 sign).

## Results

Fill in from the backtest output (one row per ticker × horizon).

| Ticker | Horizon | Model pinball | Best baseline | Best baseline pinball | P10–P90 coverage | Dir. hit-rate | Sharpe |
|--------|---------|---------------|---------------|-----------------------|------------------|---------------|--------|
| _TBD_  | 1d      | _TBD_         | _TBD_         | _TBD_                 | _TBD_            | _TBD_         | _TBD_  |

## Conclusion

_TBD — does the model demonstrate edge over baselines? Ship / no-ship decision and rationale._
