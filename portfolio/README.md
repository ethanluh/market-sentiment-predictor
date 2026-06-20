# Portfolio assets

A one-page visual **case study** of `market-sentiment-predictor`, suitable for a
portfolio / Upwork listing. It folds three things into one image:

1. **Architecture** — the five-stage pipeline (ingestion → FinBERT sentiment →
   graph diffusion → quantile regression → backtest/serve).
2. **Sample forecast** — the P10/P50/P90 log-return cone the model emits at the
   `1h` / `1d` / `5d` horizons, drawn as a price band around spot.
3. **Backtest** — the model's mean pinball loss vs. the momentum / ARIMA / GARCH
   baselines, with `[P10,P90]` coverage and Sharpe.

## Generate

```bash
pip install -r portfolio/requirements.txt
python portfolio/generate_case_study.py
# -> portfolio/case_study.png  and  portfolio/case_study.pdf
```

With no data files present it renders **clearly-labelled illustrative numbers**
so the layout is complete out of the box.

## Use real numbers

Drop either/both JSON files in `portfolio/data/` and re-run; they are picked up
automatically and the "illustrative" caption disappears for that panel.

**`portfolio/data/prediction.json`** — one prediction (log-returns), e.g. the
output of `model.predict_one(...)` per horizon:

```json
{
  "ticker": "AAPL",
  "as_of": "2024-01-05",
  "spot": 185.00,
  "quantiles": {
    "1h": {"p10": -0.004, "p50": 0.0006, "p90": 0.005},
    "1d": {"p10": -0.014, "p50": 0.002,  "p90": 0.017},
    "5d": {"p10": -0.031, "p50": 0.004,  "p90": 0.038}
  }
}
```

**`portfolio/data/backtest.json`** — mirrors
`src.prediction.backtest.BacktestResult` (only these keys are read):

```json
{
  "horizon": "1d",
  "pinball_loss": {"p10": 0.0061, "p50": 0.0074, "p90": 0.0059},
  "coverage": {"p10_p90": 0.81},
  "sharpe": 1.34,
  "n_predictions": 248,
  "baseline_pinball": {
    "momentum": {"p10": 0.0079, "p50": 0.0091, "p90": 0.0077},
    "arima":    {"p10": 0.0074, "p50": 0.0086, "p90": 0.0072},
    "garch":    {"p10": 0.0070, "p50": 0.0083, "p90": 0.0069}
  }
}
```

A quick way to produce real numbers locally (needs API keys + full
`requirements.txt`):

```bash
python scripts/train_model.py --ticker AAPL --start 2022-01-01 --out models/aapl.joblib
python -m src.prediction.backtest --ticker AAPL --start 2023-01-01 --end 2024-01-01
```

then transcribe the printed `BacktestResult` / prediction into the JSON above.
