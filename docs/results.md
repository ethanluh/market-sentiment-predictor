# Results: does the model have predictive edge?

**Status: first real-data evaluation (2026-11-02). The pre-registered gate FAILED. Do not paper trade or
trade this model.** Raw numbers: `docs/results/2026-11-02-5d.json`. Methodology: `docs/testing_the_thesis.md`.

## What was run

`python -m src.eval.run --live --universe config/universe_starter.txt --start 2019-09-01 --horizon 5d`

- 23 liquid large caps across all 11 GICS sectors, 5-day cumulative log-return forecasts (P10/P50/P90), daily bars.
- Features from the shared strict builder (`src/prediction/dataset.py`): price and volume features, a sector
  proximity feature from the correlation graph of the other 22 tickers, VIX regime (classifier refit per fold on
  training rows only), and FinBERT sentiment from the Alpaca/Benzinga archive read at each bar's close.
  The share of bars with at least one article in the 7-day lookback ranges from 63% to 95% by ticker (table below).
- Purged walk-forward: 252-bar training window, 21-bar test blocks, 5-bar embargo, a fresh model per fold.
  Per-ticker model: gradient-boosted quantile regression (200 trees, depth 3). 1,781 feature bars per ticker
  (2019-10-01 to 2026-10-30), of which about 1,512 are scored out of sample (prices snapshot 2026-11-02).
- Two configurations on the same rows: `full` and `price_only` (sentiment, sector, regime, trends and insider
  features neutralised). Baselines are updated every test row: empirical quantiles, EWMA volatility
  (RiskMetrics, normal intervals), momentum, and a zero forecast.

## The gate, fixed in code before the run (`src/eval/gate.py`)

| Criterion | Rule | Result |
|---|---|---|
| Panel (average across tickers per date) | mean loss below the empirical-quantile baseline, one-sided p <= 0.05 | mean relative loss difference +13.6%, p = 1.000. **Fail** (model is worse) |
| Real edge in individual tickers | at least 3 tickers with positive skill and Benjamini-Hochberg q <= 0.10 | 0 of 23. **Fail** |
| Calibration | mean P10-P90 coverage in [0.70, 0.90], every ticker in [0.60, 0.95] | mean 0.641. **Fail** |

## Forecaster comparison (mean pinball loss across tickers; lower is better)

| Forecaster | Mean pinball |
|---|---|
| EWMA volatility baseline | 0.00940 |
| Empirical quantiles baseline | 0.00948 |
| Momentum baseline | 0.01060 |
| **Model (full)** | **0.01077** |
| Zero forecast | 0.01443 |

Skill versus the empirical-quantile baseline is negative for all 23 tickers (`full`: median -13.4%; `price_only`: median -13.3%).
Trading proxy (position = sign of P50, 1 bp costs, non-overlapping 5-day holds): median net Sharpe 0.05
against 0.64 for buy-and-hold, directional hit rate 50.5%.

## Does sentiment (and the graph and regime) add anything over price features alone?

No. `full` versus `price_only`, same rows: panel mean relative loss difference +0.03%, two-sided p = 0.93.
One ticker passes at q <= 0.10 (JNJ), which is about what chance produces across 23 tests.

## Per ticker

| Ticker | News coverage | Skill (full) | Skill (price only) | DM p (one-sided) | q (BH) | P10-P90 coverage |
|---|---|---|---|---|---|---|
| AAPL | 95% | -13.9% | -11.3% | 1.000 | 1.000 | 0.66 |
| AMZN | 95% | -13.4% | -11.6% | 1.000 | 1.000 | 0.66 |
| BAC | 94% | -13.4% | -11.4% | 1.000 | 1.000 | 0.65 |
| CAT | 90% | -13.3% | -12.6% | 1.000 | 1.000 | 0.62 |
| CVX | 93% | -19.0% | -20.9% | 1.000 | 1.000 | 0.64 |
| GOOGL | 95% | -15.2% | -16.9% | 1.000 | 1.000 | 0.64 |
| HD | 92% | -10.6% | -11.1% | 1.000 | 1.000 | 0.66 |
| HON | 86% | -10.5% | -10.3% | 1.000 | 1.000 | 0.63 |
| JNJ | 95% | -9.9% | -14.3% | 1.000 | 1.000 | 0.62 |
| JPM | 95% | -10.7% | -12.3% | 1.000 | 1.000 | 0.68 |
| KO | 89% | -14.6% | -13.6% | 1.000 | 1.000 | 0.64 |
| LIN | 63% | -17.4% | -14.0% | 1.000 | 1.000 | 0.61 |
| LLY | 94% | -12.7% | -12.8% | 1.000 | 1.000 | 0.65 |
| META | 86% | -14.4% | -15.9% | 1.000 | 1.000 | 0.63 |
| MSFT | 95% | -15.3% | -15.7% | 1.000 | 1.000 | 0.64 |
| NEE | 80% | -14.4% | -14.4% | 1.000 | 1.000 | 0.64 |
| NVDA | 95% | -11.0% | -10.2% | 1.000 | 1.000 | 0.63 |
| PG | 78% | -11.8% | -13.3% | 1.000 | 1.000 | 0.61 |
| PLD | 65% | -13.0% | -13.3% | 1.000 | 1.000 | 0.66 |
| UNH | 90% | -17.1% | -17.6% | 1.000 | 1.000 | 0.63 |
| V | 94% | -13.0% | -12.0% | 1.000 | 1.000 | 0.62 |
| WMT | 95% | -14.1% | -12.9% | 1.000 | 1.000 | 0.65 |
| XOM | 95% | -13.9% | -13.6% | 1.000 | 1.000 | 0.67 |

## What this does and does not show

- **This pipeline, as built, has no edge.** The model is worse than a rolling empirical-quantile baseline and
  worse than momentum, and its intervals are overconfident (about 64% coverage for a nominal 80%), which is the
  signature of overfitting 11 features on 252 rows per ticker.
- **The only forecastable thing in these data is the scale of returns, not their direction.** The best forecaster
  is the plain EWMA-volatility baseline, and the directional hit rate is 50.5%.
- **It does not prove sentiment is worthless.** It shows that this feature set and model class extracted nothing
  from this archive. The news is Benzinga only through Alpaca, old articles carry no source name so the
  credibility weighting does little, and coverage is uneven (63% to 95% of bars).
- **One configuration, one run.** The model was not tuned. It must not be tuned on this evaluation window: doing so
  would turn this test set into a training set, and any later pass would need a fresh holdout.
- **Price snapshots are local.** `data/cache/prices` is git-ignored, so `--snapshot 2026-11-02` reproduces this
  run only on the machine that made it; the committed JSON holds the resulting numbers.

## Decision and next steps

Paper trading stays off (`--submit` is not to be used with this model). The next attempt is a new, pre-registered
hypothesis, not a retune of this one: select the model on 2020-2023 only, test once on 2024-2026, use the
EWMA-volatility baseline (not the empirical quantiles) as the bar to beat, and re-run the same gate. Candidate
changes are a pooled model across tickers, volatility-normalised targets, and strong regularisation.
