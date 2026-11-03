# Protocol H2: a pooled, volatility-normalised model, tested once

**Registered before any development run.** No result of any candidate below existed when this was written, and the
holdout has not been opened. Changes after the first development run are amendments (see the end), never edits.
Background: the first evaluation (`docs/results.md`) failed the gate. The best forecaster was the plain EWMA volatility
baseline, so the scale of returns is forecastable and their direction was not.

## Hypotheses

- **H2a.** A model that forecasts the volatility-standardised 5-day return from features, pooled over the 23 tickers,
  beats both volatility baselines (normal quantiles, and empirical standardised quantiles) on a holdout.
- **H2b.** The best pipeline that uses the news-derived features beats the best price-only pipeline on that holdout.
  This tests "the best news pipeline against the best price pipeline", which can differ in model class; matched-class
  pairs (for example `gb-50-news` against `gb-50-price`) are reported alongside, informationally.

If H2a fails, H2b is moot. Either answer is reported.

## Data

- Universe: `config/universe_starter.txt` (23 tickers). Prices: Yahoo adjusted daily closes from 2019-09-01, saved as a
  dated snapshot (`data/cache/prices`). News: the Alpaca archive from 2020-01-01, scored with FinBERT at full 512 tokens.
  Features come only from `src/prediction/dataset.py` (strict, bar-close aligned, with truncation-leakage tests that
  include negative controls).
- **Pooled rows start at 2020-01-02 for every candidate** (the first news date), so all candidates see the same rows; the
  October to December 2019 rows, which have no news archive, are dropped for all of them.
- Sparse-news tickers (LIN, PLD, NEE, PG) stay in the pool. Their coverage is reported, not hidden.

## Target and the volatility forecast

`sigma_t` is the RiskMetrics EWMA volatility (lambda 0.94, `src/prediction/volatility.py::ewma_vol`), known at the close of
bar t. The target is `y_t = (r_{t+1} + ... + r_{t+5}) / (sigma_t * sqrt(5))`. The model predicts the 0.1, 0.5 and 0.9
quantiles of `y`, and the forecast in return units is `sigma_t * sqrt(5) *` that quantile. With no features this is
filtered historical simulation (FHS).

## Baselines and the control (names are fixed)

- `garch`: `sigma_t * sqrt(5)` times normal quantiles.
- `fhs` (a baseline, per ticker): `sigma_t * sqrt(5)` times the empirical standardised quantiles of that ticker's own last
  252 fully-realised targets.
- `fhs-pooled` (a candidate, the featureless control): the same, but from the pooled training window of all tickers.
- Reference only: `empirical_quantile`, `momentum`, `zero`.

Expected, and measured on iid simulations (`tests/eval/test_baselines.py`): at the 5-day horizon the sum of five returns is
close to normal, so `fhs` and `garch` are nearly equal (FHS is about 1% worse from estimation noise on normal data and wins by
about 5% only at 1 step on strongly skewed data). In practice "beat both" therefore mostly means "beat the normal-quantile
baseline", and the model's only room to win is the information in its features.

## Features (all vol-normalised, hence comparable across tickers)

| Name | Definition |
|---|---|
| mom5 | `momentum_5d / (sigma * sqrt(5))` |
| mom20 | `momentum_20d / (sigma * sqrt(20))` |
| ret1 | `return_lag_1 / sigma` |
| volratio | `realized_vol_20d / sigma` |
| volz | `volume_zscore` |
| sent | `sentiment_agg` |
| sectprox | `sector_proximity` |
| regime | `regime_label` (VIX k-means, refit on each training window only) |

Feature set `price` = {mom5, mom20, ret1, volratio, volz}. Feature set `news` = `price` plus {sent, sectprox, regime}.
`estimated_lag_hours` is excluded (near-constant, `docs/graph_model.md`); trends and insider flow are off. Features are
standardised with a scaler fit on the training window only.

## Candidates (fixed; no others will be tried under this protocol)

All are pooled (one model per quantile, fit on the stacked rows of all tickers), use random seed 0, and repair quantile
crossing by sorting each row's three quantiles.

| id | estimator |
|---|---|
| `fhs-pooled` | no features (the control, not eligible to be frozen) |
| `lq-a0.01-{price,news}` | `QuantileRegressor(alpha=0.01, solver="highs")` |
| `lq-a0.001-{price,news}` | `QuantileRegressor(alpha=0.001, solver="highs")` |
| `gb-50-{price,news}` | `GradientBoostingRegressor(loss="quantile", n_estimators=50, max_depth=2, learning_rate=0.05, min_samples_leaf=200, subsample=0.7)` |
| `gb-100-{price,news}` | the same with `n_estimators=100` |

Eight eligible candidates. Complexity order, simplest first (used only for tie-breaking):
`lq-a0.01-price`, `lq-a0.01-news`, `lq-a0.001-price`, `lq-a0.001-news`, `gb-50-price`, `gb-50-news`, `gb-100-price`,
`gb-100-news`.

## Splits and folds

- The pooled calendar is the union of trading dates. A **rolling window of 504 dates** (all tickers' rows on those dates
  stacked, about 11,600 rows) trains each fold; the model is **refit every 21 dates**, with a **5-date embargo** between
  the last training date and the first test date.
- A row at date *t* is a **development row** if its target's last bar (five bars later) falls on or before 2023-12-31, and a
  **holdout row** if *t* is on or after 2024-01-01. The last five bars of 2023 are in neither set. A 21-date block that
  crosses a boundary scores only its rows inside the set (rows outside are not scored, though they may be trained on).
- The first fold is the first date with 504 + 5 dates of history, so development test dates run from about 2022-01 to
  2023-12 (about 24 blocks). "Select on 2020-2023" means training data from 2020; the test dates are 2022-2023.
- **Frozen** means: estimator and hyperparameters, feature set, feature definitions, window (504), refit cadence (21),
  embargo, seed, and the gate. The fitted parameters, the scaler and the regime classifier are refit on each fold from past
  data only, so during the holdout the model trains on holdout labels as they are revealed, exactly as a live system would.

## Selection (development rows only)

Per candidate and per ticker, the loss of a row is the mean of the three pinball losses in **return units** (the unit the
gate scores). The **relative differential** of a candidate against a reference is `(loss_candidate - loss_reference)`
divided by that ticker's mean reference loss over all its development rows. **S(c, ref)** is the mean over **all**
development test dates of the relative differential averaged across tickers on that date. Negative is better.

1. Rank the eight eligible candidates by `S(c, fhs)`. The **frozen overall winner** is the lowest; candidates within 0.001
   of the lowest are tied and the earliest in the complexity order wins.
2. The **best `price` candidate** and **best `news` candidate** are chosen the same way within their feature set; both are
   also frozen (for H2b). The overall winner is whichever of the two has the lower `S(c, fhs)`.
3. **H2a is rejected on development, and the holdout is not opened,** unless the overall winner has `S <= -0.005` (at least
   0.5% lower loss) against each of `fhs`, `garch` and `fhs-pooled`. In that case the featureless forecasters are the
   answer and the protocol ends there.

Models are trained to minimise pinball loss in standardised units (return-pinball weighted by `1 / (sigma * sqrt(5))`),
while selection and the gate score return-unit pinball, where high-volatility periods weigh more. A candidate can win in
one unit and lose in the other; standardised-unit pinball is reported as a diagnostic.

## The holdout test (run once, on the frozen candidates)

Evaluated per ticker and pooled, with the machinery in `src/eval/` (panel test over dates, Benjamini-Hochberg across
tickers, coverage bands), against `garch` and `fhs`. **H2a passes only if all of:**

1. For each of the two baselines: the panel mean relative loss difference is below zero and the one-sided panel p-value is
   at most 0.05 (the larger p of the Newey-West HAC and block-bootstrap tests).
2. At least 3 tickers have positive skill against both baselines and a Benjamini-Hochberg q-value of at most 0.10 against
   both (BH is run separately over the 23 p-values of each baseline; a ticker must pass under both). Per-ticker p-values use
   the same combined HAC and block-bootstrap test as the panel. The false-discovery guarantee for this intersection of two
   BH runs is not formally established; it is conservative in practice.
3. Mean P10-P90 coverage in [0.70, 0.90] and every ticker in [0.60, 0.95].
4. A ticker with fewer than 500 holdout scored rows is removed from every criterion, the panel included, and listed. If more
   than 25% of tickers (more than 5 of 23) are removed, the gate fails.

**H2b passes only if** the frozen best-`news` pipeline beats the frozen best-`price` pipeline on the holdout (panel mean
relative difference below zero, one-sided p at most 0.05). If the overall winner is a `price` candidate, H2a is still
tested on it and H2b is still tested on the pair. The holdout evaluates the frozen set (overall winner, best-`price`,
best-`news`, the baselines) in a single run; that run is the one shot.

Passing H2a permits one thing: logging weekly shadow forecasts (no orders) to build a prospective record. It does not
permit paper or live trading.

## One shot

`src/eval/protocol.py` (a later change, before the holdout) freezes the chosen candidates with a hash and the git commit,
refuses to run the holdout without a matching freeze, and refuses a second run. Re-running for any reason requires a new
hypothesis id and is reported as a second test, not as a replacement.

## Code that must exist before the development run (implemented in later changes)

- The pooled model, with exactly the estimators above, and a pooled walk-forward evaluator that emits per-ticker results in
  return units (`src/prediction/vol_scaled.py`, `src/eval/pooled.py`).
- The gate generalised to several reference baselines, a minimum-usable-ticker fraction, and a model-versus-model panel test
  for H2b (`src/eval/gate.py`, `src/eval/panel.py`).
- The freeze and one-shot guard (`src/eval/protocol.py`).

## What will be reported regardless of outcome

The full development ranking (all eight candidates and the control), the frozen choices, the holdout numbers for each, both
baselines, per-ticker results, news coverage, the H2b comparison, and the standardised-unit diagnostic. A failure is written
up as plainly as a pass.

## Known limitations

- **The holdout is not perfectly virgin for the question.** The first model was evaluated over 2019-2026 and its failure is
  known, which can shape what one thinks to try. No model from this protocol has seen the holdout, the candidate list is
  fixed above, and the stronger evidence is the prospective shadow record that a pass would start.
- Eight candidates on development invite selection bias; the holdout is the only unbiased estimate and it is used once.
- Per-ticker and panel p-values assume a persistent but not extremely persistent loss differential; in simulation the
  combined test is somewhat liberal for AR(0.9) and above.
- Pooling assumes the standardised return distribution is shared across tickers; sector-specific effects are out of scope.
- The news is Benzinga only (through Alpaca), old articles have no source name, and coverage is uneven.

## Amendments

- **A1, 2026-11-02 (before any development run; the holdout is unopened).** After an independent review of the first draft:
  fixed the naming of the per-ticker baseline `fhs` versus the pooled control `fhs-pooled`; defined the 504-date window,
  the development and holdout row sets, the embargo between them, and the first fold; defined the selection statistic, its
  normalisation, the tie tolerance and tie order, and a 0.5% materiality threshold; defined H2b as a comparison of two frozen
  pipelines evaluated in the single holdout run; specified the multiple-testing procedure over two baselines; specified that
  excluded tickers leave the panel too and capped them at 25%; made per-ticker p-values use the combined HAC and block-bootstrap
  test (a code change made with this amendment); added the estimator settings and seed; stated the standardised-versus-return
  loss mismatch.
