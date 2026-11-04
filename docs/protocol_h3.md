# Protocol H3: volatility targeting, tested once

**Registered before any development run.** No result of any rule below existed when this was written, and the holdout has
not been opened. Changes after the first development run are amendments (see the end), never edits.
Background: two evaluations (`docs/results.md`, `docs/results_h2.md`) found one supported signal, EWMA volatility. Direction
models and news features added nothing. H3 asks whether that forecast is useful for sizing exposure.

## Hypotheses

- **H3a (stability).** Scaling each position by `target_vol / forecast_vol` lowers the relative dispersion of the realised
  portfolio volatility (defined below) to at most 0.80 times that of unscaled buy-and-hold.
- **H3b (non-inferiority).** The scaled portfolio's net-of-cost annualised Sharpe ratio is not worse than buy-and-hold's by
  more than 0.10.
- **H3c (forecast value, informational).** The frozen volatility forecaster has a lower `VV` ratio than `rolling63` on the
  holdout (point estimate and the same bootstrap, the strategies compared at their own frozen targets). If `rolling63` is
  the frozen forecaster, H3c is reported as not applicable. It is not part of the pass rule.

H3 passes only if H3a and H3b both hold on the holdout. Either answer is reported.

## Data

- Universe: `config/universe_starter.txt`, 23 tickers. SPY, named there as a benchmark, is neither traded nor used here. Prices: Yahoo adjusted daily
  closes from 2019-09-01 as a dated snapshot (`data/cache/prices`). No news is used.
- Daily simple returns `r_t = close_t / close_{t-1} - 1`. Volatility forecasts use the log return.
- A ticker is available on a bar when its forecast and its next-bar return are both finite. On every scored bar at least 20
  tickers must be available. Fewer raises; nothing is filled. A bar is named by its return date (the bar `t+1` that earns the
  weight set at `t`), so a weight set on 2023-12-29 earns the first holdout return and belongs to the holdout.

## The rule

For ticker `i` at the close of bar `t`, `sigma_ann = sqrt(252) * forecast_i,t`, where the forecast uses returns up to and
including bar `t`. The weight is `w_i,t = min(1, target / sigma_ann)`, held through bar `t+1`. There is no leverage and no
shorting; unused weight earns zero. The portfolio return on bar `t+1` is the mean over available tickers of
`w_i,t * r_i,t+1` minus cost. A ticker's first available bar has `w_i,t-1 = 0`, so its entry is charged. At the
split the last development weight is carried into the holdout, not reset.

- Cost: `10 bps * |w_i,t - w_i,t-1|` per ticker per bar, averaged the same way. There is no rebalance band. Sensitivity runs
  at 0 and 20 bps are reported.
- Buy-and-hold: `w = 1` for every ticker every bar (equal weight, daily rebalanced, zero cost). The zero cost favours the
  baseline, which is the conservative direction.
- `target` is fixed per forecaster as the median of its annualised forecast over all dev dates and tickers, and is frozen.

## Forecasters (names are fixed; the order is the registered priority for ties, not a complexity claim)

1. `ewma94`: RiskMetrics EWMA, lambda 0.94 (`src/prediction/volatility.py::ewma_vol`).
2. `ewma97`: the same with lambda 0.97.
3. `rolling63`: square root of the mean squared log return over the last 63 bars (uncentred, like `ewma_vol`), NaN until 63
   returns exist. `ewma_vol` is NaN until 20 returns exist (its `min_obs`). All three forecast from log returns.

## Metrics

- **Relative dispersion** `VV = std(RV21) / mean(RV21)` over the scored dates, where `RV21_t` is the square root of the mean
  squared portfolio return over the 21 bars ending at `t`. Windows never straddle the dev/holdout split: the first
  holdout `RV21` needs 21 holdout bars. The standard deviation uses ddof 1.
- Net Sharpe: `sqrt(252) * mean / std` of daily net portfolio returns.
- Also reported: maximum drawdown, worst 21-bar return, turnover, cost drag, mean exposure, and per-ticker VV and Sharpe.
  Maximum drawdown and worst month are compared with buy-and-hold scaled to the strategy's mean exposure, since lower
  exposure alone lowers them. `VV` and Sharpe are scale-free, so an exposure-matched control cannot differ on them.

## Inference

Paired moving-block bootstrap over the scored bars: block length 21, block starts uniform over the first `n - 20` positions
(no wrap), `ceil(n / 21)` blocks concatenated and truncated to `n`, 2000 replicates, `numpy.random.default_rng(0)`. Each
replicate applies one set of resampled indices to the strategy and buy-and-hold series jointly. `RV21` is computed once on
the original bars and then resampled as a series (so no window is spliced across a block join); Sharpe resamples the daily
net returns.

- H3a passes if the point estimate of `VV_strategy / VV_buyhold` is at most 0.80 and the 95th percentile of the replicate
  ratios is below 1.0.
- H3b passes if the 5th percentile of the replicate Sharpe differences is at least -0.10. This is a deliberately modest
  non-inferiority bar, not a claim of improvement.

## Splits and selection

- Development: 2020-01-02 to 2023-12-31. Holdout: 2024-01-01 to the last bar of the snapshot. The rule has no fitted
  parameter other than the frozen `target`, and every forecast uses only bars up to `t`, so no embargo is applied; the
  `RV21` windows are kept inside each period as stated above.
- Selection on development: find the lowest `VV` ratio against buy-and-hold; among candidates whose ratio is within 0.02 of
  it, the first in the registered order wins. The selection proceeds only if the winner meets the full H3a and H3b rules
  (the same point and bootstrap criteria as on the holdout) on development; otherwise the holdout stays closed and the
  result is documented.
- One shot: `src/eval/protocol.py` records the frozen choice, the `target`, the hash of every `.py` under `src/`, this
  document and the universe, and refuses a dirty tree, a changed file, an uncommitted freeze, or a second run. Files:
  `docs/results/h3/{dev,freeze,holdout_run,holdout}.json`. The guard is procedural, not tamper-proof.

## What will be reported regardless of outcome

All three forecasters on development, the frozen rule on the holdout, the cost sensitivity, per-ticker results, the
drawdown comparison, and H3c.

## Known limitations

- Development (2020 to 2023) contains the March 2020 crash, where vol targeting is expected to look good. The holdout has a
  different regime mix; a pass on development alone proves nothing.
- The calendar period of the holdout is known to the researcher. No design choice here has used it.
- The 23 sleeves are highly correlated, so the effective sample is the daily series, not 23 independent tests.
- Long-only with a cap of 1 only de-risks: mean exposure falls, and any return given up is in the Sharpe difference.
- Daily closes only. Execution at the next open, borrow, taxes and intraday gaps are not modelled.

## Amendments

None yet.
