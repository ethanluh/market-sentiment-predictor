# H3 development result

Protocol: `docs/protocol_h3.md` (amendments A1 and A2). Run: `python -m src.eval.h3 dev --universe config/universe_starter.txt --snapshot 2026-11-02`.
Data: 23 tickers, price snapshot 2026-11-02, 1006 scored bars from 2020-01-02 to 2023-12-31. Raw output:
`docs/results/h3/dev.json`; every number below is read from it.

## Result

The selected forecaster is `ewma94`. It meets H3a and fails H3b on development, so under the protocol **the holdout
stays closed and H3 is not confirmed**. No freeze was made and the 2024 onward data was not touched.

| Forecaster | VV ratio | 95th pct | Sharpe diff | 5th pct | H3a (ratio <= 0.80, 95th pct < 1) | H3b (5th pct >= -0.10) | Mean exposure | Turnover per bar |
|---|---|---|---|---|---|---|---|---|
| `ewma94` | 0.572 | 0.771 | +0.079 | -0.166 | pass | fail | 0.858 | 0.0119 |
| `ewma97` | 0.695 | 0.841 | -0.008 | -0.198 | pass | fail | 0.864 | 0.0062 |
| `rolling63` | 0.797 | 0.909 | -0.069 | -0.255 | pass | fail | 0.861 | 0.0047 |

`VV ratio` is the relative dispersion of 21-bar realised portfolio volatility against buy-and-hold. Sharpe diff is net of
10 bps per unit of turnover, scaled strategy minus buy-and-hold.

## What the numbers say

- **Stability (H3a) is large and holds for all three forecasters.** `ewma94` cuts the dispersion of realised volatility to
   0.57 of buy-and-hold, with a 95th percentile of 0.77. The ordering is `ewma94`, then `ewma97`, then `rolling63`.
- **H3b fails on width, not on a loss.** The point estimate of the Sharpe difference for `ewma94` is +0.079
  (strategy 0.980, buy-and-hold 0.902), but its 5th percentile is -0.166, which is 0.24
  below the point estimate. With 1006 bars the paired Sharpe difference is too noisy to certify a margin of
  -0.10 unless the true edge is well above zero. The data neither show that Sharpe suffers nor establish that it does not.
- **Costs are not the issue.** Turnover is 0.0119 per bar. The Sharpe difference at three cost levels:

| Forecaster | 0 bps | 10 bps (protocol) | 20 bps |
|---|---|---|---|
| `ewma94` | +0.099 | +0.079 | +0.058 |
| `ewma97` | +0.002 | -0.008 | -0.018 |
| `rolling63` | -0.062 | -0.069 | -0.076 |

- **Drawdown and worst month improve, even after discounting the lower exposure.** The matched column scales buy-and-hold to
  the strategy's mean exposure; it is informational, not a pass criterion.

| Forecaster | Max drawdown | Matched | Buy-and-hold | Worst 21 bars | Matched | Buy-and-hold |
|---|---|---|---|---|---|---|
| `ewma94` | -19.5% | -28.2% | -32.3% | -18.7% | -27.4% | -31.4% |
| `ewma97` | -23.0% | -28.3% | -32.3% | -22.1% | -27.6% | -31.4% |
| `rolling63` | -25.4% | -28.2% | -32.3% | -24.5% | -27.5% | -31.4% |

- **Single names (informational point estimates).** Minimum, median and maximum across the 23 tickers:

| Forecaster | VV ratio | Sharpe diff | Names with Sharpe diff below 0 |
|---|---|---|---|
| `ewma94` | 0.451 / 0.581 / 0.806 | -0.136 / +0.011 / +0.193 | 11 of 23 |
| `ewma97` | 0.570 / 0.701 / 0.866 | -0.173 / -0.030 / +0.151 | 16 of 23 |
| `rolling63` | 0.650 / 0.801 / 1.040 | -0.203 / -0.053 / +0.109 | 20 of 23 |

## Caveats

- Development is 2020 to 2023 and contains the March 2020 crash, where vol targeting is expected to look good. This is
  one period and one run.
- The 23 sleeves are strongly correlated, so these are not 23 independent confirmations.
- H3c is a holdout statistic and was not computed.
- The matched-exposure columns are descriptive only. The protocol's pass rule uses VV ratio and Sharpe.

## Decision

The pre-registered rule is applied as written: the holdout stays closed. The protocol's H3b margin of -0.10 was chosen
without a power calculation, and this run shows it is too tight for a four-year daily sample. Relaxing it now would be a
choice made after seeing development results. That is permitted only as a recorded amendment, and it should be the
owner's decision, not a silent one. Nothing here supports paper trading, and `--submit` stays off.
