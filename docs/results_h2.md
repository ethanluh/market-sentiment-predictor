# H2 development result: the holdout stays closed

Protocol: `docs/protocol_h2.md` (pre-registered, amendments A1 and A2 dated before this run). Data: 23 tickers, price
snapshot 2026-11-02, development rows only (test dates from early 2022 through 2023, data after 2023-12-31
removed before selection). Raw output: `docs/results/h2/dev.json`. Code hash at the run: `86223e6dfd7e5e38`.

## Result

**H2a is rejected on development. No candidate clears the pre-registered materiality margin, so no freeze was made and the
2024-2026 holdout was not opened.** This is the outcome the protocol says ends the line: the featureless forecasters are the answer.

S(c, ref) is the mean over development dates of the panel-averaged relative pinball difference of candidate `c` against
`ref` (negative is better; the winner needs at most -0.5% against each of `fhs`, `garch` and `fhs-pooled`).

| Candidate | vs `fhs` (per-ticker FHS) | vs `garch` (normal quantiles) | vs `fhs-pooled` (control) |
|---|---|---|---|
| `lq-a0.01-price` | -1.321% | +0.405% | +0.026% |
| `lq-a0.01-news` | -1.354% | +0.371% | -0.008% |
| `lq-a0.001-price` | -1.095% | +0.636% | +0.256% |
| `lq-a0.001-news` | -0.974% | +0.760% | +0.379% |
| `gb-50-price` | -1.070% | +0.661% | +0.280% |
| `gb-50-news` | -1.248% | +0.481% | +0.100% |
| `gb-100-price` | -0.804% | +0.932% | +0.550% |
| `gb-100-news` | -1.000% | +0.734% | +0.353% |

Overall winner by the pre-registered rule: `lq-a0.01-price` (best price `lq-a0.01-price`, best news `lq-a0.01-news`).
Reasons it fails: lq-a0.01-price: S vs garch = +0.4047%, need <= -0.50%; lq-a0.01-price: S vs fhs-pooled = +0.0260%, need <= -0.50%.

## What the numbers say

- Every candidate beats the per-ticker `fhs` baseline by 0.8% to 1.4%. That gain comes from pooling (more data for the
  quantile shape), not from features: the featureless pooled control `fhs-pooled` matches the candidates within +0.6% to -0.01%.
- No candidate beats `garch`. All are 0.4% to 0.9% worse than EWMA volatility with normal quantiles. At a 5-day horizon the
  sum of five returns is close to normal, so an empirical shape adds estimation noise rather than information.
- Sentiment, sector proximity and regime add nothing measurable: the best news candidate differs from the best price
  candidate by 0.03% of loss, far under the 0.5% margin. This repeats the first evaluation (`docs/results.md`), where
  full and price-only models differed by 0.03%.
- More flexible models do worse (`gb-100-price` is the worst candidate against every reference), the usual sign of fitting noise.

## Caveats

- The result is on a 2022-2023 test window of one market regime; it is a negative on this window, not a proof that
  features never help.
- One run of eight preset candidates; no tuning was done after seeing the numbers.
- The winner `lq-a0.01-price` was picked by the tie rule (within 0.001 of the lowest S, simplest first); the lowest raw S
  was `lq-a0.01-news`. The decision does not depend on it: neither clears the margin.

## Decision

The directional model is not supported by this data. `--submit` (paper trading) stays off for it. What the evidence does
support is the volatility forecast itself: EWMA volatility with normal quantiles (`garch`) is the strongest forecaster in
both evaluations. Possible next steps, each needing its own pre-registered hypothesis and a fresh holdout: a volatility-only
product (vol-targeted sizing), different information (intraday or earnings-event data), or longer horizons.
