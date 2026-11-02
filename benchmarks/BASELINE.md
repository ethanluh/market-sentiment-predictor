# Benchmark baseline

Run: `pytest benchmarks/ --benchmark-only`. Timings are machine-specific; compare runs from
the same machine (`--benchmark-save` / `--benchmark-compare`), not these numbers across machines.

Recorded 2026-11-02 on the maintainer's Linux dev machine, Python 3.14, minimum time per call:

| Benchmark | Min |
|---|---|
| `aggregate`, 1000 articles | 0.5 ms |
| sector graph + proximity, 20 tickers | 2.5 ms |
| `estimate_lag`, uncached (500 matrix exponentials) | 4.4 ms |
| `QuantileReturnModel` fit + predict, linear, 400 rows | 62 ms |
| `build_feature_matrix`, 100 rows, 5 peers | 101 ms |
| `build_feature_matrix`, 250 rows, 5 peers | 251 ms |
| `QuantileReturnModel` fit + predict, gbr, 400 rows | 1.1 s |

Reading it: feature construction costs about 1 ms per row and scales linearly (measured with
the diffusion lag already cached, as in production). With the gbr backend, model fitting
dominates: it trains one 200-tree estimator per quantile, already in scikit-learn's compiled
code. With the linear backend, feature construction costs more than fitting. A walk-forward
over N folds refits N times, so on gbr with 400-row training slices it takes roughly N x 1 s.
