# Upwork Portfolio Entry

## Project Title

**News-Driven Stock Return Forecasting with NLP & Graph Models** *(60 chars)*

*(Alternates, all under 70 chars:)*
- *Quantitative Stock Forecasting: FinBERT + Graph + Quantiles* (59)
- *News-Driven Stock Return Prediction (NLP + Graph + Quantiles)* (61)

---

## My Role

**Sole Developer & ML Engineer** — designed and built the entire system end to end:
data architecture, the five-stage modeling pipeline, the backtesting harness, the
serving API, and the test/CI infrastructure. Responsible for all technical
decisions, from the choice of a quantile (rather than point) forecasting target
to the graph-theoretic model of how information propagates into price.

---

## Project Description

A production-grade pipeline that predicts short-horizon stock returns from the
flow of financial news, filings, and social signals — not as a single number,
but as a **calibrated probability range**.

Most retail "sentiment" tools collapse news into a buy/sell label. This system
keeps sentiment continuous and models the harder questions: *how credible is the
source, when will the information actually hit the price, and how uncertain is
the move?* It runs as five sequential stages:

1. **Multi-source ingestion** — market prices, news (NewsAPI / Alpha Vantage),
   SEC EDGAR filings, Form 4 insider transactions, Reddit social signal, and
   Google Trends, with a point-in-time news archive so backtests stay free of
   look-ahead bias.
2. **Financial sentiment (FinBERT)** — transformer-based sentiment scored on a
   continuous [-1, 1] scale, combined across sources with **credibility
   weighting** and **recency decay** rather than naive averaging.
3. **Graph diffusion** — a heat-kernel model on a graph Laplacian over actor
   categories (SEC/Corporate → Institutional → Press → Retail) estimates the
   **lag before news impacts price**, plus a rolling-correlation sector graph
   for cross-asset spillover.
4. **Quantile return prediction** — predicts the **P10 / P50 / P90** of
   log-return at 1-hour, 1-day, and 5-day horizons, conditioned on the current
   VIX volatility regime (low/med/high). Quantiles are sorted to prevent
   crossing, so the output is always a valid distribution.
5. **Backtesting & serving** — a custom **walk-forward backtester** scores the
   model against momentum, ARIMA, and EWMA-GARCH baselines using **pinball
   loss** and interval-coverage metrics, and a **FastAPI** service exposes the
   model over a `/predict` endpoint, containerized with Docker.

The codebase is fully typed (enforced by `mypy`), tested with `pytest`, and gated
by CI that runs formatting, type-checking, and the test suite on every change.
Heavy dependencies (PyTorch / FinBERT) are lazily imported so the test/CI
environment stays lightweight.

---

## Skills

`Python` · `Machine Learning` · `Quantitative Finance` · `Time-Series Forecasting`
· `Natural Language Processing (NLP)` · `FinBERT / Transformers` · `PyTorch` ·
`scikit-learn` · `Quantile Regression` · `Graph Theory / Network Models` ·
`pandas` · `NumPy` · `SciPy` · `FastAPI` · `REST API Development` · `Docker` ·
`Backtesting & Strategy Evaluation` · `Data Engineering / ETL` ·
`API Integration (NewsAPI, SEC EDGAR, yfinance, Reddit)` · `pytest` ·
`Type-Safe Python (mypy)` · `CI/CD (GitHub Actions)`

---

## Deliverables

- **End-to-end forecasting pipeline** — orchestrated in-process across all five
  stages, runnable for any ticker from a single command.
- **Trained model artifacts** — quantile models with opt-in enrichment features
  (peer signals, Google Trends, insider flow, VIX regime), each degrading
  gracefully when a data source is unavailable.
- **Walk-forward backtesting harness** — pinball loss, interval coverage,
  directional hit rate, Sharpe, and a P&L curve, benchmarked against three
  classical baselines.
- **FastAPI serving layer** — `POST /predict` + `GET /health`, Dockerized for
  deployment.
- **Multi-source data layer** — cached connectors for prices, news, filings,
  insider flow, social, and trends, plus a point-in-time news archive.
- **Test suite + CI** — `pytest` coverage mirroring the source tree, with
  `black`, `isort`, and `mypy` enforced on every commit.
- **Documentation** — architecture, graph-model, data-source/credibility, and
  deployment guides.

---

*Suggested attachments for the listing:* `case_study.png` (hero image) and
`case_study.pdf` (downloadable one-pager), both in this folder.
