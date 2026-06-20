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

*(598 chars — fits Upwork's 600 limit; paste as one paragraph.)*

A production-grade pipeline that forecasts short-horizon stock returns from financial news, filings, and social signals as a calibrated probability range, not a single number. Five stages: multi-source ingestion (news, SEC, insider, social, prices); continuous FinBERT sentiment with credibility weighting; a graph-diffusion model estimating when news hits price; and quantile regression predicting P10/P50/P90 returns at 1h/1d/5d by VIX regime. A walk-forward backtester scores it against momentum/ARIMA/GARCH baselines, served via a Dockerized FastAPI endpoint. Fully typed, tested, and CI-gated.

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
