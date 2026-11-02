# Paper trading

A long-only, whole-share, weekly-rebalance trader that runs against an Alpaca **paper** account.
There is no live mode: `AlpacaBroker.paper_from_env` always builds a paper client and refuses a
client that is not on the paper endpoint, and any other mode raises `LiveTradingNotImplemented`.
Going live needs its own reviewed change.

## Setup

1. Create an Alpaca account and generate **paper** API keys (Paper Trading, API Keys).
2. Put them in `.env` (git-ignored; see `.env.example`):
   `ALPACA_PAPER_KEY_ID`, `ALPACA_PAPER_SECRET_KEY`.
3. Edit `config/universe_starter.txt` if you want different tickers. It is the only place
   tickers are chosen. SPY is the evaluation benchmark and is never traded.

## Running a cycle

Run after the close; the CLI refuses while the market is open.

```bash
# Dry run (default): plans, checks risk limits, writes the ledger, sends nothing.
python -m src.trading.run --forecasts data/trading/forecasts.json

# Send paper orders (queued as DAY market orders for the next open).
python -m src.trading.run --forecasts data/trading/forecasts.json --submit
```

`forecasts.json` holds one P10/P50/P90 5-day log-return forecast per universe ticker:

```json
{"as_of": "2026-01-09", "horizon": "5d",
 "forecasts": {"AAPL": {"p10": -0.02, "p50": 0.01, "p90": 0.04}}}
```

`as_of` must equal the broker's New York market date (so a Saturday run needs a file dated
Saturday), `horizon` must match the configured horizon, unknown keys and non-finite or
absurd values (beyond +/-100%) are refused, and every universe ticker must be present (a missing
forecast would read as "no signal" and sell a held name).

Run from a directory that has your `.env`, or export the Alpaca variables: `.env` is looked up
from the current directory upward, so a worktree without one fails loudly. Daily bars dated
today are refused before 16:30 New York time, so run after that. A dry run still appends records
to the ledger, flagged `dry_run`, and they never count toward the high-water mark.

## Decision rules

- Hold a name only if its P50 exceeds round-trip cost plus buffer (25 bps by default) and its P10
  is no worse than -8%. Equal weight, capped at 5% per name, gross exposure at most
  `1 / (1 + slippage buffer)`; the rest stays in cash. No shorting, margin, options or leverage.
- Orders are whole shares, sells before buys. Drift smaller than 0.5% of equity is left alone.

## Safety limits (every one raises and aborts the cycle before any order is sent)

Kill switch file (`touch data/trading/KILL`), daily loss over 2%, drawdown over 10% from the
ledger's high-water mark, more than 60 orders a day, any price under $5, price bars older than 5
days for any held or traded name, bars that disagree with the broker's equity by over 2%, a
non-finite number anywhere, an order that would exceed the position or gross cap, or a plan that
needs margin. Orders open at the broker block planning.

A halt means a human decides. The system never liquidates on its own, so sells are blocked too.

## The ledger

`data/trading/ledger-paper.jsonl` is append-only: forecasts, plans, equity snapshots (only after
validation), breaches, and a write-ahead `order_intent` before every submission followed by
`order_accepted` or `order_failed`. One file per mode, so a paper high-water mark can never govern
live trading. A torn last line stops all appends until repaired by hand.

## After a halt, partial fill or failure

Order ids are deterministic per day, ticker and side. If an order was attempted but positions do
not reflect it (partial fill, rejection, a crash mid-cycle), the next run stops with "attempted
earlier". Check the broker, resolve it by hand, and rerun on the next market date.

## Known blockers before trusting any forecast

- **Train/serve skew.** `src/pipeline/run.py::run` builds live features without VIX/regime or peer
  returns and takes news from NewsAPI, while a model trained with `--use-vix`, peers and the news
  archive expects them. Forecasts for the trading CLI therefore come from a file for now; a live
  forecast source must reuse the training feature builder before `--submit` is meaningful.
- **`estimated_lag_hours`** is nearly constant for press and retail origins (see
  `docs/graph_model.md`). The ablation in `src/eval` shows whether the feature helps at all.
- The model has not yet shown an edge on historical data. Run `src/eval` first.

## Scheduling

A weekly systemd timer or cron entry on your own machine, after Friday's close, is enough. There
is no daemon. Install it yourself; nothing here installs a schedule.
