#!/usr/bin/env bash
#
# Cron wrapper: append fresh news to the point-in-time archive for one or more
# tickers. Designed to be scheduled with crontab so the archive accumulates over
# time (see docs/testing_the_thesis.md). The underlying writer is append-only, so
# re-running is safe — already-archived days are skipped.
#
# Configuration via environment variables (or a repo-root .env, loaded below):
#   TICKERS      space-separated tickers to fetch   (default: "AAPL")
#   NEWS_SOURCE  newsapi | alphavantage             (default: "newsapi")
#   PYTHON_BIN   python interpreter to use          (default: "python3")
#   NEWS_API_KEY / ALPHAVANTAGE_API_KEY             (per the chosen source)
#
# Example crontab entry (daily at 06:30, log to a file):
#   30 6 * * * TICKERS="AAPL MSFT" NEWS_SOURCE=alphavantage \
#     /path/to/repo/scripts/cron_news_archive.sh >> /path/to/repo/news_cron.log 2>&1
#
set -euo pipefail

# Resolve repo root from this script's own location so cron's cwd doesn't matter.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load API keys from .env if present (NEWS_API_KEY / ALPHAVANTAGE_API_KEY).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
TICKERS="${TICKERS:-AAPL}"
NEWS_SOURCE="${NEWS_SOURCE:-newsapi}"

for ticker in $TICKERS; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] build_news_archive ticker=$ticker source=$NEWS_SOURCE"
  "$PYTHON_BIN" scripts/build_news_archive.py --ticker "$ticker" --source "$NEWS_SOURCE"
done
