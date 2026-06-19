"""Mocked tests for the network ingestion clients (no real network)."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# price.fetch_prices / fetch_returns_matrix
# --------------------------------------------------------------------------- #
class TestPriceClient:
    def test_empty_intraday_raises_with_hint(self, monkeypatch):
        fake_yf = types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        from src.ingestion.price import fetch_prices

        with pytest.raises(ValueError, match="intraday"):
            fetch_prices("AAPL", interval="1h")

    def test_empty_daily_raises_without_hint(self, monkeypatch):
        fake_yf = types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        from src.ingestion.price import fetch_prices

        with pytest.raises(ValueError) as exc:
            fetch_prices("AAPL", interval="1d")
        assert "intraday" not in str(exc.value)

    def test_returns_matrix_combines_tickers(self, monkeypatch):
        import src.ingestion.price as price_mod

        idx = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")

        def _fake_fetch(ticker, start=None, end=None, interval="1d"):
            return pd.DataFrame({"log_return": [0.0, 0.01, -0.01, 0.02]}, index=idx)

        monkeypatch.setattr(price_mod, "fetch_prices", _fake_fetch)
        out = price_mod.fetch_returns_matrix(["AAPL", "MSFT"], start="2023-01-01")
        assert list(out.columns) == ["AAPL", "MSFT"]
        assert out.index.tz is not None


# --------------------------------------------------------------------------- #
# social.fetch_reddit
# --------------------------------------------------------------------------- #
class _FakeSubmission:
    def __init__(self, title, body, score, created_utc):
        self.title = title
        self.selftext = body
        self.score = score
        self.created_utc = created_utc


class _FakeSubreddit:
    def __init__(self, submissions):
        self._subs = submissions

    def search(self, ticker, limit=25):
        return list(self._subs)[:limit]


class _FakeReddit:
    def __init__(self, submissions):
        self._subs = submissions

    def subreddit(self, name):
        return _FakeSubreddit(self._subs)


class TestSocialClient:
    def test_fetch_reddit_parses_posts(self, monkeypatch):
        import src.ingestion.social as social_mod

        subs = [_FakeSubmission("DD on AAPL", "buy", 42, 1_700_000_000.0)]
        monkeypatch.setattr(social_mod, "_reddit_client", lambda: _FakeReddit(subs))

        posts = social_mod.fetch_reddit("AAPL", subreddits=["investing"], limit=5)
        assert len(posts) == 1
        p = posts[0]
        assert p.title == "DD on AAPL"
        assert p.subreddit == "investing"
        assert p.score == 42
        assert p.created_at.tzinfo is not None
        # "investing" is an informed subreddit.
        assert p.source_category == social_mod.NODE_INFORMED_RETAIL

    def test_reddit_client_requires_credentials(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        import src.ingestion.social as social_mod

        with pytest.raises(RuntimeError):
            social_mod._reddit_client()


# --------------------------------------------------------------------------- #
# filings.fetch_filings (dir-walk / sort / skip-unparseable)
# --------------------------------------------------------------------------- #
class TestFilingsClient:
    def _make_tree(self, root, ticker="AAPL", form="8-K"):
        base = root / "sec-edgar-filings" / ticker / form
        # Two parseable filings (different acceptance datetimes) + one unparseable.
        (base / "acc-old").mkdir(parents=True)
        (base / "acc-old" / "full-submission.txt").write_text(
            "<ACCEPTANCE-DATETIME>20230101090000\n"
        )
        (base / "acc-new").mkdir(parents=True)
        (base / "acc-new" / "full-submission.txt").write_text(
            "<ACCEPTANCE-DATETIME>20230315140000\n"
        )
        (base / "acc-bad").mkdir(parents=True)
        (base / "acc-bad" / "full-submission.txt").write_text("no header here\n")

    def test_sorted_newest_first_and_skips_unparseable(self, monkeypatch, tmp_path):
        import src.ingestion.filings as filings_mod

        monkeypatch.setattr(filings_mod, "EDGAR_DOWNLOAD_ROOT", tmp_path)
        self._make_tree(tmp_path)

        # Stub the downloader so .get() is a no-op (files already on disk).
        class _Downloader:
            def __init__(self, *a, **k):
                pass

            def get(self, *a, **k):
                return 0

        monkeypatch.setitem(
            sys.modules, "sec_edgar_downloader", types.SimpleNamespace(Downloader=_Downloader)
        )

        filings = filings_mod.fetch_filings("AAPL", form="8-K")
        # The unparseable folder is skipped; the rest are newest-first.
        assert len(filings) == 2
        assert filings[0].filed_at > filings[1].filed_at
        assert filings[0].filed_at == datetime(2023, 3, 15, 14, 0, tzinfo=timezone.utc)
        assert all(f.source_category == filings_mod.NODE_SEC_CORP for f in filings)

    def test_limit_applied(self, monkeypatch, tmp_path):
        import src.ingestion.filings as filings_mod

        monkeypatch.setattr(filings_mod, "EDGAR_DOWNLOAD_ROOT", tmp_path)
        self._make_tree(tmp_path)

        class _Downloader:
            def __init__(self, *a, **k):
                pass

            def get(self, *a, **k):
                return 0

        monkeypatch.setitem(
            sys.modules, "sec_edgar_downloader", types.SimpleNamespace(Downloader=_Downloader)
        )

        filings = filings_mod.fetch_filings("AAPL", form="8-K", limit=1)
        assert len(filings) == 1
        # The single returned filing is the most recent parseable one.
        assert filings[0].filed_at == datetime(2023, 3, 15, 14, 0, tzinfo=timezone.utc)
