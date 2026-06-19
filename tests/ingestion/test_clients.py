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


# --------------------------------------------------------------------------- #
# trends.fetch_trends (Google Trends via pytrends)
# --------------------------------------------------------------------------- #
def _install_pytrends(monkeypatch, frame=None, raise_on_call=False, captured=None):
    """Inject a fake ``pytrends.request.TrendReq`` (no network / real dep)."""

    class _FakeTrendReq:
        def __init__(self, *a, **k):
            pass

        def build_payload(self, kw_list, timeframe=None, geo="", **k):
            if captured is not None:
                captured["kw_list"] = kw_list
                captured["timeframe"] = timeframe
                captured["geo"] = geo

        def interest_over_time(self):
            if raise_on_call:
                raise RuntimeError("rate limited")
            return frame

    pkg = types.ModuleType("pytrends")
    req_mod = types.ModuleType("pytrends.request")
    req_mod.TrendReq = _FakeTrendReq
    monkeypatch.setitem(sys.modules, "pytrends", pkg)
    monkeypatch.setitem(sys.modules, "pytrends.request", req_mod)


def _trends_frame(term="AAPL", n=30, partial_tail=1):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")  # tz-naive on purpose
    is_partial = [False] * (n - partial_tail) + [True] * partial_tail
    return pd.DataFrame({term: [i % 100 for i in range(n)], "isPartial": is_partial}, index=idx)


class TestTrendsClient:
    def test_parses_0_100_series_and_drops_ispartial(self, monkeypatch):
        _install_pytrends(monkeypatch, frame=_trends_frame())
        from src.ingestion.trends import fetch_trends

        s = fetch_trends("AAPL")
        assert isinstance(s, pd.Series)
        assert s.name == "search_interest"
        assert s.dtype == float
        assert s.between(0, 100).all()
        assert len(s) == 30
        # UTC-aware index even though the source frame was tz-naive.
        assert s.index.tz is not None

    def test_timeframe_derived_from_start_end(self, monkeypatch):
        captured: dict = {}
        _install_pytrends(monkeypatch, frame=_trends_frame(), captured=captured)
        from src.ingestion.trends import fetch_trends

        fetch_trends("AAPL", start="2024-01-01", end="2024-06-01", geo="US")
        assert captured["timeframe"] == "2024-01-01 2024-06-01"
        assert captured["kw_list"] == ["AAPL"]
        assert captured["geo"] == "US"

    def test_explicit_timeframe_overrides_dates(self, monkeypatch):
        captured: dict = {}
        _install_pytrends(monkeypatch, frame=_trends_frame(), captured=captured)
        from src.ingestion.trends import fetch_trends

        fetch_trends("AAPL", start="2024-01-01", end="2024-06-01", timeframe="today 5-y")
        assert captured["timeframe"] == "today 5-y"

    def test_keyword_overrides_ticker(self, monkeypatch):
        captured: dict = {}
        _install_pytrends(monkeypatch, frame=_trends_frame(term="Apple stock"), captured=captured)
        from src.ingestion.trends import fetch_trends

        s = fetch_trends("AAPL", keyword="Apple stock")
        assert captured["kw_list"] == ["Apple stock"]
        assert s.name == "search_interest"

    def test_empty_response_raises(self, monkeypatch):
        _install_pytrends(monkeypatch, frame=pd.DataFrame())
        from src.ingestion.trends import fetch_trends

        with pytest.raises(ValueError, match="No Google Trends data"):
            fetch_trends("AAPL")

    def test_missing_term_column_raises(self, monkeypatch):
        # A frame that lacks the queried term column is treated as no data.
        _install_pytrends(monkeypatch, frame=_trends_frame(term="MSFT"))
        from src.ingestion.trends import fetch_trends

        with pytest.raises(ValueError, match="No Google Trends data"):
            fetch_trends("AAPL")

    def test_interest_call_failure_raises_valueerror(self, monkeypatch):
        _install_pytrends(monkeypatch, raise_on_call=True)
        from src.ingestion.trends import fetch_trends

        with pytest.raises(ValueError, match="No Google Trends data"):
            fetch_trends("AAPL")

    def test_missing_dependency_raises_runtimeerror(self, monkeypatch):
        # pytrends is not installed in CI; ensure no fake lingers, then assert the
        # informative error fires from the lazy import.
        monkeypatch.delitem(sys.modules, "pytrends", raising=False)
        monkeypatch.delitem(sys.modules, "pytrends.request", raising=False)
        from src.ingestion.trends import fetch_trends

        with pytest.raises(RuntimeError, match="pytrends is required"):
            fetch_trends("AAPL")
