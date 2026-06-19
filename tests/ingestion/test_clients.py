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
# form4.fetch_form4 (SEC Form 4 insider trades)
# --------------------------------------------------------------------------- #
def _form4_xml(transactions) -> str:
    """Build an ownershipDocument XML from (code, acquired_disposed, shares) rows."""
    rows = "".join(f"""
        <nonDerivativeTransaction>
          <securityTitle><value>Common Stock</value></securityTitle>
          <transactionDate><value>2023-03-09</value></transactionDate>
          <transactionCoding>
            <transactionFormType>4</transactionFormType>
            <transactionCode>{code}</transactionCode>
            <equitySwapInvolved>0</equitySwapInvolved>
          </transactionCoding>
          <transactionAmounts>
            <transactionShares><value>{shares}</value></transactionShares>
            <transactionPricePerShare><value>150.0</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode>
          </transactionAmounts>
        </nonDerivativeTransaction>""" for code, ad, shares in transactions)
    return (
        '<?xml version="1.0"?>\n'
        "<ownershipDocument><documentType>4</documentType>"
        f"<nonDerivativeTable>{rows}</nonDerivativeTable></ownershipDocument>"
    )


class _Form4Downloader:
    def __init__(self, *a, **k):
        pass

    def get(self, *a, **k):
        return 0


class TestForm4Client:
    def _make_tree(self, root, ticker="AAPL", with_informative=True):
        base = root / "sec-edgar-filings" / ticker / "4"
        # acc-buy: a P purchase (+1000) + an A grant (must be excluded); parsed
        # from the clean primary-document.xml.
        (base / "acc-buy").mkdir(parents=True)
        (base / "acc-buy" / "full-submission.txt").write_text(
            "<ACCEPTANCE-DATETIME>20230310140000\n"
        )
        buy_txns = [("P", "A", 1000), ("A", "A", 5000)] if with_informative else [("A", "A", 5000)]
        (base / "acc-buy" / "primary-document.xml").write_text(_form4_xml(buy_txns))
        # acc-sell: an S sale (-2000), parsed via the full-submission.txt SGML fallback.
        (base / "acc-sell").mkdir(parents=True)
        sell_xml = (
            _form4_xml([("S", "D", 2000)]) if with_informative else _form4_xml([("M", "A", 7000)])
        )
        (base / "acc-sell" / "full-submission.txt").write_text(
            "<ACCEPTANCE-DATETIME>20230320140000\n" + sell_xml
        )

    def _install(self, monkeypatch, tmp_path, **kw):
        import src.ingestion.filings as filings_mod

        monkeypatch.setattr(filings_mod, "EDGAR_DOWNLOAD_ROOT", tmp_path)
        monkeypatch.setitem(
            sys.modules, "sec_edgar_downloader", types.SimpleNamespace(Downloader=_Form4Downloader)
        )
        self._make_tree(tmp_path, **kw)

    def test_parses_signed_ps_transactions_and_excludes_routine(self, monkeypatch, tmp_path):
        self._install(monkeypatch, tmp_path)
        from src.ingestion.form4 import fetch_form4

        s = fetch_form4("AAPL")
        assert s.name == "insider_net_shares"
        assert s.index.tz is not None
        # Two informative transactions: P buy (+1000), S sale (-2000). A grant dropped.
        assert sorted(s.tolist()) == [-2000.0, 1000.0]
        # The purchase is stamped with the earlier acceptance datetime.
        assert s.loc[s == 1000.0].index[0] == pd.Timestamp("2023-03-10 14:00", tz="UTC")

    def test_uses_acceptance_datetime_index(self, monkeypatch, tmp_path):
        self._install(monkeypatch, tmp_path)
        from src.ingestion.form4 import fetch_form4

        s = fetch_form4("AAPL")
        assert set(s.index) == {
            pd.Timestamp("2023-03-10 14:00", tz="UTC"),
            pd.Timestamp("2023-03-20 14:00", tz="UTC"),
        }

    def test_no_informative_transactions_raises(self, monkeypatch, tmp_path):
        # Only grants (A) / option exercises (M) present -> no P/S signal.
        self._install(monkeypatch, tmp_path, with_informative=False)
        from src.ingestion.form4 import fetch_form4

        with pytest.raises(ValueError, match="No Form 4 insider transactions"):
            fetch_form4("AAPL")


class TestForm4Parser:
    """Defensive parsing: malformed input contributes nothing, never crashes."""

    def test_malformed_xml_returns_empty(self, tmp_path):
        from src.ingestion.form4 import _parse_form4_transactions

        (tmp_path / "primary-document.xml").write_text("<ownershipDocument><not closed")
        assert _parse_form4_transactions(tmp_path) == []

    def test_missing_ownership_doc_returns_empty(self, tmp_path):
        from src.ingestion.form4 import _parse_form4_transactions

        # full-submission.txt without an <ownershipDocument> block.
        (tmp_path / "full-submission.txt").write_text("<SEC-HEADER>only header</SEC-HEADER>")
        assert _parse_form4_transactions(tmp_path) == []

    def test_no_files_returns_empty(self, tmp_path):
        from src.ingestion.form4 import _parse_form4_transactions

        assert _parse_form4_transactions(tmp_path) == []

    def test_transaction_missing_shares_is_skipped(self, tmp_path):
        from src.ingestion.form4 import _parse_form4_transactions

        # A P transaction with no <transactionShares> must be skipped, not crash.
        xml = (
            '<?xml version="1.0"?><ownershipDocument><nonDerivativeTable>'
            "<nonDerivativeTransaction><transactionCoding>"
            "<transactionCode>P</transactionCode></transactionCoding>"
            "<transactionAmounts>"
            "<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>"
            "</transactionAmounts></nonDerivativeTransaction>"
            "</nonDerivativeTable></ownershipDocument>"
        )
        (tmp_path / "primary-document.xml").write_text(xml)
        assert _parse_form4_transactions(tmp_path) == []


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
