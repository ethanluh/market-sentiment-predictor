from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from alpaca.data.models.news import NewsSet

from src.ingestion.alpaca_news import _clean, fetch_alpaca_news

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def item(i: int, created: str, symbols: list[str], **kw: object) -> dict:
    base = {
        "id": i,
        "headline": f"Headline {i}",
        "author": "a",
        "created_at": created,
        "updated_at": "2030-01-01T00:00:00Z",
        "summary": "short summary",
        "content": "<p>Full &amp; <b>rich</b> text</p>",
        "url": f"https://example.com/{i}",
        "images": [],
        "symbols": symbols,
        "source": "benzinga",
    }
    return {**base, **kw}


class FakeClient:
    def __init__(self, raw: list[dict]) -> None:
        self.raw, self.requests = raw, []

    def get_news(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return NewsSet({"news": self.raw})


def test_articles_are_sorted_oldest_first_with_clean_text() -> None:
    client = FakeClient(
        [
            item(2, "2024-01-03T15:00:00Z", ["AAPL"]),
            item(1, "2024-01-02T14:30:00Z", ["AAPL"]),
        ]
    )
    arts = fetch_alpaca_news("aapl", START, client=client)
    assert [a.headline for a in arts] == ["Headline 1", "Headline 2"]
    assert arts[0].body == "Full & rich text"
    assert arts[0].source == "benzinga" and arts[0].url == "https://example.com/1"
    assert arts[0].published_at == datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)


def test_published_at_is_created_at_not_updated_at() -> None:
    art = fetch_alpaca_news(
        "AAPL", START, client=FakeClient([item(1, "2024-01-02T14:30:00Z", ["AAPL"])])
    )[0]
    assert art.published_at.year == 2024  # updated_at is 2030; using it would leak the future


def test_items_without_the_requested_tag_raise_instead_of_vanishing() -> None:
    client = FakeClient([item(1, "2024-01-02T14:30:00Z", ["BRK-B"])])
    with pytest.raises(ValueError, match="lack the BRK.B tag"):
        fetch_alpaca_news("BRK.B", START, client=client)


def test_multi_symbol_items_are_kept() -> None:
    client = FakeClient([item(1, "2024-01-02T14:30:00Z", ["AAPL", "MSFT"])])
    assert len(fetch_alpaca_news("AAPL", START, client=client)) == 1


def test_created_at_window_is_enforced_locally() -> None:
    """The API may bound by updated_at; articles created outside [start, end] must be dropped."""
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    client = FakeClient(
        [
            item(1, "2023-12-31T23:59:59Z", ["AAPL"]),  # created before start, maybe edited later
            item(2, "2024-01-15T10:00:00Z", ["AAPL"]),
            item(3, "2024-02-01T00:00:01Z", ["AAPL"]),  # created after end
        ]
    )
    got = fetch_alpaca_news("AAPL", START, end=end, client=client)
    assert [a.headline for a in got] == ["Headline 2"]


def test_summary_is_the_fallback_body() -> None:
    client = FakeClient([item(1, "2024-01-02T14:30:00Z", ["AAPL"], content="")])
    assert fetch_alpaca_news("AAPL", START, client=client)[0].body == "short summary"


def test_request_is_built_with_the_requested_window() -> None:
    client = FakeClient([])
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    assert fetch_alpaca_news("aapl", START, end=end, limit=10, client=client) == []
    req = client.requests[0]
    assert (req.symbols, req.start, req.end, req.sort, req.limit, req.include_content) == (
        "AAPL",
        START,
        end,
        "asc",
        10,
        True,
    )


def test_start_is_required() -> None:
    with pytest.raises(ValueError, match="start is required"):
        fetch_alpaca_news("AAPL", None, client=FakeClient([]))  # type: ignore[arg-type]


def test_missing_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        fetch_alpaca_news("AAPL", START)


def test_naive_or_missing_timestamps_raise() -> None:
    naive = SimpleNamespace(
        id=1,
        created_at=datetime(2024, 1, 2),
        symbols=["AAPL"],
        headline="h",
        content="",
        summary="",
        source="",
        url="",
    )
    client = SimpleNamespace(get_news=lambda req: SimpleNamespace(data={"news": [naive]}))
    with pytest.raises(ValueError, match="naive"):
        fetch_alpaca_news("AAPL", START, client=client)
    missing = SimpleNamespace(**{**naive.__dict__, "created_at": None})
    client = SimpleNamespace(get_news=lambda req: SimpleNamespace(data={"news": [missing]}))
    with pytest.raises(ValueError, match="no created_at"):
        fetch_alpaca_news("AAPL", START, client=client)


def test_clean_handles_empty_and_entities() -> None:
    assert _clean(None) == "" and _clean("") == ""
    assert _clean("a&nbsp;&lt;b&gt;<br/>c") == "a <b> c"


def test_archive_builder_treats_a_date_only_end_as_end_of_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.build_news_archive as bna

    seen = {}

    def fake_fetch(ticker, start, end=None, limit=None, client=None):  # type: ignore[no-untyped-def]
        seen.update(start=start, end=end)
        return []

    monkeypatch.setattr("src.ingestion.alpaca_news.fetch_alpaca_news", fake_fetch)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bna._fetch_articles("alpaca", "AAPL", start, datetime(2024, 2, 1, tzinfo=timezone.utc))
    assert seen["end"] == datetime(2024, 2, 1, 23, 59, 59, 999999, tzinfo=timezone.utc)
    bna._fetch_articles("alpaca", "AAPL", start, datetime(2024, 2, 1, 12, 0, tzinfo=timezone.utc))
    assert seen["end"] == datetime(2024, 2, 1, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="--start is required"):
        bna._fetch_articles("alpaca", "AAPL", None, None)
