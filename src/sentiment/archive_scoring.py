"""
Batch FinBERT scoring of the news archive, resumable and shared across tickers.

Scores every archived article that is missing from its ticker's score cache, scoring each unique
URL once even when several tickers carry it. ``score_fn`` is injected so this module needs no torch
(tests fake it); the CLI passes ``src.sentiment.inference.score_batch``. Texts are sorted by length
so batches pad little. Caches are flushed atomically (each file via temp-then-rename) every ``flush_texts`` scores and
once more on any exit, so an interrupted run keeps its work and resumes. Bad scores raise; nothing is skipped silently.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.ingestion import news_archive as na

ScoreFn = Callable[[list[str]], list[float]]


@dataclass(frozen=True)
class ScoreReport:
    tickers: int
    articles: int  # archived articles across the selected tickers
    already_scored: int
    scored_now: int  # unique URLs scored in this run


def score_archive(
    tickers: list[str],
    score_fn: ScoreFn,
    *,
    batch_size: int = 16,
    since: datetime | None = None,
    max_articles: int | None = None,
    flush_texts: int = 2000,
    log: Callable[[str], None] = print,
) -> ScoreReport:
    """
    Fill the score caches for ``tickers``. ``since`` skips older articles; ``max_articles`` caps
    the unique URLs scored this run (a smoke-test knob; the shortest texts go first).
    """
    if batch_size < 1 or flush_texts < 1:
        raise ValueError("batch_size and flush_texts must be >= 1")

    caches: dict[str, dict[str, float]] = {}
    urls_by_ticker: dict[str, set[str]] = {}
    text_by_url: dict[str, str] = {}
    total = already = 0
    for ticker in tickers:
        articles = na.read_all(ticker)
        if since is not None:
            articles = [a for a in articles if a.published_at >= since]
        caches[ticker] = na._load_score_cache(ticker)
        urls_by_ticker[ticker] = {a.url for a in articles}
        total += len(articles)
        already += sum(1 for a in articles if a.url in caches[ticker])
        for a in articles:
            if a.url not in caches[ticker]:
                text_by_url.setdefault(a.url, a.text)

    # A URL scored in one ticker's cache but missing from another's is copied, not rescored.
    known = {u: s for c in caches.values() for u, s in c.items()}
    todo = {u: t for u, t in text_by_url.items() if u not in known}
    order = sorted(todo, key=lambda u: len(todo[u]))
    if max_articles is not None:
        order = order[:max_articles]

    def flush(scores: dict[str, float]) -> None:
        for ticker, cache in caches.items():
            changed = False
            for url in urls_by_ticker[ticker]:
                if url in cache:
                    continue
                if url in scores:
                    cache[url] = scores[url]
                    changed = True
                elif url in known:  # scored earlier under another ticker: copy, never rescore
                    cache[url] = known[url]
                    changed = True
            if changed:
                na._save_score_cache(ticker, cache)

    scores: dict[str, float] = {}
    started = time.perf_counter()
    since_flush = 0
    try:
        for start in range(0, len(order), batch_size):
            urls = order[start : start + batch_size]
            out = score_fn([todo[u] for u in urls])
            if len(out) != len(urls):
                raise RuntimeError(f"score_fn returned {len(out)} scores for {len(urls)} texts")
            for url, score in zip(urls, out):
                if not math.isfinite(score) or not -1.0 <= score <= 1.0:
                    raise RuntimeError(f"invalid sentiment score {score!r} for {url}")
                scores[url] = float(score)
            since_flush += len(urls)
            if since_flush >= flush_texts:
                flush(scores)
                since_flush = 0
                done = len(scores)
                rate = done / (time.perf_counter() - started)
                log(
                    f"scored {done}/{len(order)} ({rate:.1f}/s, ~{(len(order) - done) / rate / 60:.0f} min left)"
                )
    finally:
        flush(scores)  # an interrupt or error keeps every score computed so far
    return ScoreReport(len(tickers), total, already, len(scores))


def merge_caches(source_dir: Path, tickers: list[str], tolerance: float = 5e-3) -> int:
    """
    Merge score caches produced elsewhere (a GPU box) into the local ones; returns entries added.

    Entries present in both must agree within ``tolerance`` (fp32 differs slightly between
    devices); a larger gap raises, since it means the two runs scored different text or models.
    Invalid scores raise. Local entries are never lost.
    """
    added = 0
    for ticker in tickers:
        path = source_dir / f"{ticker.upper()}.json"
        if not path.exists():
            continue
        try:
            incoming = {str(k): float(v) for k, v in json.loads(path.read_text("utf-8")).items()}
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"corrupt score cache {path}") from exc
        local = na._load_score_cache(ticker)
        for url, score in incoming.items():
            if not math.isfinite(score) or not -1.0 <= score <= 1.0:
                raise ValueError(f"invalid score {score!r} for {url} in {path}")
            if url in local:
                if abs(local[url] - score) > tolerance:
                    raise ValueError(f"{ticker} {url}: local {local[url]} vs incoming {score}")
            else:
                local[url] = score
                added += 1
        na._save_score_cache(ticker, local)
    return added
