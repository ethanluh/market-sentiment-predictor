"""
Score the news archive with FinBERT (resumable; see docs/scoring.md).

    python scripts/score_archive.py                       # every ticker in the archive
    python scripts/score_archive.py --tickers AAPL MSFT --since 2022-01-01
    python scripts/score_archive.py --verify 50           # recompute 50 cached scores, compare

Uses a GPU when torch finds one. Set MSP_DATA_ROOT to point at a copy of the data tree.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import news_archive as na  # noqa: E402
from src.sentiment.archive_scoring import merge_caches, score_archive  # noqa: E402

VERIFY_TOLERANCE = 5e-3  # fp32 differs slightly between CPU and GPU kernels and batch shapes


def _parse_since(value: str | None) -> datetime | None:
    """A naive value is taken as UTC; an explicit offset is converted, never discarded."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )


def _baseline_urls(baseline_root: Path, ticker: str) -> set[str]:
    """URLs already scored under ``baseline_root`` (a data tree); strict about a corrupt file."""
    path = baseline_root / "processed" / "news_scores" / f"{ticker.upper()}.json"
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text("utf-8")))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"corrupt score cache {path}") from exc


def _verify(tickers: list[str], n: int, seed: int, baseline_root: Path | None = None) -> int:
    """
    Recompute a random sample of cached scores. With ``baseline_root`` only URLs absent from that
    data tree's caches are sampled, i.e. exactly the scores a merge would add.
    """
    from src.sentiment.inference import score_batch

    pool: list[tuple[str, str, float]] = []
    for ticker in tickers:
        cache = na._load_score_cache(ticker)
        known = _baseline_urls(baseline_root, ticker) if baseline_root else set()
        for a in na.read_all(ticker):
            if a.url in cache and a.url not in known:
                pool.append((ticker, a.text, cache[a.url]))
    if not pool and baseline_root:
        print("no new scores to verify (the cloud added nothing)")
        return 0
    if not pool:
        print("nothing cached to verify", file=sys.stderr)
        return 1
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    fresh = score_batch([t for _, t, _ in sample])
    worst = max(abs(f - c) for f, (_, _, c) in zip(fresh, sample))
    print(f"verified {len(sample)} cached scores: max abs difference {worst:.5f}")
    if worst > VERIFY_TOLERANCE:
        print(f"FAILED: exceeds tolerance {VERIFY_TOLERANCE}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tickers", nargs="*", help="default: every directory in the archive")
    p.add_argument("--since", help="ISO date; skip older articles")
    p.add_argument("--batch-size", type=int, help="default: 64 on a GPU, 16 on CPU")
    p.add_argument("--max-articles", type=int, help="cap unique URLs scored this run (smoke test)")
    p.add_argument("--verify", type=int, metavar="N", help="recompute N cached scores and compare")
    p.add_argument(
        "--baseline-root",
        type=Path,
        metavar="DIR",
        help="with --verify: sample only scores absent from this data tree's caches",
    )
    p.add_argument(
        "--merge-from", type=Path, metavar="DIR", help="merge score caches made elsewhere"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    tickers = args.tickers or sorted(d.name for d in na.ARCHIVE_ROOT.iterdir() if d.is_dir())
    if not tickers:
        p.error(f"no tickers found under {na.ARCHIVE_ROOT}")
    if args.merge_from:
        print(f"merged {merge_caches(args.merge_from, tickers)} new scores from {args.merge_from}")
        return 0
    if args.verify:
        return _verify(tickers, args.verify, args.seed, args.baseline_root)

    import torch

    from src.sentiment.inference import score_batch

    since = _parse_since(args.since)
    batch = args.batch_size or (64 if torch.cuda.is_available() else 16)
    print(
        f"device: {'cuda' if torch.cuda.is_available() else 'cpu'} | batch {batch} | {len(tickers)} tickers"
    )
    report = score_archive(
        tickers, score_batch, batch_size=batch, since=since, max_articles=args.max_articles
    )
    print(f"done: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
