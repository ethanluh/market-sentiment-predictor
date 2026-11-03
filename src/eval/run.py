"""
CLI for the evaluation harness.

    python -m src.eval.run --synthetic --signal 0.5 --horizon 1d
    python -m src.eval.run --synthetic --signal 0 --ablation   # control: expect no edge

Real data (needs the news archive, its FinBERT scores, and network for prices):

    python -m src.eval.run --live --universe config/universe_starter.txt --start 2019-01-01
    python -m src.eval.run --live --universe config/universe_starter.txt --snapshot 2026-11-03
                                                        # reproduce an earlier run exactly

A failed gate is a valid result and exits 0; pass --require-pass to exit 1 on a fail.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.eval.ablation import ablation_configs, neutralize
from src.eval.runner import evaluate, render_markdown, save_summary
from src.eval.synthetic import make_synthetic


def _run_live(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from src.eval.live import LiveEvalConfig, render_report_markdown, run_live_eval, save_report
    from src.ingestion.price_cache import cached_sources
    from src.prediction.dataset import default_sources

    tickers = tuple(
        line.strip().upper()
        for line in args.universe.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    cfg = LiveEvalConfig(
        universe=tickers,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        configs=tuple(args.configs),
        train_window=args.train_window,
        test_window=args.test_window,
        backend=args.backend,
        use_news_archive=not args.no_news,
        cost_bps=args.cost_bps,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    snapshot = date.fromisoformat(args.snapshot) if args.snapshot else None
    sources, snapshot_used = cached_sources(default_sources(), snapshot=snapshot)
    report = run_live_eval(cfg, sources)
    day = snapshot_used().isoformat()
    out = args.out or Path(f"docs/results/{day}-{args.horizon}.json")
    save_report(report, out, day)
    print(render_report_markdown(report))
    print(f"\nsaved {out} (price snapshot {day})")
    return 1 if args.require_pass and not report.gate.passed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="evaluate on seeded synthetic data")
    p.add_argument("--live", action="store_true", help="evaluate on real prices, news and scores")
    p.add_argument("--universe", type=Path, default=Path("config/universe_starter.txt"))
    p.add_argument("--start", default="2019-01-01", help="live: first price date")
    p.add_argument("--end", default=None, help="live: last price date (exclusive)")
    p.add_argument(
        "--configs", nargs="+", default=["price_only", "full"], help="live: ablation configs"
    )
    p.add_argument("--snapshot", help="live: reuse the price snapshot saved on this YYYY-MM-DD")
    p.add_argument("--no-news", action="store_true", help="live: price features only")
    p.add_argument("--require-pass", action="store_true", help="live: exit 1 if the gate fails")
    p.add_argument("--signal", type=float, default=0.5, help="planted signal strength (0 = none)")
    p.add_argument("--n-days", type=int, default=1500)
    p.add_argument("--horizon", choices=["1h", "1d", "5d"], default="1d")
    p.add_argument("--backend", choices=["gbr", "linear"], default="gbr")
    p.add_argument("--train-window", type=int, default=252)
    p.add_argument("--test-window", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablation", action="store_true", help="run price-only / per-group / full")
    p.add_argument("--out", type=Path, help="write the summary JSON here")
    args = p.parse_args(argv)
    if args.synthetic == args.live:
        p.error("pass exactly one of --synthetic or --live")
    if args.live:
        return _run_live(args, p)

    data = make_synthetic(n_days=args.n_days, signal=args.signal, seed=args.seed)
    configs = ablation_configs() if args.ablation else {"full": []}
    summaries: dict[str, object] = {}
    for name, dropped in configs.items():
        result = evaluate(
            neutralize(data.features, dropped),
            data.log_returns,
            args.horizon,
            vix=None if "regime" in dropped else data.vix,
            train_window=args.train_window,
            test_window=args.test_window,
            backend=args.backend,
            cost_bps=args.cost_bps,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        summaries[name] = result.summary
        print(f"## {name}\n\n{render_markdown(result.summary)}\n")
    if args.out:
        save_summary(summaries, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
