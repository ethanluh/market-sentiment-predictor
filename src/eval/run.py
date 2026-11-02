"""
CLI for the evaluation harness.

    python -m src.eval.run --synthetic --signal 0.5 --horizon 1d
    python -m src.eval.run --synthetic --signal 0 --ablation   # control: expect no edge

Live-data runs are not wired in yet; the harness itself is data-agnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.ablation import ablation_configs, neutralize
from src.eval.runner import evaluate, render_markdown, save_summary
from src.eval.synthetic import make_synthetic


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="evaluate on seeded synthetic data")
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
    if not args.synthetic:
        p.error("live data is not wired in yet; pass --synthetic")

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
