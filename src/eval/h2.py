"""
Run the H2 protocol (``docs/protocol_h2.md``): development selection, freeze, then the one-shot holdout.

    python -m src.eval.h2 dev      --universe config/universe_starter.txt --snapshot 2026-11-03
    python -m src.eval.h2 freeze   # after committing the dev result; refuses a dirty tree
    python -m src.eval.h2 holdout  --universe config/universe_starter.txt --snapshot 2026-11-03

The development step sees only data up to ``DEV_END``. The holdout step verifies the freeze, claims
the one-shot marker and only then evaluates; it cannot be run twice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.eval import protocol
from src.eval.h2_gate import H2GateRule, apply_h2_gate, compare_pipelines
from src.eval.h2_select import Selection, select
from src.eval.live import _clean
from src.eval.pooled import pooled_walk_forward
from src.eval.runner import EvalResult
from src.prediction.vol_scaled import CANDIDATES

OUT_DIR = Path("docs/results/h2")
START = "2019-09-01"  # prices start before the news archive so features warm up


@dataclass
class Panel:
    features: dict[str, pd.DataFrame]
    log_returns: dict[str, pd.Series]
    vix: pd.Series


def load_panel(universe: tuple[str, ...], start: str, sources: object | None = None) -> Panel:
    """Per-ticker strict datasets from the shared builder (the same path training and eval use)."""
    from src.prediction.dataset import DatasetConfig, build_dataset

    feats: dict[str, pd.DataFrame] = {}
    rets: dict[str, pd.Series] = {}
    vix: pd.Series | None = None
    for ticker in universe:
        ds = build_dataset(
            DatasetConfig(
                ticker=ticker,
                start=start,
                peers=tuple(t for t in universe if t != ticker),
                use_vix=True,
                use_news_archive=True,
            ),
            sources,  # type: ignore[arg-type]
            regime="none",
        )
        feats[ticker], rets[ticker] = ds.features, ds.log_returns
        if ds.vix is None:
            raise RuntimeError(f"{ticker}: the dataset has no VIX series")
        vix = ds.vix if vix is None else vix
    assert vix is not None
    return Panel(feats, rets, vix)


def truncate(panel: Panel, end: str) -> Panel:
    """The panel as it was at ``end``: later features, returns and VIX are removed, not just unscored."""
    cut = pd.Timestamp(end, tz="UTC")
    return Panel(
        {t: f.loc[f.index <= cut] for t, f in panel.features.items()},
        {t: r.loc[r.index <= cut] for t, r in panel.log_returns.items()},
        panel.vix.loc[panel.vix.index <= cut],
    )


def run_dev(
    panel: Panel,
    candidate_ids: tuple[str, ...] = tuple(CANDIDATES),
    n_boot: int = 100,
) -> dict[str, dict[str, EvalResult]]:
    """Every candidate on development rows only (data after ``DEV_END`` is removed first)."""
    dev = truncate(panel, protocol.DEV_END)
    cache: dict = {}
    return {
        cid: pooled_walk_forward(
            dev.features,
            dev.log_returns,
            dev.vix,
            cid,
            label_end_max=protocol.DEV_END,
            n_boot=n_boot,
            baseline_cache=cache,
        )
        for cid in candidate_ids
    }


def run_holdout(
    panel: Panel,
    frozen: dict[str, str],
    rule: H2GateRule = H2GateRule(),
    holdout_start: str = protocol.HOLDOUT_START,
    n_boot: int = 500,
) -> dict[str, object]:
    """The frozen pipelines and baselines on holdout rows; H2a on the overall winner, H2b news vs price."""
    cache: dict = {}
    ids = sorted({frozen["overall"], frozen["best_price"], frozen["best_news"]})
    results = {
        cid: pooled_walk_forward(
            panel.features,
            panel.log_returns,
            panel.vix,
            cid,
            score_start=holdout_start,
            n_boot=n_boot,
            baseline_cache=cache,
        )
        for cid in ids
    }
    h2a = apply_h2_gate(results[frozen["overall"]], rule)
    h2b = compare_pipelines(results[frozen["best_news"]], results[frozen["best_price"]], rule)
    return {
        "frozen": frozen,
        "h2a": {"passed": h2a.passed, "reasons": h2a.reasons, "details": h2a.details},
        "h2b": {"passed": h2b.passed, "reasons": h2b.reasons, "details": h2b.details},
        "summaries": {c: {t: r.summary for t, r in rs.items()} for c, rs in results.items()},
    }


def _save(doc: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(doc), indent=2) + "\n", encoding="utf-8")


def _universe(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _panel_from_args(args: argparse.Namespace) -> tuple[Panel, str]:
    from src.ingestion.price_cache import cached_sources
    from src.prediction.dataset import default_sources

    snapshot = date.fromisoformat(args.snapshot) if args.snapshot else None
    sources, used = cached_sources(default_sources(), snapshot=snapshot)
    panel = load_panel(_universe(args.universe), START, sources)
    return panel, used().isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    parser.add_argument("step", choices=["dev", "freeze", "holdout"])
    parser.add_argument("--universe", type=Path, default=Path("config/universe_starter.txt"))
    parser.add_argument("--snapshot", help="reuse the price snapshot saved on this YYYY-MM-DD")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    repo = Path(".").resolve()
    dev_path, freeze_path = args.out_dir / "dev.json", args.out_dir / "freeze.json"
    marker, holdout_path = args.out_dir / "holdout_run.json", args.out_dir / "holdout.json"

    if args.step == "dev":
        panel, snapshot = _panel_from_args(args)
        results = run_dev(panel)
        sel = select(results)
        _save(
            {
                "price_snapshot": snapshot,
                "universe": list(_universe(args.universe)),
                "code_sha256": protocol.code_hash(repo),
                "selection": asdict(sel),
            },
            dev_path,
        )
        print(json.dumps(_clean(asdict(sel)), indent=2))
        print(f"saved {dev_path}; proceed to the holdout: {sel.proceed}")
        return 0

    if args.step == "freeze":
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        if dev["code_sha256"] != protocol.code_hash(repo):
            raise protocol.ProtocolError(
                "the code changed since the development run; rerun dev before freezing"
            )
        sel = Selection(**dev["selection"])
        context = {"universe": dev["universe"], "dev_snapshot": dev["price_snapshot"]}
        doc = protocol.freeze(sel, repo, freeze_path, context=context)
        print(f"froze {doc['frozen']} at {doc['git_commit']}; commit {freeze_path} now")
        return 0

    protocol.require_unclaimed(marker, repo)
    freeze_doc = protocol.verify_freeze(freeze_path, repo)
    if not args.snapshot:
        parser.error("the holdout needs an explicit --snapshot so the data is pinned and recorded")
    if list(_universe(args.universe)) != freeze_doc["context"]["universe"]:  # type: ignore[index]
        raise protocol.ProtocolError("the universe differs from the one frozen with the selection")
    panel, snapshot = _panel_from_args(args)  # loading shows no outcomes, so a load failure is free
    protocol.claim_holdout(marker, freeze_doc)  # before any evaluation: the shot is spent from here
    report = run_holdout(panel, freeze_doc["frozen"])  # type: ignore[arg-type]
    report["price_snapshot"] = snapshot
    _save(report, holdout_path)
    protocol.complete_holdout(marker, holdout_path)
    print(json.dumps(_clean({k: report[k] for k in ("h2a", "h2b")}), indent=2))
    print(f"saved {holdout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
