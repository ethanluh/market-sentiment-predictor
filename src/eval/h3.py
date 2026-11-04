"""
Run the H3 protocol (``docs/protocol_h3.md``): development selection, freeze, then the one-shot holdout.

    python -m src.eval.h3 dev      --universe config/universe_starter.txt --snapshot 2026-11-02
    python -m src.eval.h3 freeze   # after committing the dev result; refuses a dirty tree
    python -m src.eval.h3 holdout  --universe config/universe_starter.txt --snapshot 2026-11-02

The development step sees only returns up to ``DEV_END``. The holdout step verifies the freeze, claims
the one-shot marker and only then evaluates; it cannot be run twice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import protocol, voltarget
from src.eval.h2 import _universe  # the one universe-file parser
from src.eval.live import _clean
from src.eval.significance import block_bootstrap_ci

OUT_DIR = Path("docs/results/h3")
START = "2019-09-01"  # the same window as the H2 snapshot: warm-up before the first scored bar
COST_SENSITIVITY = (0.0, 20.0)
SCORE_KEYS = ("vv_ratio", "ratio_hi", "sharpe_diff", "sharpe_diff_lo", "h3a", "h3b")


def fingerprint() -> dict[str, object]:
    """Everything the protocol fixes that is not a file."""
    return {
        "hypothesis": "H3",
        "start": START,
        "dev_start": voltarget.DEV_START,
        "dev_end": voltarget.DEV_END,
        "holdout_start": voltarget.HOLDOUT_START,
        "forecasters": list(voltarget.FORECASTERS),
        "cost_bps": voltarget.COST_BPS,
        "min_tickers": voltarget.MIN_TICKERS,
        "rv_window": voltarget.RV_WINDOW,
        "block": voltarget.BLOCK,
        "n_boot": voltarget.N_BOOT,
        "seed": voltarget.SEED,
        "max_vv_ratio": voltarget.MAX_VV_RATIO,
        "min_sharpe_diff": voltarget.MIN_SHARPE_DIFF,
        "tie_tolerance": voltarget.TIE_TOLERANCE,
        "cost_sensitivity": list(COST_SENSITIVITY),
    }


H3 = protocol.Protocol("H3", ("docs/protocol_h3.md", "config/universe_starter.txt"), fingerprint)


def load_returns(universe: tuple[str, ...], sources: object) -> pd.DataFrame:
    """Daily log returns, one column per ticker, on the union of their dates (gaps stay NaN)."""
    cols = {
        t: sources.prices(t, START, None, "1d")["log_return"]  # type: ignore[attr-defined]
        for t in universe
    }
    return pd.DataFrame(cols).sort_index()


def truncate(log_returns: pd.DataFrame, end: str) -> pd.DataFrame:
    """The returns as they were at ``end``: later bars are removed, not just unscored."""
    day = log_returns.index.tz_convert("UTC").normalize()
    return log_returns[np.asarray(day <= pd.Timestamp(end, tz="UTC"))]


def _per_ticker(
    log_returns: pd.DataFrame, weights: pd.DataFrame, start: str, end: str | None
) -> dict[str, dict[str, object]]:
    """
    Point estimates only: the single-name dispersion ratio and Sharpe difference. A name that is not
    available on every bar of the period is reported as skipped with the reason, never scored on
    spliced windows, and never allowed to stop the headline result.
    """
    out: dict[str, dict[str, object]] = {}
    ok = weights.shift(1).notna() & log_returns.notna()
    day = ok.index.tz_convert("UTC").normalize()
    in_period = np.asarray(day >= pd.Timestamp(start, tz="UTC"))
    if end is not None:
        in_period &= np.asarray(day <= pd.Timestamp(end, tz="UTC"))
    for t in log_returns.columns:
        avail = ok[t].to_numpy()[in_period]
        if avail.sum() < 2 * voltarget.RV_WINDOW:
            out[t] = {"skipped": f"only {int(avail.sum())} available bars"}
        elif not avail.all():
            out[t] = {"skipped": f"{int((~avail).sum())} missing bars in the period"}
        else:
            bt = voltarget.backtest(
                log_returns[[t]], weights[[t]], min_tickers=1, scored_from=start
            ).period(start, end)
            rv_s = voltarget.rv21(bt.net).dropna().to_numpy()
            rv_b = voltarget.rv21(bt.buyhold).dropna().to_numpy()
            out[t] = {
                "vv_ratio": voltarget.dispersion(rv_s) / voltarget.dispersion(rv_b),
                "sharpe_diff": voltarget.net_sharpe(bt.net.to_numpy())
                - voltarget.net_sharpe(bt.buyhold.to_numpy()),
            }
    return out


def _sensitivity(
    log_returns: pd.DataFrame, weights: pd.DataFrame, start: str, end: str | None
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bps in COST_SENSITIVITY:
        bt = voltarget.backtest(log_returns, weights, cost_bps=bps).period(start, end)
        out[str(bps)] = {
            "sharpe": voltarget.net_sharpe(bt.net.to_numpy()),
            "sharpe_diff": voltarget.net_sharpe(bt.net.to_numpy())
            - voltarget.net_sharpe(bt.buyhold.to_numpy()),
        }
    return out


def _score(
    log_returns: pd.DataFrame, name: str, target: float, start: str, end: str | None
) -> tuple[voltarget.Backtest, dict[str, object]]:
    sigma = voltarget.annualised_forecasts(log_returns, name)
    weights = voltarget.target_weights(sigma, target)
    bt = voltarget.backtest(log_returns, weights).period(start, end)
    result = voltarget.evaluate(bt)
    result["target"] = target
    result["cost_sensitivity"] = _sensitivity(log_returns, weights, start, end)
    result["per_ticker"] = _per_ticker(log_returns, weights, start, end)
    return bt, result


def run_dev(log_returns: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Every forecaster on development, each with its own target calibrated on development."""
    dev = truncate(log_returns, voltarget.DEV_END)
    results: dict[str, dict[str, object]] = {}
    for name in voltarget.FORECASTERS:
        sigma = voltarget.annualised_forecasts(dev, name)
        target = voltarget.calibrate_target(sigma, voltarget.DEV_START, voltarget.DEV_END)
        _, results[name] = _score(dev, name, target, voltarget.DEV_START, voltarget.DEV_END)
    return results


def frozen_from(
    sel: voltarget.Selection, results: dict[str, dict[str, object]]
) -> dict[str, object]:
    return {
        "forecaster": sel.winner,
        "targets": {n: results[n]["target"] for n in voltarget.FORECASTERS},
    }


def freeze(
    sel: voltarget.Selection,
    results: dict[str, dict[str, object]],
    repo: Path,
    path: Path,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Freeze the winner and all three targets; refuses a result the protocol says must not reach the holdout."""
    if not sel.proceed:
        raise protocol.ProtocolError(
            "the development result does not clear the protocol; the holdout stays closed: "
            + "; ".join(sel.reasons)
        )
    scores = {n: {k: results[n][k] for k in SCORE_KEYS} for n in voltarget.FORECASTERS}
    return protocol.write_freeze(
        H3, frozen_from(sel, results), dict(scores), repo, path, None, context
    )


def run_holdout(log_returns: pd.DataFrame, frozen: dict[str, object]) -> dict[str, object]:
    """The frozen forecaster at its frozen target on the holdout, plus H3c against ``rolling63``."""
    name = str(frozen["forecaster"])
    targets: dict[str, float] = frozen["targets"]  # type: ignore[assignment]
    start = voltarget.HOLDOUT_START
    bt, result = _score(log_returns, name, targets[name], start, None)
    report: dict[str, object] = {
        "forecaster": name,
        "target": targets[name],
        "result": result,
        "passed": bool(result["h3a"] and result["h3b"]),
    }
    if name == "rolling63":
        report["h3c"] = {"applicable": False}
        return report
    rolling, _ = _score(log_returns, "rolling63", targets["rolling63"], start, None)
    rv_f, rv_r = voltarget.rv21(bt.net).dropna(), voltarget.rv21(rolling.net).dropna()
    if not rv_f.index.equals(rv_r.index):
        raise ValueError("H3c needs the two forecasters on the same bars")
    a, b = rv_f.to_numpy(), rv_r.to_numpy()
    _, hi = block_bootstrap_ci(
        lambda x, y: voltarget.dispersion(x) / voltarget.dispersion(y),
        a,
        b,
        block=voltarget.BLOCK,
        n_boot=voltarget.N_BOOT,
        seed=voltarget.SEED,
        level=0.90,
    )
    ratio = voltarget.dispersion(a) / voltarget.dispersion(b)
    report["h3c"] = {"applicable": True, "vv_ratio_vs_rolling63": ratio, "ratio_hi": hi,
                     "lower": bool(ratio < 1.0)}  # fmt: skip
    return report


def _save(doc: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(doc), indent=2) + "\n", encoding="utf-8")


def _returns_from_args(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    from src.ingestion.price_cache import cached_sources
    from src.prediction.dataset import default_sources

    snapshot = date.fromisoformat(args.snapshot) if args.snapshot else None
    sources, used = cached_sources(default_sources(), snapshot=snapshot)
    returns = load_returns(_universe(args.universe), sources)
    return returns, used().isoformat()


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
        returns, snapshot = _returns_from_args(args)
        results = run_dev(returns)
        sel = voltarget.select(results)
        _save(
            {
                "price_snapshot": snapshot,
                "universe": list(_universe(args.universe)),
                "code_sha256": protocol.code_hash(repo, H3),
                "results": results,
                "selection": asdict(sel),
            },
            dev_path,
        )
        print(json.dumps(_clean(asdict(sel)), indent=2))
        print(f"saved {dev_path}; proceed to the holdout: {sel.proceed}")
        return 0

    if args.step == "freeze":
        dev = json.loads(dev_path.read_text(encoding="utf-8"))
        if dev["code_sha256"] != protocol.code_hash(repo, H3):
            raise protocol.ProtocolError(
                "the code changed since the development run; rerun dev before freezing"
            )
        sel = voltarget.Selection(**dev["selection"])
        context = {"universe": dev["universe"], "dev_snapshot": dev["price_snapshot"]}
        doc = freeze(sel, dev["results"], repo, freeze_path, context=context)
        print(f"froze {doc['frozen']} at {doc['git_commit']}; commit {freeze_path} now")
        return 0

    protocol.require_unclaimed(marker, repo, H3)
    freeze_doc = protocol.verify_freeze(freeze_path, repo, H3)
    if not args.snapshot:
        parser.error("the holdout needs an explicit --snapshot so the data is pinned and recorded")
    if list(_universe(args.universe)) != freeze_doc["context"]["universe"]:  # type: ignore[index]
        raise protocol.ProtocolError("the universe differs from the one frozen with the selection")
    if args.snapshot != freeze_doc["context"]["dev_snapshot"]:  # type: ignore[index]
        raise protocol.ProtocolError(
            f"the holdout snapshot {args.snapshot} differs from the development snapshot "
            f"{freeze_doc['context']['dev_snapshot']}; adjusted closes may have been revised"  # type: ignore[index]
        )
    returns, snapshot = _returns_from_args(args)  # loading shows no outcomes, so a failure is free
    protocol.claim_holdout(marker, freeze_doc, H3)  # before any evaluation: the shot is spent here
    report = run_holdout(returns, freeze_doc["frozen"])  # type: ignore[arg-type]
    report["price_snapshot"] = snapshot
    _save(report, holdout_path)
    protocol.complete_holdout(marker, holdout_path, H3)
    print(json.dumps(_clean({k: report[k] for k in ("passed", "h3c")}), indent=2))
    print(f"saved {holdout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
