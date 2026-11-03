"""
One-shot enforcement for the H2 holdout.

``freeze`` records the chosen candidates with a fingerprint of the configuration and a hash of the
code that defines them. ``verify_freeze`` refuses to continue when anything differs, when the git
tree is dirty, or when the freeze file is not committed. ``claim_holdout`` creates a marker
exclusively before the holdout is evaluated, so a second run in the same checkout, or a rerun after
a crash, is refused.

This is a process guard, not tamper-proofing: the marker is a file, so deleting it, or running from
another clone, bypasses it. Commit the marker with the results (it records the freeze commit), and a
marker that was ever committed blocks a later run in any checkout that has that history. After a crash
following the claim, delete the untracked marker deliberately and say so in the write-up.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.eval import h2_gate, h2_select, pooled
from src.eval.h2_select import Selection
from src.prediction import vol_scaled

HYPOTHESIS = "H2"
DEV_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
POOL_START = pooled.POOL_START
EXTRA_FROZEN = ("docs/protocol_h2.md", "config/universe_starter.txt")


class ProtocolError(RuntimeError):
    """A pre-registration rule was about to be broken."""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise ProtocolError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _require_clean_tree(repo: Path) -> None:
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise ProtocolError(f"the git tree is not clean; commit or discard first:\n{dirty}")


def frozen_files(repo: Path) -> tuple[str, ...]:
    """Every Python file under ``src/`` plus the protocol text and the universe: any change to code
    that can affect a result invalidates the freeze."""
    py = sorted(str(p.relative_to(repo)) for p in (repo / "src").rglob("*.py"))
    return (*EXTRA_FROZEN, *py)


def code_hash(repo: Path, files: tuple[str, ...] | None = None) -> str:
    files = frozen_files(repo) if files is None else files
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode() + b"\0" + (repo / name).read_bytes() + b"\0")
    return h.hexdigest()


def fingerprint() -> dict[str, object]:
    """Everything the protocol fixes that is not a file: constants, candidates, gate and selection rules."""
    return {
        "hypothesis": HYPOTHESIS,
        "dev_end": DEV_END,
        "holdout_start": HOLDOUT_START,
        "pool_start": POOL_START,
        "train_dates": pooled.TRAIN_DATES,
        "refit_every": pooled.REFIT_EVERY,
        "baseline_window": pooled.BASELINE_WINDOW,
        "seed": vol_scaled.SEED,
        "candidates": [asdict(c) for c in (vol_scaled.CONTROL, *vol_scaled.COMPLEXITY_ORDER)],
        "gate": asdict(h2_gate.H2GateRule()),
        "tie_tolerance": h2_select.TIE_TOLERANCE,
        "materiality": h2_select.MATERIALITY,
    }


def freeze(
    selection: Selection,
    repo: Path,
    path: Path,
    files: tuple[str, ...] | None = None,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Write the freeze file (``context`` is stored as given: the universe and the dev snapshot). Refuses an uncommitted tree, an existing freeze, and a selection the
    protocol says must not reach the holdout.
    """
    if not selection.proceed:
        raise ProtocolError(
            "the development result does not clear the protocol; the holdout stays closed: "
            + "; ".join(selection.reasons)
        )
    _require_clean_tree(repo)
    doc: dict[str, object] = {
        "frozen": {
            "overall": selection.overall,
            "best_price": selection.best_price,
            "best_news": selection.best_news,
        },
        "dev_scores": selection.scores,
        "fingerprint": fingerprint(),
        "context": context or {},
        "code_sha256": code_hash(repo, files),
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
    except FileExistsError:
        raise ProtocolError(f"{path} already exists; a freeze is never overwritten") from None
    return doc


def verify_freeze(
    path: Path, repo: Path, files: tuple[str, ...] | None = None
) -> dict[str, object]:
    """Load the freeze and check it against the current code, tree and history; raise on any mismatch."""
    if not path.is_file():
        raise ProtocolError(f"no freeze file at {path}; run the freeze step first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc["fingerprint"] != json.loads(json.dumps(fingerprint())):
        raise ProtocolError("the protocol constants, candidates or rules differ from the freeze")
    if doc["code_sha256"] != code_hash(repo, files):
        raise ProtocolError("frozen code changed after the freeze; the freeze no longer applies")
    _require_clean_tree(repo)
    try:
        _git(repo, "ls-files", "--error-unmatch", str(path))
    except ProtocolError:
        raise ProtocolError(
            f"{path} is not committed; commit the freeze before the holdout"
        ) from None
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", doc["git_commit"], "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ProtocolError("the freeze commit is not an ancestor of HEAD")
    return doc


def require_unclaimed(marker: Path, repo: Path) -> None:
    """Raise if the marker exists, or was ever committed (even if later deleted)."""
    if marker.exists():
        raise ProtocolError(_ALREADY.format(marker=marker, h=HYPOTHESIS))
    if _git(repo, "log", "--all", "--oneline", "--", str(marker.resolve())):
        raise ProtocolError(
            f"{marker} appears in git history: the {HYPOTHESIS} holdout was already run. "
            "A new test needs a new hypothesis id"
        )


_ALREADY = (
    "{marker} exists: the {h} holdout was already run or started. It is not run twice; a new test "
    "needs a new hypothesis id and is reported as a second test"
)


def claim_holdout(marker: Path, freeze_doc: dict[str, object]) -> None:
    """Create the one-shot marker; a second claim, or a claim after a crashed run, raises."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x", encoding="utf-8") as fh:
            json.dump(
                {
                    "hypothesis": HYPOTHESIS,
                    "freeze_commit": freeze_doc["git_commit"],
                    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "status": "started",
                },
                fh,
                indent=2,
            )
            fh.write("\n")
    except FileExistsError:
        raise ProtocolError(_ALREADY.format(marker=marker, h=HYPOTHESIS)) from None


def complete_holdout(marker: Path, results: Path) -> None:
    doc = json.loads(marker.read_text(encoding="utf-8"))
    doc.update(status="completed", results=str(results))
    marker.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
