import json
import subprocess
from pathlib import Path

import pytest

from src.eval import protocol
from src.eval.h2_select import Selection
from src.eval.protocol import (
    H2,
    Protocol,
    ProtocolError,
    claim_holdout,
    complete_holdout,
    freeze,
    verify_freeze,
    write_freeze,
)

FILES = ("a.py",)


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", "-c", "user.name=t",
         "-c", "user.email=t@t", *args],
        check=True, capture_output=True,
    )  # fmt: skip


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _sel(proceed: bool = True) -> Selection:
    reasons = [] if proceed else ["not enough"]
    return Selection({"c": {"fhs": -0.02}}, "p", "n", "n", proceed, reasons)


def _frozen(repo: Path) -> Path:
    path = repo / "out" / "freeze.json"
    freeze(_sel(), repo, path, FILES)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "freeze")
    return path


def test_freeze_then_verify_round_trips(repo: Path) -> None:
    path = _frozen(repo)
    doc = verify_freeze(path, repo, H2, FILES)
    assert doc["frozen"] == {"overall": "n", "best_price": "p", "best_news": "n"}


def test_freeze_refuses_a_dirty_tree_an_existing_freeze_and_a_closed_selection(repo: Path) -> None:
    (repo / "junk.txt").write_text("x")
    with pytest.raises(ProtocolError, match="not clean"):
        freeze(_sel(), repo, repo / "out" / "f.json", FILES)
    (repo / "junk.txt").unlink()
    with pytest.raises(ProtocolError, match="stays closed"):
        freeze(_sel(False), repo, repo / "out" / "f.json", FILES)
    path = _frozen(repo)
    with pytest.raises(ProtocolError, match="already exists"):
        freeze(_sel(), repo, path, FILES)


def test_verify_refuses_a_missing_freeze(repo: Path) -> None:
    with pytest.raises(ProtocolError, match="no freeze file"):
        verify_freeze(repo / "nope.json", repo, H2, FILES)


def test_verify_refuses_changed_code_even_when_committed(repo: Path) -> None:
    path = _frozen(repo)
    (repo / "a.py").write_text("x = 2\n")
    git(repo, "commit", "-q", "-am", "tweak")
    with pytest.raises(ProtocolError, match="frozen code changed"):
        verify_freeze(path, repo, H2, FILES)


def test_verify_refuses_a_dirty_tree(repo: Path) -> None:
    path = _frozen(repo)
    (repo / "scratch.txt").write_text("x")
    with pytest.raises(ProtocolError, match="not clean"):
        verify_freeze(path, repo, H2, FILES)


def test_verify_refuses_an_uncommitted_freeze(repo: Path) -> None:
    path = repo / "out" / "freeze.json"
    freeze(_sel(), repo, path, FILES)
    (repo / ".gitignore").write_text("out/\n")  # clean tree, but the freeze is untracked
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore")
    with pytest.raises(ProtocolError, match="not committed"):
        verify_freeze(path, repo, H2, FILES)


def test_verify_refuses_a_changed_fingerprint_and_a_foreign_commit(repo: Path) -> None:
    path = _frozen(repo)
    doc = json.loads(path.read_text())
    doc["fingerprint"]["train_dates"] = 999
    path.write_text(json.dumps(doc))
    git(repo, "commit", "-q", "-am", "edit freeze")
    with pytest.raises(ProtocolError, match="constants"):
        verify_freeze(path, repo, H2, FILES)

    doc["fingerprint"] = json.loads(json.dumps(protocol.h2_fingerprint()))
    doc["git_commit"] = "0" * 40
    path.write_text(json.dumps(doc))
    git(repo, "commit", "-q", "-am", "edit freeze again")
    with pytest.raises(ProtocolError, match="ancestor|failed"):
        verify_freeze(path, repo, H2, FILES)


def test_the_holdout_can_be_claimed_exactly_once(tmp_path: Path) -> None:
    marker = tmp_path / "run.json"
    doc = {"git_commit": "abc"}
    claim_holdout(marker, doc, H2)
    assert json.loads(marker.read_text())["status"] == "started"
    with pytest.raises(ProtocolError, match="already run or started"):
        claim_holdout(marker, doc, H2)  # a crashed run also burns the shot
    complete_holdout(marker, tmp_path / "holdout.json", H2)
    assert json.loads(marker.read_text())["status"] == "completed"
    with pytest.raises(ProtocolError, match="already run or started"):
        claim_holdout(marker, doc, H2)


def test_code_hash_changes_with_any_frozen_file(repo: Path) -> None:
    before = protocol.code_hash(repo, H2, FILES)
    (repo / "a.py").write_text("x = 3\n")
    assert protocol.code_hash(repo, H2, FILES) != before


def test_the_default_hash_covers_every_source_file_and_the_universe(repo: Path) -> None:
    (repo / "src" / "eval").mkdir(parents=True)
    (repo / "src" / "eval" / "multiple.py").write_text("x = 1\n")
    for name in H2.extra_frozen:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("v1\n")
    files = protocol.frozen_files(repo, H2)
    assert "src/eval/multiple.py" in files and "config/universe_starter.txt" in files
    before = protocol.code_hash(repo, H2)
    for name in ("src/eval/multiple.py", "config/universe_starter.txt", "docs/protocol_h2.md"):
        original = (repo / name).read_text()
        (repo / name).write_text(original + "# edit\n")
        assert protocol.code_hash(repo, H2) != before, name
        (repo / name).write_text(original)
    assert protocol.code_hash(repo, H2) == before


def test_the_real_repo_hashes_every_source_file() -> None:
    root = Path(__file__).resolve().parents[2]
    files = protocol.frozen_files(root, H2)
    assert {"src/eval/runner.py", "src/eval/protocol.py", "src/eval/h2.py"} <= set(files)
    protocol.code_hash(root, H2)  # raises FileNotFoundError if a listed file is missing


def test_a_marker_that_was_ever_committed_blocks_a_run_even_after_deletion(repo: Path) -> None:
    marker = repo / "out" / "holdout_run.json"
    protocol.require_unclaimed(marker, repo, H2)  # nothing yet
    claim_holdout(marker, {"git_commit": "abc"}, H2)
    with pytest.raises(ProtocolError, match="already run or started"):
        protocol.require_unclaimed(marker, repo, H2)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "marker")
    marker.unlink()
    git(repo, "commit", "-q", "-am", "delete the marker")
    with pytest.raises(ProtocolError, match="git history"):
        protocol.require_unclaimed(marker, repo, H2)


H3 = Protocol("H3", ("docs/protocol_h3.md",), lambda: {"hypothesis": "H3", "k": 1})


def _h3_frozen(repo: Path) -> Path:
    path = repo / "out" / "freeze_h3.json"
    write_freeze(H3, {"rule": "ewma94"}, {"ewma94": 0.1}, repo, path, FILES)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "freeze h3")
    return path


def test_a_second_hypothesis_freezes_and_verifies_under_its_own_protocol(repo: Path) -> None:
    path = _h3_frozen(repo)
    doc = verify_freeze(path, repo, H3, FILES)
    assert doc["hypothesis"] == "H3" and doc["frozen"] == {"rule": "ewma94"}


def test_a_freeze_never_verifies_under_another_hypothesis(repo: Path) -> None:
    path = _h3_frozen(repo)
    with pytest.raises(ProtocolError, match="freeze for H3, not H2"):
        verify_freeze(path, repo, H2, FILES)
    h2_path = _frozen(repo)
    with pytest.raises(ProtocolError, match="freeze for H2, not H3"):
        verify_freeze(h2_path, repo, H3, FILES)


def test_each_hypothesis_has_its_own_one_shot_marker(repo: Path) -> None:
    marker = repo / "out" / "holdout_run.json"
    claim_holdout(marker, {"git_commit": "abc"}, H3)
    assert json.loads(marker.read_text())["hypothesis"] == "H3"
    with pytest.raises(ProtocolError, match="H3 holdout was already run"):
        protocol.require_unclaimed(marker, repo, H3)


def test_the_extra_frozen_files_belong_to_the_protocol(repo: Path) -> None:
    assert protocol.frozen_files(repo, H3)[0] == "docs/protocol_h3.md"
    assert protocol.frozen_files(repo, H2)[0] == "docs/protocol_h2.md"


def test_a_marker_owned_by_another_hypothesis_is_named_not_mistaken_for_ours(repo: Path) -> None:
    marker = repo / "out" / "holdout_run.json"
    claim_holdout(marker, {"git_commit": "abc"}, H2)
    with pytest.raises(ProtocolError, match="belongs to H2, not H3"):
        protocol.require_unclaimed(marker, repo, H3)
    with pytest.raises(ProtocolError, match="belongs to H2, not H3"):
        claim_holdout(marker, {"git_commit": "abc"}, H3)
    with pytest.raises(ProtocolError, match="belongs to H2, not H3"):
        complete_holdout(marker, repo / "r.json", H3)


def test_a_fingerprint_that_names_another_hypothesis_is_refused(repo: Path) -> None:
    liar = Protocol("H3", ("docs/protocol_h3.md",), lambda: {"hypothesis": "H2"})
    with pytest.raises(ProtocolError, match="must agree"):
        write_freeze(liar, {}, {}, repo, repo / "out" / "f.json", FILES)


def test_an_old_format_freeze_without_a_hypothesis_is_named_as_such(repo: Path) -> None:
    path = _frozen(repo)
    doc = json.loads(path.read_text())
    del doc["hypothesis"]
    path.write_text(json.dumps(doc))
    git(repo, "commit", "-q", "-am", "old format")
    with pytest.raises(ProtocolError, match="no recorded hypothesis"):
        verify_freeze(path, repo, H2, FILES)


def test_the_h2_freeze_document_keeps_its_shape(repo: Path) -> None:
    doc = freeze(_sel(), repo, repo / "out" / "f.json", FILES)
    assert set(doc) == {
        "hypothesis", "frozen", "dev_scores", "fingerprint", "context", "code_sha256",
        "git_commit", "created_utc",
    }  # fmt: skip
    assert json.loads(json.dumps(doc["fingerprint"])) == json.loads(
        json.dumps(protocol.h2_fingerprint())
    )
    assert doc["dev_scores"] == {"c": {"fhs": -0.02}}
