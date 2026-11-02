import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion import news_archive as na
from src.ingestion.news import NewsArticle
from src.sentiment.archive_scoring import merge_caches, score_archive

DAY = datetime(2024, 1, 5, tzinfo=timezone.utc)


def art(url: str, text: str, day: datetime = DAY) -> NewsArticle:
    return NewsArticle(headline=text, body="", source="benzinga", url=url, published_at=day)


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(na, "ARCHIVE_ROOT", tmp_path / "raw")
    monkeypatch.setattr(na, "SCORE_CACHE_ROOT", tmp_path / "scores")
    return tmp_path


class Fake:
    """Scores a text by its length, so results are predictable and calls are countable."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str]) -> list[float]:
        self.texts.extend(texts)
        return [min(len(t) / 100.0, 1.0) for t in texts]


def test_scores_land_in_each_tickers_cache(archive: Path) -> None:
    na.write_day("AAA", DAY, [art("u1", "x" * 10), art("u2", "y" * 50)])
    fake = Fake()
    report = score_archive(["AAA"], fake, batch_size=4, log=lambda m: None)
    # text is "headline." (the empty body is stripped), so 10 and 50 characters score as 11 and 51
    assert na._load_score_cache("AAA") == {"u1": pytest.approx(0.11), "u2": pytest.approx(0.51)}
    assert (report.articles, report.already_scored, report.scored_now) == (2, 0, 2)


def test_a_url_shared_by_two_tickers_is_scored_once(archive: Path) -> None:
    shared = art("shared", "z" * 30)
    na.write_day("AAA", DAY, [shared, art("a-only", "a" * 20)])
    na.write_day("BBB", DAY, [shared, art("b-only", "b" * 20)])
    fake = Fake()
    report = score_archive(["AAA", "BBB"], fake, batch_size=8, log=lambda m: None)
    assert sum("z" * 30 in t for t in fake.texts) == 1 and report.scored_now == 3
    assert na._load_score_cache("AAA")["shared"] == na._load_score_cache("BBB")["shared"]


def test_a_rerun_scores_nothing_new(archive: Path) -> None:
    na.write_day("AAA", DAY, [art("u1", "x" * 10)])
    score_archive(["AAA"], Fake(), log=lambda m: None)
    fake = Fake()
    report = score_archive(["AAA"], fake, log=lambda m: None)
    assert fake.texts == [] and report.scored_now == 0 and report.already_scored == 1


def test_an_interrupted_run_resumes_from_its_flushed_batches(archive: Path) -> None:
    na.write_day("AAA", DAY, [art(f"u{i}", "t" * (i + 1)) for i in range(6)])
    calls = {"n": 0}

    def dies_on_third_batch(texts: list[str]) -> list[float]:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return [0.1] * len(texts)

    with pytest.raises(KeyboardInterrupt):
        score_archive(["AAA"], dies_on_third_batch, batch_size=2, flush_texts=2, log=lambda m: None)
    assert len(na._load_score_cache("AAA")) == 4  # two flushed batches survived
    fake = Fake()
    report = score_archive(["AAA"], fake, batch_size=2, log=lambda m: None)
    assert len(fake.texts) == 2 and report.scored_now == 2 and len(na._load_score_cache("AAA")) == 6


def test_texts_are_scored_shortest_first_to_limit_padding(archive: Path) -> None:
    na.write_day("AAA", DAY, [art("u1", "long" * 20), art("u2", "s"), art("u3", "mid" * 5)])
    fake = Fake()
    score_archive(["AAA"], fake, batch_size=1, log=lambda m: None)
    assert [len(t) for t in fake.texts] == sorted(len(t) for t in fake.texts)


def test_since_and_max_articles_limit_the_work(archive: Path) -> None:
    old = datetime(2023, 1, 5, tzinfo=timezone.utc)
    na.write_day("AAA", old, [art("old", "o" * 5, old)])
    na.write_day("AAA", DAY, [art("n1", "n" * 5), art("n2", "m" * 9)])
    report = score_archive(
        ["AAA"], Fake(), since=datetime(2024, 1, 1, tzinfo=timezone.utc), log=lambda m: None
    )
    assert set(na._load_score_cache("AAA")) == {"n1", "n2"} and report.articles == 2
    na.write_day("BBB", DAY, [art("b1", "b" * 5), art("b2", "c" * 9)])
    assert score_archive(["BBB"], Fake(), max_articles=1, log=lambda m: None).scored_now == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -2.0])
def test_invalid_scores_raise_and_are_never_cached(archive: Path, bad: float) -> None:
    na.write_day("AAA", DAY, [art("u1", "x")])
    with pytest.raises(RuntimeError, match="invalid sentiment score"):
        score_archive(["AAA"], lambda texts: [bad] * len(texts), log=lambda m: None)
    assert na._load_score_cache("AAA") == {}


def test_a_wrong_number_of_scores_raises(archive: Path) -> None:
    na.write_day("AAA", DAY, [art("u1", "x"), art("u2", "y")])
    with pytest.raises(RuntimeError, match="returned 1 scores for 2 texts"):
        score_archive(["AAA"], lambda texts: [0.0], batch_size=2, log=lambda m: None)


def test_corrupt_archive_files_and_caches_raise(archive: Path) -> None:
    na.write_day("AAA", DAY, [art("u1", "x")])
    next((na.ARCHIVE_ROOT / "AAA").glob("*.json")).write_text("{broken")
    with pytest.raises(ValueError, match="corrupt news archive file"):
        score_archive(["AAA"], Fake(), log=lambda m: None)
    (na.ARCHIVE_ROOT / "AAA").joinpath("2024-01-05.json").write_text(json.dumps([]))
    na.SCORE_CACHE_ROOT.mkdir(parents=True)
    (na.SCORE_CACHE_ROOT / "AAA.json").write_text("not json")
    with pytest.raises(ValueError, match="corrupt score cache"):
        score_archive(["AAA"], Fake(), log=lambda m: None)


def test_a_crash_between_temp_write_and_rename_keeps_the_old_cache(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    na._save_score_cache("AAA", {"old": 0.1})

    def crash(src, dst):  # type: ignore[no-untyped-def]
        raise OSError("power cut")

    with monkeypatch.context() as crashing:  # scoped, so the archive fixture's paths stay patched
        crashing.setattr(na.os, "replace", crash)
        with pytest.raises(OSError):
            na._save_score_cache("AAA", {"old": 0.1, "new": 0.2})
    assert na._load_score_cache("AAA") == {"old": 0.1}  # the previous file is intact
    na._save_score_cache("AAA", {"old": 0.1, "new": 0.2})  # a leftover .tmp does not block saving
    assert na._load_score_cache("AAA") == {"old": 0.1, "new": 0.2}


def test_a_successful_save_leaves_no_temp_file(archive: Path) -> None:
    na._save_score_cache("AAA", {"u": 0.5})
    assert [p.name for p in na.SCORE_CACHE_ROOT.iterdir()] == ["AAA.json"]


def test_a_url_cached_for_one_ticker_is_copied_to_another_not_rescored(archive: Path) -> None:
    shared = art("shared", "z" * 30)
    na.write_day("AAA", DAY, [shared])
    na.write_day("BBB", DAY, [shared, art("b-only", "b" * 20)])
    na._save_score_cache("AAA", {"shared": 0.3})
    fake = Fake()
    report = score_archive(["AAA", "BBB"], fake, log=lambda m: None)
    assert len(fake.texts) == 1 and "z" * 30 not in "".join(fake.texts)  # only b-only was scored
    assert na._load_score_cache("BBB")["shared"] == 0.3 and report.scored_now == 1


def test_work_done_before_a_failure_is_kept_even_without_a_periodic_flush(archive: Path) -> None:
    na.write_day("AAA", DAY, [art(f"u{i}", "t" * (i + 1)) for i in range(6)])
    calls = {"n": 0}

    def bad_third_batch(texts: list[str]) -> list[float]:
        calls["n"] += 1
        return [float("nan")] * len(texts) if calls["n"] == 3 else [0.2] * len(texts)

    with pytest.raises(RuntimeError, match="invalid sentiment score"):
        score_archive(
            ["AAA"], bad_third_batch, batch_size=2, flush_texts=10_000, log=lambda m: None
        )
    assert (
        len(na._load_score_cache("AAA")) == 4
    )  # first two batches survive; the bad one never lands


@pytest.mark.parametrize(
    "raw", ['{"u": 1.5}', '{"u": NaN}', '{"u": Infinity}', '{"u": null}', "[1, 2]"]
)
def test_cached_scores_are_validated_on_load(archive: Path, raw: str) -> None:
    na.SCORE_CACHE_ROOT.mkdir(parents=True)
    (na.SCORE_CACHE_ROOT / "AAA.json").write_text(raw)
    with pytest.raises(ValueError):
        na._load_score_cache("AAA")


def test_merge_adds_scores_and_never_drops_local_ones(archive: Path, tmp_path: Path) -> None:
    na._save_score_cache("AAA", {"local-only": 0.4, "both": 0.5})
    other = tmp_path / "remote"
    other.mkdir()
    (other / "AAA.json").write_text(json.dumps({"both": 0.5004, "remote-only": -0.2}))
    assert merge_caches(other, ["AAA", "BBB"]) == 1  # BBB has no remote file
    assert na._load_score_cache("AAA") == {"local-only": 0.4, "both": 0.5, "remote-only": -0.2}


def test_merge_refuses_disagreement_and_invalid_scores(archive: Path, tmp_path: Path) -> None:
    na._save_score_cache("AAA", {"u": 0.5})
    other = tmp_path / "remote"
    other.mkdir()
    (other / "AAA.json").write_text(json.dumps({"u": 0.9}))
    with pytest.raises(ValueError, match="local 0.5 vs incoming 0.9"):
        merge_caches(other, ["AAA"])
    (other / "AAA.json").write_text('{"v": 2.0}')
    with pytest.raises(ValueError, match="invalid score"):
        merge_caches(other, ["AAA"])
    (other / "AAA.json").write_text("garbage")
    with pytest.raises(ValueError, match="corrupt score cache"):
        merge_caches(other, ["AAA"])
    assert na._load_score_cache("AAA") == {"u": 0.5}  # nothing was changed by the failures


def test_since_keeps_an_explicit_timezone() -> None:
    spec = importlib.util.spec_from_file_location("score_cli", Path("scripts/score_archive.py"))
    cli = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(cli)  # type: ignore[union-attr]
    assert cli._parse_since(None) is None
    assert cli._parse_since("2022-01-01") == datetime(2022, 1, 1, tzinfo=timezone.utc)
    assert cli._parse_since("2022-01-01T00:00:00-05:00") == datetime(
        2022, 1, 1, 5, tzinfo=timezone.utc
    )


def test_invalid_batch_arguments_raise(archive: Path) -> None:
    with pytest.raises(ValueError):
        score_archive(["AAA"], Fake(), batch_size=0)


def test_invalid_flush_threshold_raises(archive: Path) -> None:
    with pytest.raises(ValueError):
        score_archive(["AAA"], Fake(), flush_texts=0)


# ---- score_archive.py --verify --baseline-root --------------------------------------------------


def _cli():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("score_cli2", Path("scripts/score_archive.py"))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def fake_inference(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    import sys
    import types

    seen: list[str] = []
    mod = types.ModuleType("src.sentiment.inference")
    mod.score_batch = lambda texts, **kw: (seen.extend(texts), [min(len(t) / 100.0, 1.0) for t in texts])[1]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.sentiment.inference", mod)
    return seen


def test_verify_with_a_baseline_samples_only_scores_the_baseline_lacks(
    archive: Path, fake_inference: list[str]
) -> None:
    na.write_day("AAA", DAY, [art("have1", "a" * 10), art("have2", "b" * 20), art("new", "c" * 30)])
    na._save_score_cache("AAA", {"have1": 0.11, "have2": 0.21, "new": 0.31})
    base = archive / "baseline" / "processed" / "news_scores"
    base.mkdir(parents=True)
    (base / "AAA.json").write_text(json.dumps({"have1": 0.11, "have2": 0.21}))
    assert _cli()._verify(["AAA"], 10, 0, archive / "baseline") == 0
    assert fake_inference == ["c" * 30 + "."]  # only the URL the baseline lacks was recomputed


def test_verify_fails_when_a_new_cloud_score_is_wrong(
    archive: Path, fake_inference: list[str]
) -> None:
    na.write_day("AAA", DAY, [art("new", "c" * 30)])
    na._save_score_cache("AAA", {"new": 0.9})  # the cloud claims 0.9; recomputation gives 0.31
    base = archive / "baseline"
    assert _cli()._verify(["AAA"], 10, 0, base) == 1


def test_verify_with_a_baseline_and_nothing_new_passes_without_recomputing(
    archive: Path, fake_inference: list[str]
) -> None:
    na.write_day("AAA", DAY, [art("have", "a" * 10)])
    na._save_score_cache("AAA", {"have": 0.11})
    base = archive / "baseline" / "processed" / "news_scores"
    base.mkdir(parents=True)
    (base / "AAA.json").write_text(json.dumps({"have": 0.11}))
    assert _cli()._verify(["AAA"], 10, 0, archive / "baseline") == 0 and fake_inference == []


def test_a_corrupt_baseline_cache_raises(archive: Path, fake_inference: list[str]) -> None:
    na.write_day("AAA", DAY, [art("u", "a")])
    na._save_score_cache("AAA", {"u": 0.1})
    base = archive / "baseline" / "processed" / "news_scores"
    base.mkdir(parents=True)
    (base / "AAA.json").write_text("{not json")
    with pytest.raises(ValueError, match="corrupt score cache"):
        _cli()._verify(["AAA"], 10, 0, archive / "baseline")
