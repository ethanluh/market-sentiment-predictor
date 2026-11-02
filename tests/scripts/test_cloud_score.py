"""Runs the real scripts/cloud_score.sh against a fake gcloud and a fake python (no cloud, no torch)."""

import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cloud_score.sh"
PROJECT = "msp-scoring-test"

FAKE_GCLOUD = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
args="$*"
rc=0
case "$args" in
  "compute instances create"*)
    zone=""
    for a in "$@"; do case "$a" in --zone=*) zone="${a#--zone=}" ;; esac; done
    case " ${FAKE_EXHAUSTED_ZONES:-} " in
      *" $zone "*)
        printf 'ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n---\ncode: ZONE_RESOURCE_POOL_EXHAUSTED\nmessage: The zone projects/p/zones/%s does not have\n  enough resources available to fulfill the request.\n' "$zone" >&2
        exit 1 ;;
    esac
    touch "$FAKE_STATE/vm"
    for a in "$@"; do
      case "$a" in --metadata-from-file=startup-script=*) cp "${a#*=startup-script=}" "$FAKE_STATE/startup.sh" ;; esac
    done
    [ -n "${FAKE_CREATE_FAIL:-}" ] && exit 1
    if [ -n "${FAKE_FAILED:-}" ]; then touch "$FAKE_STATE/failed"
    elif [ -z "${FAKE_NEVER_DONE:-}" ]; then touch "$FAKE_STATE/done"; fi ;;
  "compute instances delete"*)
    [ -z "${FAKE_DELETE_SLEEP:-}" ] || sleep "$FAKE_DELETE_SLEEP"
    [ -n "${FAKE_LEAK_VM:-}" ] || rm -f "$FAKE_STATE/vm" ;;
  "compute instances list"*)
    if [ -n "${FAKE_LIST_FAIL:-}" ]; then echo "permission denied" >&2; rc=1
    elif [ -f "$FAKE_STATE/vm" ]; then echo vm
    else echo "WARNING: The following filter keys were not present in any resource : name" >&2; fi ;;
  "compute instances describe"*)
    n=$(( $(cat "$FAKE_STATE/describes" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKE_STATE/describes"
    [ -z "${FAKE_DESCRIBE_FAILS:-}" ] || [ "$n" -gt "$FAKE_DESCRIBE_FAILS" ] || exit 1
    [ -z "${FAKE_DONE_AFTER:-}" ] || [ "$n" -lt "$FAKE_DONE_AFTER" ] || touch "$FAKE_STATE/done"
    [ -f "$FAKE_STATE/vm" ] && echo "${FAKE_VM_STATUS:-RUNNING}" || echo GONE ;;
  "storage buckets create"*) touch "$FAKE_STATE/bucket" ;;
  "storage cp /"*)
    mkdir -p "$FAKE_STATE/uploads"
    for a in "$@"; do case "$a" in /*.tgz) cp "$a" "$FAKE_STATE/uploads/" ;; esac; done ;;
  "storage buckets list"*)
    if [ -f "$FAKE_STATE/bucket" ]; then echo bucket
    else echo "WARNING: The following filter keys were not present in any resource : name" >&2; fi ;;
  "storage rm"*) rm -f "$FAKE_STATE/bucket" ;;
  "storage ls"*"/status/DONE") [ -f "$FAKE_STATE/done" ] || rc=1 ;;
  "storage ls"*"/status/FAILED") [ -f "$FAKE_STATE/failed" ] || rc=1 ;;
  "storage cp gs://"*)
    for a in "$@"; do
      if [ "$prev" = "storage_cp_src" ]; then dest="$a"; break; fi
      case "$a" in gs://*) prev=storage_cp_src; src="$a" ;; esac
    done
    case "$src" in
      *results.tgz) cp "$FAKE_RESULTS" "$dest" ;;
      *) echo "log" > "$dest" ;;
    esac ;;
esac
exit $rc
"""

FAKE_PY = r"""#!/usr/bin/env bash
echo "PY root=${MSP_DATA_ROOT:-} $*" >> "$FAKE_LOG"
case "$*" in
  *"import torch"*) exit "${FAKE_IMPORT_RC:-0}" ;;
  *"--verify"*) exit "${FAKE_VERIFY_RC:-0}" ;;
esac
exit 0
"""


def _exe(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def env(tmp_path: Path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "gcloud", FAKE_GCLOUD)
    fake_py = _exe(tmp_path / "fakepy", FAKE_PY)
    state = tmp_path / "state"
    state.mkdir()

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "score_archive.py").write_text("# stub\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    data = tmp_path / "data"
    (data / "raw" / "news" / "AAA").mkdir(parents=True)
    (data / "raw" / "news" / "AAA" / "2024-01-05.json").write_text("[]")
    (data / "processed" / "news_scores").mkdir(parents=True)
    (data / "processed" / "news_scores" / "AAA.json").write_text("{}")

    results_src = tmp_path / "res" / "processed" / "news_scores"
    results_src.mkdir(parents=True)
    (results_src / "AAA.json").write_text('{"u": 0.5}')
    results = tmp_path / "results.tgz"
    with tarfile.open(results, "w:gz") as tf:
        tf.add(tmp_path / "res" / "processed", arcname="processed")

    log = tmp_path / "calls.log"
    log.write_text("")
    e = dict(os.environ)
    e.update(
        PATH=f"{bin_dir}:{e['PATH']}",
        FAKE_LOG=str(log),
        FAKE_STATE=str(state),
        FAKE_RESULTS=str(results),
        MSP_POLL_SECONDS="0",
        T_REPO=str(repo),
        T_DATA=str(data),
        T_PY=str(fake_py),
        T_LOG=str(log),
        T_STATE=str(state),
    )
    return e


def run(env: dict, *extra: str, project: str = PROJECT) -> subprocess.CompletedProcess:
    cmd = [
        "bash",
        str(SCRIPT),
        "--project",
        project,
        "--repo",
        env["T_REPO"],
        "--data-dir",
        env["T_DATA"],
        "--python",
        env["T_PY"],
        *extra,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)


def calls(env: dict) -> list[str]:
    return Path(env["T_LOG"]).read_text().splitlines()


def index(lines: list[str], needle: str) -> int:
    return next(i for i, line in enumerate(lines) if needle in line)


def test_dry_run_creates_nothing_and_reports_the_plan(env: dict) -> None:
    p = run(env)
    assert p.returncode == 0, p.stderr
    log = "\n".join(calls(env))
    for forbidden in (
        "instances create",
        "instances delete",
        "buckets create",
        "storage rm",
        "storage cp",
    ):
        assert forbidden not in log
    assert "dry run" in p.stdout and "estimated cost" in p.stdout and "hard cap 3 h" in p.stdout
    assert "[dry-run] gcloud compute instances create" in p.stdout


@pytest.mark.parametrize("project", ["edgewild", "msp", "prod-msp-scoring-1"])
def test_only_msp_scoring_projects_are_accepted(env: dict, project: str) -> None:
    p = run(env, project=project)
    assert p.returncode != 0 and "only msp-scoring-* projects" in p.stderr


def test_a_project_is_required(env: dict) -> None:
    env.pop("MSP_GCP_PROJECT", None)
    p = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "no project" in p.stderr


def test_the_vm_is_capped_and_self_deleting_with_minimal_scopes(env: dict) -> None:
    run(env, "--run")
    create = next(c for c in calls(env) if c.startswith("compute instances create"))
    for flag in (
        "--max-run-duration=3h",
        "--instance-termination-action=DELETE",
        "--scopes=storage-rw",
        "--accelerator=type=nvidia-tesla-t4,count=1",
        "--maintenance-policy=TERMINATE",
        "--image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580",
        f"--project={PROJECT}",
    ):
        assert flag in create
    assert "SPOT" not in create


def test_max_hours_sets_the_vm_lifetime_cap(env: dict) -> None:
    run(env, "--run", "--max-hours", "1")
    create = next(c for c in calls(env) if c.startswith("compute instances create"))
    assert "--max-run-duration=1h" in create and "--max-run-duration=3h" not in create


def test_spot_is_opt_in(env: dict) -> None:
    run(env, "--run", "--spot")
    assert "--provisioning-model=SPOT" in next(
        c for c in calls(env) if c.startswith("compute instances create")
    )


def test_happy_path_verifies_before_merging_and_cleans_up(env: dict) -> None:
    p = run(env, "--run")
    assert p.returncode == 0, p.stderr
    lines = calls(env)
    order = [
        index(lines, n)
        for n in (
            "storage buckets create",
            "storage cp",
            "compute instances create",
            "--verify",
            "--merge-from",
            "compute instances delete",
            "storage rm -r",
        )
    ]
    assert order == sorted(order) and len(set(order)) == len(order)
    verify, merge = index(lines, "--verify"), index(lines, "--merge-from")
    assert verify < merge
    assert merge < index(lines, "compute instances delete") < index(lines, "storage rm -r")
    assert "no billable resources remain" in p.stdout
    assert (
        not (Path(env["T_STATE"]) / "vm").exists()
        and not (Path(env["T_STATE"]) / "bucket").exists()
    )


def test_a_failed_verification_merges_nothing_but_still_cleans_up(env: dict) -> None:
    env["FAKE_VERIFY_RC"] = "1"
    p = run(env, "--run")
    assert p.returncode != 0 and "verification failed; nothing was merged" in p.stderr
    log = "\n".join(calls(env))
    assert "--merge-from" not in log
    assert "compute instances delete" in log and "storage rm" in log
    assert "kept the downloaded results" in p.stderr


def test_a_failed_vm_job_stops_before_verify_and_cleans_up(env: dict) -> None:
    env["FAKE_FAILED"] = "1"
    p = run(env, "--run")
    log = "\n".join(calls(env))
    assert p.returncode != 0 and "the VM job failed" in p.stderr
    assert "--verify" not in log and "--merge-from" not in log
    assert "compute instances delete" in log and "storage rm" in log


def test_a_vm_that_vanishes_is_reported_not_waited_for(env: dict) -> None:
    env["FAKE_NEVER_DONE"] = "1"
    env["FAKE_VM_STATUS"] = "TERMINATED"
    p = run(env, "--run")
    assert p.returncode != 0 and "the VM is TERMINATED" in p.stderr
    assert "--merge-from" not in "\n".join(calls(env))


def test_a_leftover_vm_makes_the_run_fail_loudly(env: dict) -> None:
    env["FAKE_LEAK_VM"] = "1"
    p = run(env, "--run")
    assert p.returncode != 0 and "still exists" in p.stderr


def test_a_missing_local_torch_stops_a_real_run_before_any_spending(env: dict) -> None:
    env["FAKE_IMPORT_RC"] = "1"
    p = run(env, "--run")
    assert p.returncode != 0 and "cannot import torch" in p.stderr
    assert "instances create" not in "\n".join(calls(env)) and "buckets create" not in "\n".join(
        calls(env)
    )


def test_a_secret_in_the_code_archive_blocks_the_upload(env: dict) -> None:
    repo = Path(env["T_REPO"])
    (repo / ".env").write_text("ALPACA_PAPER_SECRET_KEY=hunter2\n")
    _git(repo, "add", "-f", ".env")
    _git(repo, "commit", "-q", "-m", "oops")
    p = run(env, "--run")
    assert p.returncode != 0 and "secret-looking file" in p.stderr
    assert "buckets create" not in "\n".join(calls(env))


def test_tickers_limit_the_upload_and_are_passed_through(env: dict) -> None:
    p = run(env, "--run", "--tickers", "AAA")
    assert p.returncode == 0, p.stderr
    create = next(c for c in calls(env) if c.startswith("compute instances create"))
    assert "tickers=AAA" in create
    assert "--tickers AAA" in next(c for c in calls(env) if "--verify" in c)
    q = run(env, "--tickers", "ZZZ")
    assert q.returncode != 0 and "no archive for ticker ZZZ" in q.stderr


def test_no_inputs_other_than_news_scores_and_code_are_uploaded(env: dict) -> None:
    run(env, "--run")
    uploads = next(c for c in calls(env) if c.startswith("storage cp") and "code.tgz" in c)
    assert (
        uploads.count(".tgz") == 3
        and "code.tgz" in uploads
        and "news.tgz" in uploads
        and "scores.tgz" in uploads
    )
    with tarfile.open(Path(env["T_STATE"]) / "uploads" / "code.tgz") as tf:
        assert not [n for n in tf.getnames() if n.endswith(".env") or ".env." in n]


def test_invalid_numbers_are_rejected(env: dict) -> None:
    assert run(env, "--max-hours", "0").returncode != 0
    assert run(env, "--max-hours", "abc").returncode != 0
    assert run(env, "--verify-n", "0").returncode != 0


def test_a_blank_env_example_template_is_allowed(env: dict) -> None:
    repo = Path(env["T_REPO"])
    (repo / ".env.example").write_text(
        "ALPACA_PAPER_KEY_ID=\nALPACA_PAPER_SECRET_KEY=\nLOG_LEVEL=INFO\n"
    )
    _git(repo, "add", ".env.example")
    _git(repo, "commit", "-q", "-m", "template")
    assert run(env).returncode == 0


def test_a_filled_in_env_example_blocks_the_upload(env: dict) -> None:
    repo = Path(env["T_REPO"])
    (repo / ".env.example").write_text("ALPACA_PAPER_SECRET_KEY=abcdef123456\n")
    _git(repo, "add", ".env.example")
    _git(repo, "commit", "-q", "-m", "template with a secret")
    p = run(env)
    assert p.returncode != 0 and "non-empty secret-looking value" in p.stderr


# ---- verify and merge use the right data roots --------------------------------------------------


def test_verify_checks_the_downloaded_scores_and_merge_targets_the_local_tree(env: dict) -> None:
    assert run(env, "--run").returncode == 0
    lines = calls(env)
    verify = next(c for c in lines if "--verify" in c)
    merge = next(c for c in lines if "--merge-from" in c)
    # Verify reads the downloaded scores (a temp tree), samples only scores new relative to the
    # local tree, and never points at the local data directory itself.
    assert f"--baseline-root {env['T_DATA']}" in verify and f"root={env['T_DATA']} " not in verify
    assert "/v " in verify.split("--verify")[0] + " "
    # Merge writes into the local tree and reads the downloaded caches.
    assert f"root={env['T_DATA']} " in merge and "/v/processed/news_scores" in merge


def test_a_relative_data_dir_still_verifies_correctly(env: dict, tmp_path: Path) -> None:
    cmd = [
        "bash",
        str(SCRIPT),
        "--project",
        PROJECT,
        "--repo",
        env["T_REPO"],
        "--data-dir",
        "data",
        "--python",
        env["T_PY"],
        "--run",
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    verify = next(c for c in calls(env) if "--verify" in c)
    assert f"--baseline-root {env['T_DATA']}" in verify  # absolute, not the dangling "data"


# ---- cleanup under failure and signals ----------------------------------------------------------


def test_a_create_that_fails_after_making_the_vm_still_deletes_it(env: dict) -> None:
    env["FAKE_CREATE_FAIL"] = "1"
    p = run(env, "--run")
    assert p.returncode != 0
    assert (
        not (Path(env["T_STATE"]) / "vm").exists()
        and not (Path(env["T_STATE"]) / "bucket").exists()
    )


def _start(env: dict, *extra: str) -> subprocess.Popen:
    cmd = [
        "bash",
        str(SCRIPT),
        "--project",
        PROJECT,
        "--repo",
        env["T_REPO"],
        "--data-dir",
        env["T_DATA"],
        "--python",
        env["T_PY"],
        "--run",
        *extra,
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait_for(env: dict, needle: str, timeout: float = 15.0) -> None:
    import time

    end = time.time() + timeout
    while time.time() < end:
        if any(needle in line for line in calls(env)):
            return
        time.sleep(0.05)
    raise AssertionError(f"never saw {needle!r} in the gcloud log")


def test_sigterm_exits_nonzero_and_cleans_up(env: dict) -> None:
    import signal

    env["FAKE_NEVER_DONE"] = "1"
    env["MSP_POLL_SECONDS"] = "1"
    proc = _start(env)
    _wait_for(env, "compute instances describe")
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode == 143
    assert (
        not (Path(env["T_STATE"]) / "vm").exists()
        and not (Path(env["T_STATE"]) / "bucket").exists()
    )


def test_a_second_signal_during_cleanup_does_not_abandon_it(env: dict) -> None:
    import signal

    env["FAKE_NEVER_DONE"] = "1"
    env["MSP_POLL_SECONDS"] = "1"
    env["FAKE_DELETE_SLEEP"] = "2"
    proc = _start(env)
    _wait_for(env, "compute instances describe")
    proc.send_signal(signal.SIGTERM)
    _wait_for(env, "compute instances delete")
    proc.send_signal(signal.SIGTERM)  # impatient second Ctrl-C while the VM delete is in flight
    proc.communicate(timeout=30)
    assert not (Path(env["T_STATE"]) / "vm").exists()
    assert not (Path(env["T_STATE"]) / "bucket").exists()


def test_transient_status_failures_are_tolerated(env: dict) -> None:
    env["FAKE_NEVER_DONE"] = "1"
    env["FAKE_DESCRIBE_FAILS"] = "2"
    env["FAKE_DONE_AFTER"] = "4"
    p = run(env, "--run")
    assert p.returncode == 0, p.stderr
    assert "--merge-from" in "\n".join(calls(env))


def test_persistent_status_failures_stop_the_run(env: dict) -> None:
    env["FAKE_NEVER_DONE"] = "1"
    env["FAKE_DESCRIBE_FAILS"] = "999"
    p = run(env, "--run")
    assert p.returncode != 0 and "five times in a row" in p.stderr
    assert not (Path(env["T_STATE"]) / "vm").exists()


def test_the_startup_script_is_valid_shell_and_stops_the_gpu(env: dict) -> None:
    run(env, "--run")
    startup = Path(env["T_STATE"]) / "startup.sh"
    assert subprocess.run(["bash", "-n", str(startup)]).returncode == 0
    text = startup.read_text()
    assert "shutdown -h now" in text and "MSP_DATA_ROOT=/opt/job/data" in text
    assert "curl -sf" in text and "ALPACA" not in text and "SECRET" not in text


# ---- upload guards ------------------------------------------------------------------------------


def _commit(repo: Path, rel: str, content: str) -> None:
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)
    _git(repo, "add", "-f", rel)
    _git(repo, "commit", "-q", "-m", f"add {rel}")


@pytest.mark.parametrize(
    "name",
    [
        ".env.production",
        "config/prod.env",
        "id_rsa",
        "service_account.json",
        ".netrc",
        ".envrc",
        "certs/server.p12",
        "secrets.json",
        ".npmrc",
        "sa-key.json",
    ],
)
def test_secret_looking_file_names_block_the_upload(env: dict, name: str) -> None:
    _commit(Path(env["T_REPO"]), name, "x")
    p = run(env)
    assert p.returncode != 0 and "secret-looking file" in p.stderr


@pytest.mark.parametrize(
    "line",
    [
        "API_KEY_2=abc",
        "api_key=abc",
        "export X_SECRET_KEY=abc",
        "NEWSAPI_KEY = abc",
        "PASSWD=abc",
        "TOKEN=abc",
    ],
)
def test_filled_in_templates_block_the_upload_in_any_spelling(env: dict, line: str) -> None:
    _commit(Path(env["T_REPO"]), ".env.example", line + "\n")
    p = run(env)
    assert p.returncode != 0 and "non-empty secret-looking value" in p.stderr


@pytest.mark.parametrize(
    "content",
    [
        "-----BEGIN RSA PRIVATE KEY-----\nabc\n",
        "k = 'AKIAABCDEFGHIJKLMNOP'\n",
        "id = 'PKABCDEFGHIJ0123456789'\n",
    ],
)
def test_credential_shaped_text_in_any_tracked_file_blocks_the_upload(
    env: dict, content: str
) -> None:
    _commit(Path(env["T_REPO"]), "src/notes.py", content)
    p = run(env)
    assert p.returncode != 0 and "shaped like a credential" in p.stderr


def test_a_foreign_file_in_the_news_archive_blocks_the_upload(env: dict) -> None:
    (Path(env["T_DATA"]) / "raw" / "news" / "AAA" / "notes.txt").write_text("hi")
    p = run(env)
    assert p.returncode != 0 and "holds something other than" in p.stderr


def test_uncommitted_changes_to_scoring_code_are_refused(env: dict) -> None:
    (Path(env["T_REPO"]) / "scripts" / "score_archive.py").write_text("# edited, not committed\n")
    p = run(env)
    assert p.returncode != 0 and "uncommitted changes" in p.stderr


def test_a_multiline_tickers_value_becomes_one_clean_line(env: dict) -> None:
    p = run(env, "--run", "--tickers", "AAA\n   AAA")
    create = next(c for c in calls(env) if c.startswith("compute instances create"))
    assert p.returncode == 0 and "tickers=AAA AAA," in create or "tickers=AAA AAA " in create


@pytest.mark.parametrize(
    "content",
    [
        "id = 'PKABCDEFGHIJ0123456789ABCD'\n",
        "id = 'AKABCDEFGHIJ0123456789ABCD'\n",
    ],  # 26-character broker key ids
)
def test_longer_broker_key_ids_are_caught_too(env: dict, content: str) -> None:
    _commit(Path(env["T_REPO"]), "src/notes.py", content)
    p = run(env)
    assert p.returncode != 0 and "shaped like a credential" in p.stderr


# ---- zone fallback and authoritative cleanup ----------------------------------------------------


def create_zones(env: dict) -> list[str]:
    return [
        c.split("--zone=")[1].split()[0]
        for c in calls(env)
        if c.startswith("compute instances create")
    ]


def test_a_stocked_out_zone_falls_through_to_the_next_one(env: dict) -> None:
    env["FAKE_EXHAUSTED_ZONES"] = "us-central1-a us-central1-b"
    p = run(env, "--run", "--zones", "us-central1-a us-central1-b us-central1-c")
    assert p.returncode == 0, p.stderr
    assert create_zones(env) == ["us-central1-a", "us-central1-b", "us-central1-c"]
    assert "created in us-central1-c" in p.stdout
    assert "--zone=us-central1-c" in next(
        c for c in calls(env) if c.startswith("compute instances delete")
    )
    assert "--merge-from" in "\n".join(calls(env))


def test_every_zone_exhausted_fails_cleanly_without_a_false_leak_report(env: dict) -> None:
    env["FAKE_EXHAUSTED_ZONES"] = "us-central1-a us-central1-b"
    p = run(env, "--run", "--zones", "us-central1-a us-central1-b")
    assert p.returncode != 0 and "no T4 capacity in any of" in p.stderr and "--spot" in p.stderr
    assert "still exists" not in p.stderr and "no billable resources remain" in p.stdout
    assert not (Path(env["T_STATE"]) / "bucket").exists() and "--verify" not in "\n".join(
        calls(env)
    )


def test_a_non_capacity_create_failure_is_not_retried(env: dict) -> None:
    env["FAKE_CREATE_FAIL"] = "1"
    p = run(env, "--run", "--zones", "us-central1-a us-central1-b us-central1-c")
    assert p.returncode != 0 and "other than capacity" in p.stderr
    assert create_zones(env) == ["us-central1-a"]
    assert not (Path(env["T_STATE"]) / "vm").exists()


def test_a_single_zone_is_tried_once_and_only_once(env: dict) -> None:
    env["FAKE_EXHAUSTED_ZONES"] = "us-central1-a"
    p = run(env, "--run", "--zone", "us-central1-a")
    assert p.returncode != 0 and create_zones(env) == ["us-central1-a"]


def test_a_failing_listing_is_reported_not_taken_as_all_clear(env: dict) -> None:
    env["FAKE_LIST_FAIL"] = "1"
    p = run(env, "--run")
    assert p.returncode != 0 and "could not list VMs" in p.stderr
    assert "no billable resources remain" not in p.stdout


def test_the_dry_run_plans_the_first_zone_and_mentions_the_fallback(env: dict) -> None:
    p = run(env)
    assert p.returncode == 0 and p.stdout.count("[dry-run] gcloud compute instances create") == 1
    assert "--zone=us-central1-a" in p.stdout and "would try the other zones" in p.stdout
    assert "zones us-central1-a us-central1-b" in p.stdout


def test_gcloud_warnings_on_an_empty_listing_are_not_a_leak(env: dict) -> None:
    """Real gcloud warns on stderr when a filtered list is empty; that must not read as a leftover."""
    p = run(env, "--run")
    assert p.returncode == 0, p.stderr
    assert "still exists" not in p.stderr and "no billable resources remain" in p.stdout
