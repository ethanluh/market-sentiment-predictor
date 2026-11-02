#!/usr/bin/env bash
# Score the news archive on a one-off cloud GPU, verify the scores locally, merge them, and delete
# every cloud resource it created. Dry run by default: it prints the plan and an estimated cost and
# creates nothing.
#
#   scripts/cloud_score.sh --project msp-scoring-XXXX          # dry run
#   scripts/cloud_score.sh --project msp-scoring-XXXX --run    # create, score, verify, merge, delete
#
# Safety:
#   - Only a project whose id starts with "msp-scoring-" is accepted.
#   - Only data/raw/news, the existing score caches and `git archive HEAD` go to the cloud. The
#     tarballs are checked for .env files, and no API keys or broker credentials are ever uploaded.
#   - No SSH and no open ports: the VM pulls its inputs from a private bucket and pushes results
#     back. The VM deletes itself after --max-hours (termination action DELETE) even if this script
#     is killed, and this script deletes the VM and the bucket on any exit.
#   - Scores are verified on this machine before they are merged. A failed check merges nothing.
set -euo pipefail

PROJECT="${MSP_GCP_PROJECT:-}"
ZONE="us-central1-a"
MAX_HOURS=3
VERIFY_N=50
PY="python3"
TICKERS=""
SPOT=0
RUN=0
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=""
POLL_SECONDS="${MSP_POLL_SECONDS:-30}"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
MACHINE_TYPE="n1-standard-4"
# Approximate us-central1 list prices per hour (T4 plus n1-standard-4); confirm on the pricing page.
RATE_ON_DEMAND="0.54"
RATE_SPOT="0.17"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[cloud_score] $*"; }

usage() {
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --project ID     GCP project (default: $MSP_GCP_PROJECT); must start with msp-scoring-
  --zone Z         default us-central1-a
  --max-hours N    hard cap on VM lifetime, default 3
  --tickers "A B"  score only these tickers (default: everything in the archive)
  --verify-n N     new cloud scores recomputed locally before merging, default 50
  --python PATH    local python with torch and transformers (needed for --verify), default python3
  --data-dir DIR   data tree, default <repo>/data
  --spot           use a spot VM (cheaper, but a preemption loses the run)
  --run            actually create resources (default is a dry run)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --project) PROJECT="${2:?}"; shift 2 ;;
    --zone) ZONE="${2:?}"; shift 2 ;;
    --max-hours) MAX_HOURS="${2:?}"; shift 2 ;;
    --tickers) TICKERS="${2:?}"; shift 2 ;;
    --verify-n) VERIFY_N="${2:?}"; shift 2 ;;
    --python) PY="${2:?}"; shift 2 ;;
    --data-dir) DATA_DIR="${2:?}"; shift 2 ;;
    --repo) REPO="${2:?}"; shift 2 ;;
    --spot) SPOT=1; shift ;;
    --run) RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

[ -n "$PROJECT" ] || die "no project: pass --project or set MSP_GCP_PROJECT"
[[ "$PROJECT" =~ ^msp-scoring- ]] || die "refusing project '$PROJECT': only msp-scoring-* projects are allowed"
[[ "$MAX_HOURS" =~ ^[1-9][0-9]*$ ]] || die "--max-hours must be a positive integer"
[[ "$VERIFY_N" =~ ^[1-9][0-9]*$ ]] || die "--verify-n must be a positive integer"
DATA_DIR="${DATA_DIR:-$REPO/data}"
[ -d "$DATA_DIR" ] || die "data directory $DATA_DIR does not exist"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"  # absolute: it is symlinked from another directory later
REGION="${ZONE%-*}"
STAMP="$(date +%Y%m%d-%H%M%S)"
VM="msp-score-$STAMP"
BUCKET="$PROJECT-score-$(date +%s)"
TMP="$(mktemp -d)"
VM_CREATED=0
BUCKET_CREATED=0
KEEP_TMP=0

# Read-only and local commands run in both modes; commands that create or delete cloud resources go
# through here, so a dry run cannot touch the cloud.
run() {
  if [ "$RUN" = 1 ]; then "$@"; else echo "[dry-run] $*"; fi
}

cleanup() {
  local rc=$?
  trap '' INT TERM HUP  # a second Ctrl-C must not abandon the deletions below
  local leftover=0
  if [ "$RUN" = 1 ]; then
    if [ "$VM_CREATED" = 1 ]; then
      log "deleting VM $VM"
      gcloud compute instances delete "$VM" --project="$PROJECT" --zone="$ZONE" --quiet || leftover=1
    fi
    if [ "$BUCKET_CREATED" = 1 ]; then
      log "deleting bucket gs://$BUCKET"
      gcloud storage rm -r "gs://$BUCKET" --project="$PROJECT" || leftover=1
    fi
    if [ "$VM_CREATED" = 1 ] || [ "$BUCKET_CREATED" = 1 ]; then
      if [ -n "$(gcloud compute instances list --project="$PROJECT" --filter="name=$VM" --format='value(name)' 2>/dev/null)" ]; then
        echo "error: VM $VM still exists" >&2; leftover=1
      fi
      if [ -n "$(gcloud storage buckets list --project="$PROJECT" --filter="name=$BUCKET" --format='value(name)' 2>/dev/null)" ]; then
        echo "error: bucket gs://$BUCKET still exists" >&2; leftover=1
      fi
      [ "$leftover" = 0 ] && log "no billable resources remain"
    fi
  fi
  if [ "$KEEP_TMP" = 1 ]; then
    echo "kept the downloaded results and logs in $TMP" >&2
  else
    rm -rf "$TMP"
  fi
  if [ "$leftover" = 1 ] && [ "$rc" = 0 ]; then rc=1; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

command -v gcloud >/dev/null || die "gcloud is not installed"
[ -d "$DATA_DIR/raw/news" ] || die "no news archive at $DATA_DIR/raw/news"
git -C "$REPO" rev-parse HEAD >/dev/null 2>&1 || die "$REPO is not a git repository"
# The VM runs the committed code; verification runs the working tree. They must be the same.
git -C "$REPO" diff --quiet HEAD -- scripts src || die "scripts/ or src/ has uncommitted changes; commit or stash them first"

# The local verification needs torch; check it now so we never pay for a run we cannot verify.
if ! "$PY" -c 'import torch, transformers' >/dev/null 2>&1; then
  [ "$RUN" = 1 ] && die "$PY cannot import torch and transformers, which --verify needs (see --python)"
  log "warning: $PY cannot import torch and transformers; a real run would refuse to start"
fi

# --- inputs -------------------------------------------------------------------------------------
read -r -a TICKER_ARR <<<"$(echo $TICKERS)"  # one line, single-spaced, same for metadata and flags
TICKERS="${TICKER_ARR[*]:-}"
if [ "${#TICKER_ARR[@]}" -gt 0 ]; then
  NEWS_PATHS=()
  for t in "${TICKER_ARR[@]}"; do
    [ -d "$DATA_DIR/raw/news/$t" ] || die "no archive for ticker $t"
    NEWS_PATHS+=("raw/news/$t")
  done
else
  NEWS_PATHS=("raw/news")
fi

git -C "$REPO" archive --format=tar.gz -o "$TMP/code.tgz" HEAD
tar -C "$DATA_DIR" -czf "$TMP/news.tgz" "${NEWS_PATHS[@]}"
HAVE_SCORES=0
if [ -d "$DATA_DIR/processed/news_scores" ]; then
  tar -C "$DATA_DIR" -czf "$TMP/scores.tgz" processed/news_scores
  HAVE_SCORES=1
fi

# Nothing sensitive may leave this machine.
# A blank .env.example template is fine, but only if every key/secret/token/password is empty.
CODE_FILES="$(tar -tzf "$TMP/code.tgz" | grep -Ev '(^|/)\.env\.(example|sample|template)$' || true)"
SECRET_NAMES='(^|/)(\.env[^/]*|[^/]*\.env|\.envrc|\.envs/.*|\.aws/.*|\.ssh/.*|\.gnupg/.*|id_(rsa|dsa|ecdsa|ed25519)|\.netrc|\.npmrc|\.pypirc|\.git-credentials|[^/]*credentials[^/]*|[^/]*service[-_]?account[^/]*\.json|[^/]*secrets?[^/]*\.(json|ya?ml|env|txt)|[^/]*[-_]keys?\.json|[^/]*\.(pem|key|p12|pfx|jks|keystore))$'
if grep -Eiq "$SECRET_NAMES" <<<"$CODE_FILES"; then
  die "the code archive contains a secret-looking file; refusing to upload"
fi
# Content check on everything tracked: private-key headers and well-known key shapes.
if tar -xzOf "$TMP/code.tgz" 2>/dev/null | grep -aEq -- '-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|ghp_[0-9A-Za-z]{36}|\b(PK|AK)[A-Z0-9]{18,32}\b'; then
  die "the code archive contains text shaped like a credential; refusing to upload"
fi
while IFS= read -r template; do
  [ -n "$template" ] || continue
  if tar -xzOf "$TMP/code.tgz" "$template" | grep -Eiq '^[[:space:]]*(export[[:space:]]+)?[A-Za-z0-9_]*(KEY|SECRET|TOKEN|PASSW(OR)?D)[A-Za-z0-9_]*[[:space:]]*=[[:space:]]*[^[:space:]#]'; then
    die "$template has a non-empty secret-looking value; refusing to upload"
  fi
done < <(tar -tzf "$TMP/code.tgz" | grep -E '(^|/)\.env\.(example|sample|template)$' || true)
if grep -Evq '^raw/news/?$|^raw/news/[^/]+/?$|^raw/news/[^/]+/[^/]+\.json$' < <(tar -tzf "$TMP/news.tgz"); then
  die "the news archive holds something other than raw/news/<TICKER>/<day>.json; refusing to upload"
fi

size() { du -h "$1" | cut -f1; }
COST_RATE="$RATE_ON_DEMAND"; [ "$SPOT" = 1 ] && COST_RATE="$RATE_SPOT"
MAX_COST="$(awk -v h="$MAX_HOURS" -v r="$COST_RATE" 'BEGIN{printf "%.2f", h*r}')"
log "mode: $([ "$RUN" = 1 ] && echo RUN || echo 'dry run (nothing will be created)')"
log "project $PROJECT, zone $ZONE, $MACHINE_TYPE + 1x nvidia-tesla-t4, $([ "$SPOT" = 1 ] && echo spot || echo on-demand)"
log "uploads: code $(size "$TMP/code.tgz") (git HEAD $(git -C "$REPO" rev-parse --short HEAD)), news $(size "$TMP/news.tgz"), existing scores $([ "$HAVE_SCORES" = 1 ] && size "$TMP/scores.tgz" || echo none)"
log "estimated cost: about \$$COST_RATE/hour; hard cap $MAX_HOURS h => at most about \$$MAX_COST (approximate list prices; a 112k-article run should take well under 1 h)"

# --- bucket and uploads -------------------------------------------------------------------------
[ "$RUN" = 1 ] && BUCKET_CREATED=1  # marked first: cleanup must cover a create that half-succeeds
run gcloud storage buckets create "gs://$BUCKET" --project="$PROJECT" --location="$REGION" \
  --uniform-bucket-level-access
UPLOADS=("$TMP/code.tgz" "$TMP/news.tgz")
[ "$HAVE_SCORES" = 1 ] && UPLOADS+=("$TMP/scores.tgz")
run gcloud storage cp "${UPLOADS[@]}" "gs://$BUCKET/" --project="$PROJECT"

# --- the VM's job -------------------------------------------------------------------------------
cat >"$TMP/startup.sh" <<'STARTUP'
#!/bin/bash
set -uo pipefail
md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET="$(md bucket)" || exit 1
TICKERS="$(md tickers || true)"
LOG=/var/log/score.log
exec > >(tee "$LOG") 2>&1
finish() {  # $1 = DONE or FAILED
  gcloud storage cp "$LOG" "gs://$BUCKET/score.log" || true
  echo "$1" | gcloud storage cp - "gs://$BUCKET/status/$1"
  shutdown -h now  # stop the GPU billing at once; the controlling script deletes the VM afterwards
}
for _ in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi >/dev/null 2>&1 || { echo "no GPU became available"; finish FAILED; exit 1; }
mkdir -p /opt/job/code /opt/job/data && cd /opt/job
gcloud storage cp "gs://$BUCKET/code.tgz" "gs://$BUCKET/news.tgz" . || { finish FAILED; exit 1; }
gcloud storage cp "gs://$BUCKET/scores.tgz" . 2>/dev/null || true
tar xzf code.tgz -C code && tar xzf news.tgz -C data
[ -f scores.tgz ] && tar xzf scores.tgz -C data
pip install -q "transformers>=4.40,<5" pandas numpy || { finish FAILED; exit 1; }
export MSP_DATA_ROOT=/opt/job/data
ARGS=()
[ -n "$TICKERS" ] && ARGS+=(--tickers $TICKERS)
if python3 code/scripts/score_archive.py "${ARGS[@]}"; then
  tar czf results.tgz -C data processed/news_scores \
    && gcloud storage cp results.tgz "gs://$BUCKET/results.tgz" && finish DONE && exit 0
fi
finish FAILED
exit 1
STARTUP

CREATE=(gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE"
  --machine-type="$MACHINE_TYPE" --accelerator=type=nvidia-tesla-t4,count=1
  --maintenance-policy=TERMINATE
  --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release
  --boot-disk-size=100GB --boot-disk-type=pd-balanced
  --scopes=storage-rw
  --metadata="install-nvidia-driver=True,bucket=$BUCKET,tickers=$TICKERS"
  --metadata-from-file="startup-script=$TMP/startup.sh"
  --instance-termination-action=DELETE --max-run-duration="${MAX_HOURS}h")
[ "$SPOT" = 1 ] && CREATE+=(--provisioning-model=SPOT)
[ "$RUN" = 1 ] && VM_CREATED=1  # marked first: an error or Ctrl-C mid-create must still delete it
run "${CREATE[@]}"

if [ "$RUN" != 1 ]; then
  log "dry run complete. After the VM finishes, this script would download results.tgz, run"
  log "  $PY scripts/score_archive.py --verify $VERIFY_N   (on the downloaded scores, before any merge)"
  log "  $PY scripts/score_archive.py --merge-from <downloaded scores>"
  log "and then delete the VM and the bucket. Re-run with --run to do it."
  exit 0
fi

# --- wait for the VM ----------------------------------------------------------------------------
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 + 900 ))
DESCRIBE_FAILS=0
log "waiting for the VM job (poll every ${POLL_SECONDS}s, hard stop in about $MAX_HOURS h)"
while :; do
  if gcloud storage ls "gs://$BUCKET/status/DONE" >/dev/null 2>&1; then break; fi
  if gcloud storage ls "gs://$BUCKET/status/FAILED" >/dev/null 2>&1; then
    gcloud storage cp "gs://$BUCKET/score.log" "$TMP/score.log" >/dev/null 2>&1 || true
    KEEP_TMP=1
    die "the VM job failed; see $TMP/score.log"
  fi
  [ "$(date +%s)" -lt "$DEADLINE" ] || die "timed out waiting for the VM job"
  if STATUS="$(gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" --format='value(status)' 2>/dev/null)"; then
    DESCRIBE_FAILS=0
    case "$STATUS" in
      PROVISIONING|STAGING|RUNNING) ;;
      TERMINATED|STOPPING|SUSPENDED|REPAIRING|"")
        # The VM may have just finished and shut itself down: look for the marker once more.
        gcloud storage ls "gs://$BUCKET/status/DONE" >/dev/null 2>&1 && continue
        die "the VM is ${STATUS:-GONE} and no result was uploaded (preempted or deleted)" ;;
      *) die "the VM is in unexpected state $STATUS" ;;
    esac
  else
    DESCRIBE_FAILS=$((DESCRIBE_FAILS + 1))
    [ "$DESCRIBE_FAILS" -lt 5 ] || die "could not read the VM status five times in a row"
  fi
  sleep "$POLL_SECONDS"
done
log "VM job finished"

# --- verify, then merge -------------------------------------------------------------------------
gcloud storage cp "gs://$BUCKET/results.tgz" "$TMP/results.tgz" --project="$PROJECT"
gcloud storage cp "gs://$BUCKET/score.log" "$TMP/score.log" --project="$PROJECT" >/dev/null 2>&1 || true
mkdir -p "$TMP/v"
tar -xzf "$TMP/results.tgz" -C "$TMP/v"
[ -d "$TMP/v/processed/news_scores" ] || { KEEP_TMP=1; die "results.tgz has no processed/news_scores"; }
ln -s "$DATA_DIR/raw" "$TMP/v/raw"

SCORE_ARGS=()
[ "${#TICKER_ARR[@]}" -gt 0 ] && SCORE_ARGS=(--tickers "${TICKER_ARR[@]}")
log "verifying up to $VERIFY_N scores the cloud ADDED against a local recomputation"
if ! MSP_DATA_ROOT="$TMP/v" "$PY" "$REPO/scripts/score_archive.py" --verify "$VERIFY_N" \
  --baseline-root "$DATA_DIR" ${SCORE_ARGS[@]+"${SCORE_ARGS[@]}"}; then
  KEEP_TMP=1
  die "verification failed; nothing was merged. The cloud scores are in $TMP/v/processed/news_scores"
fi
log "merging into the local score caches"
MSP_DATA_ROOT="$DATA_DIR" "$PY" "$REPO/scripts/score_archive.py" --merge-from "$TMP/v/processed/news_scores" ${SCORE_ARGS[@]+"${SCORE_ARGS[@]}"}
log "done"
