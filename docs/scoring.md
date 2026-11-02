# Scoring the news archive

`scripts/score_archive.py` runs FinBERT over every archived article once and caches the scores in
`data/processed/news_scores/<TICKER>.json` (keyed by article URL), so backtests and evaluations
normally never run the model. (`make_archive_reader` still scores any article missing from the
cache, slowly, one lookback window at a time.) The scorer is resumable: every cache file is
replaced atomically (temp file, then rename) every 2,000 scores and once more on any exit, so an
interrupt keeps what was scored; a flush across tickers is not atomic as a whole, but each cache is
always self-consistent. A rerun scores only what is missing, and a URL carried by several tickers
is scored once. Corrupt or out-of-range cache entries raise instead of being trusted.

```bash
python scripts/score_archive.py                          # every ticker in the archive
python scripts/score_archive.py --tickers AAPL --since 2022-01-01
python scripts/score_archive.py --verify 50              # recompute 50 cached scores and compare
```

Scores use the full 512-token input in fp32 on any device, so a GPU and a CPU agree (`--verify`
fails above 0.005). Truncating is not a safe shortcut: on a 256-article sample, 128 tokens
correlated only 0.68 with the 512-token scores (sign agreement 83%), and 64 tokens or headlines
alone about 0.53.

## How long it takes

Measured on a 16-core CPU with no GPU: about 3.6 articles/s for articles with a full body (2,800
characters on average) and about 31/s for headline-only items. The archive built on 2026-11-02
holds 159,730 article-ticker pairs, 112,483 unique URLs, and 44k articles with empty bodies, which
is roughly 6 hours on that CPU. A T4-class GPU should take around half an hour (an estimate from
published BERT-base throughput, not measured here).

## One command on Google Cloud

`scripts/cloud_score.sh` does the whole cloud round trip and cleans up after itself. It is a dry
run unless you pass `--run`:

```bash
scripts/cloud_score.sh --project msp-scoring-XXXX            # prints the plan and estimated cost
scripts/cloud_score.sh --project msp-scoring-XXXX --run      # create, score, verify, merge, delete
```

What it does with `--run`: tars the repo (`git archive HEAD`), `data/raw/news` and the existing
score caches; creates a private bucket and uploads them; starts one T4 VM from a PyTorch GPU image;
the VM pulls the files from the bucket, runs `score_archive.py`, and pushes `results.tgz` back. Back
here it downloads the results, recomputes up to `--verify-n` (default 50) of the scores the cloud
**added** (not the ones you already had) on this machine before merging anything, merges, then
deletes the VM and the bucket and confirms nothing remains.

Safety properties, all covered by tests that run the real script against a fake `gcloud`:

- Only projects whose id starts with `msp-scoring-` are accepted.
- Only the code, the news archive and the score caches are uploaded. The code archive is checked
  for secret-looking file names (`.env*`, `*.env`, keys, `id_rsa`, `.netrc`, service-account JSON,
  and so on), for private-key headers and well-known key shapes in any tracked file, and a blank
  `.env.example` is allowed only if every key/secret/token/password entry is empty. The news
  tarball may hold only `raw/news/<TICKER>/<day>.json`. `scripts/` and `src/` must be committed,
  because the VM runs `HEAD` while verification runs your working tree.
- No SSH, no open ports, no keys on the VM.
- The VM has `--max-run-duration` (default 3 h) with termination action DELETE, so it removes itself
  even if the script is killed, and it shuts itself down as soon as it finishes so the GPU stops
  billing. The script marks the VM and bucket for cleanup before creating them, ignores further
  Ctrl-C/TERM while cleaning up, exits non-zero on a signal (130/143), tolerates a few transient
  status errors, deletes the VM and bucket on any exit, and fails loudly if either still exists.
- A failed verification merges nothing and keeps the downloaded scores for inspection.
- Cost is about $0.54/hour on demand (approximate list prices: T4 plus `n1-standard-4`), so the 3 h
  cap is at most about $1.62. `--spot` is cheaper, but a preemption loses the run.

Needs: `gcloud` logged in, GPU quota in the project (T4 and the global GPU limit), and a local Python
with `torch` and `transformers` for the verification step (`--python`).

## Running it on a cloud GPU by hand

Only the news archive goes to the cloud: no API keys, no `.env`, no broker credentials. Scoring
needs nothing but the articles and the public FinBERT weights.

1. Zip the archive: `cd data && zip -r news.zip raw/news` (about 500 MB).
2. Start a GPU machine with PyTorch. The cheapest options are a free Colab or Kaggle T4 notebook,
   or a spot T4 instance (AWS `g4dn.xlarge`, Azure `NC4as_T4_v3`; new accounts often have a zero
   GPU quota that must be raised first, which can take a day). A few hours at spot prices is
   under a dollar.
3. On the machine: clone the repo, `pip install -r requirements.txt`, unzip so the archive sits at
   `data/raw/news/`, then run `python scripts/score_archive.py`. Set `MSP_DATA_ROOT=/path/to/data`
   if the data lives elsewhere.
4. Copy the machine's `data/processed/news_scores/` to a scratch folder here (not over your own
   caches) and delete the machine. Then merge and check:

   ```bash
   python scripts/score_archive.py --merge-from /path/to/scratch/news_scores
   python scripts/score_archive.py --verify 100    # recompute 100 cached scores on this machine
   ```

`--merge-from` adds scores you do not have, never drops local ones, and raises if an entry present
in both differs by more than 0.005 or any score is outside [-1, 1]. A local rerun then fills
whatever is still missing.
