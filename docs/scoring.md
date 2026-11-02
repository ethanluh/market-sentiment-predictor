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

## Running it on a cloud GPU

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
