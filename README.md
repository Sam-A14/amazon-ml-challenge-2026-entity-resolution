# Business Entity Resolution - Amazon ML Challenge 2026

For every Source 1 business record, find all Source 2 / Source 3 records that describe the same
real-world business. Output: `output/matching_results.tsv` (scored, macro F0.5) and
`output/candidate_pairs.tsv` (the exact candidate set the model scores).

## Approach in one paragraph
Records are processed **per country** (open set - France is handled although it is absent from
training). **Blocking**: each record is represented by IDF-weighted hashed keys - name words,
order-free name word pairs, address words and adjacent address word pairs - and sparse cosine
similarity retrieves the top-k Source 2 and top-k Source 3 records per Source 1 record; a reverse
filter keeps each S2/S3 record only for its top-m Source 1 records (in training no S2/S3 record
belongs to more than one S1 entity). **Matching**: LightGBM (MIT licence, no pretrained models)
scores every candidate pair using 26 string-similarity features (rapidfuzz) and 13 competition
features (ranks and score gaps). **Decision**: probability >= threshold tuned on validation macro
F0.5, and each S2/S3 record is assigned to at most one Source 1 entity. Only the provided data is
used; no external data, APIs or lookups.

## Project structure
~~~
src/
  normalize.py      text normalisation (accents, digit/letter splits, leading zeros, <NULL>)
  blocking.py       hashed keys, IDF weighting, sparse top-k retrieval, reverse filter, cache
  features.py       pairwise string + context features
  evaluation.py     official macro F0.5, threshold + one-owner decision rule
  data_loader.py    TSV loading (tab separated, strings only)
  eval_blocking.py  blocking recall / candidate size / F0.5 ceiling on train
  train.py          candidates -> features -> LightGBM -> threshold tuning -> models/
  predict.py        test candidates -> predictions -> both output TSVs
  submission.py     exact output format writer
eda/                exploratory analysis scripts
experiments/        blocking_results.txt, experiment_log.csv
requirements.txt    pinned versions
~~~

## Setup (Windows PowerShell, Python 3.13)
~~~
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~
Place the challenge data as:
~~~
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
utils/validate_submission.py
~~~

## Reproduce end to end
~~~
python -m src.eval_blocking --data-dir dataset --countries India --sample 100000 --cache-dir cache
python -m src.train --data-dir dataset --k 8 --m 2 --cache-dir cache
python -m src.predict --data-dir dataset --output output --cache-dir cache
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
~~~
`--cache-dir` stores tokenised blocking keys (a few GB) so repeated runs skip tokenisation.
Key parameters: `--k` candidates per source per S1 record, `--m` reverse filter,
`--max-df` / `--max-df-pair` document-frequency caps for word / word-pair keys (default 5000).
Random seeds are fixed (split seed 42 + country index, LightGBM seed 42).

## Hardware / runtime
16 GB RAM, 8-core laptop, CPU only. Full training about 1.5-2 h, test prediction about 1.5-2 h.
Run one heavy script at a time.

## Results
Validation metrics (held-out Source 1 entities, official macro F0.5) and blocking metrics for
every run are logged in `experiments/experiment_log.csv` and `experiments/blocking_results.txt`.
