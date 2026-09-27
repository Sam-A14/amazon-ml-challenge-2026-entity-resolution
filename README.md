# Business Entity Resolution - Amazon ML Challenge 2026 (Team Bravo)

For every Source 1 business record, find all Source 2 / Source 3 records describing the same
real-world business. Outputs: `output/matching_results.tsv` (scored, macro F0.5) and
`output/candidate_pairs.tsv` (the exact candidate set the final model scores).

## Approach
Per country (open set, France handled although unseen in training):
1. **Blocking:** IDF-weighted hashed keys (name words, order-free name word pairs, address words,
   address word pairs, phonetic skeletons of names incl. Indian-script names transliterated to Latin),
   sparse cosine retrieval in both directions: top-8 S2/S3 per S1, and top-2 S1 per S2/S3 record.
2. **Matching:** LightGBM (MIT licence) on string, competition, house-number-relationship and
   name-frequency features. No pretrained models, no external data or APIs.
3. **Decision:** threshold tuned on validation macro F0.5; each S2/S3 record assigned to at most one
   S1 entity (no S2/S3 record belongs to two S1 entities in the training data).

## Setup (Windows PowerShell, Python 3.13; CPU only, 16 GB RAM is enough)
~~~
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~
Data layout:
~~~
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
utils/validate_submission.py
~~~

## Reproduce the final submission (3 commands + validation)
`--cache-dir` stores tokenised blocking keys and saved features (about 10 GB); use a drive with space.
~~~
# 1. blocking + features on train, first LightGBM, saves train/val features   (~2.5 h)
python -m src.train --data-dir dataset --k 8 --m 2 --rev-m 2 --n-train 300000 --cache-dir cache

# 2. blocking + features on test, saves test candidates/features               (~2.5 h)
python -m src.predict --data-dir dataset --output output --cache-dir cache

# 3. final model: adds house-number (+ name-frequency) features to the saved candidates,
#    trains the final LightGBM, tunes the threshold, writes both output files  (~1 h)
python -m src.augment --data-dir dataset --cache-dir cache --output output --baseline 0

# 4. official format check
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
~~~
Step 3 as written reproduces the submitted **v10** model (house-number + name-frequency features).
Add `--no-name-freq --model-dir models/v9` to reproduce v9 instead.

## Project structure
~~~
src/normalize.py, translit.py   text + Indian-script normalisation, phonetic skeleton
src/blocking.py                 keys, IDF, forward/reverse retrieval, candidate context features, cache
src/features.py                 pairwise string / competition features
src/train.py                    candidates -> features -> LightGBM -> threshold; saves features
src/predict.py                  test candidates -> features (saved) -> predictions
src/augment.py                  house-number + name-frequency features, final model, output files
src/evaluation.py               official macro F0.5, one-owner decision rule
src/submission.py               exact output format writer
src/eval_blocking.py, eval_reverse.py, analyze_misses.py, analyze_model.py   diagnostics
src/stage2.py                   stacking experiment (not used in the final submission)
eda/                            EDA and data-integrity checks
experiments/                    blocking_results.txt, experiment_log.csv
~~~

## Results (public leaderboard, macro F0.5)
v1 0.813 → v3 0.925 → v6 0.953 → v9 0.964 → **v10 0.967** (validation 0.9727, final submission). Details in `Documentation_template.md`.
Random seeds are fixed (split seed 42 + country index, LightGBM seed 42).
