# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Bravo  
**Team Members:** Samriddhi Srivastava (team lead), Ayushi Tiwari  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We solve entity resolution as **blocking → pairwise classification → constrained assignment**, run
separately per country so it generalises to a country unseen in training (France). Blocking combines
IDF-weighted hashed word / word-pair keys, a rule-based **Indian-script transliteration + phonetic
skeleton**, and **bidirectional search** (each Source 1 record retrieves its top candidates *and* each
Source 2/3 record retrieves its own best Source 1 records), giving ~19 candidates per Source 1 record
with a 0.9875 macro-F0.5 ceiling. A LightGBM classifier (MIT licence; no pretrained models, no external
data) scores pairs with string, competition, **house-number-relationship** and name-frequency features;
matches are accepted above a threshold tuned on the official macro F0.5 and each Source 2/3 record is
assigned to at most one Source 1 entity. Every change was driven by measured error analysis:
public leaderboard 0.813 → 0.925 → 0.953 → 0.964 → 0.967 → **0.968** (validation macro F0.5 0.9741).

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the full data (train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3, 7,638,365 true pairs;
test: 1,732,544 S1 / 4,887,273 S2 / 5,082,316 S3) established:
- **Scale:** ~10M S2+S3 test records, so all-pairs comparison is impossible.
- **Countries:** 100% of true pairs share the country label → process per country (open set;
  France = 15% of test S1, absent from training; nothing hard-coded).
- **One owner:** no S2/S3 record belongs to more than one S1 entity (0 exceptions in 7.6M pairs);
  ~26% of S2/S3 records match nothing.
- **Match counts:** 5.6% singletons; most S1 entities have 3–5 matches (≈3.5 on average).
- **Names:** exact match in only 10.7% of true pairs; shared word in 85.6% (India 76%). Noise:
  shuffled words, typos, accents, legal-suffix changes, website/handle names (`@piedmonthorizon`,
  `kerrwilliams.com`), initials, **names written phonetically in Indian scripts** (सुपर इम्पेक्स
  प्राइवेट लिमिटेड = Super Impex Private Limited), and unrelated names where only the address links.
- **Addresses:** shared word in 95.6%; ~4% empty; reordered components, abbreviations, native-script
  state names, `<NULL>` placeholders, and **house numbers that are truncated (1717→171, 169→69) or off
  by one (202→203)** in true pairs. Postal codes are rare (India 0.5%).
- **Integrity checks:** no train/test overlap (0 shared IDs, 0 identical name+address), and no
  information in row order or ID numbers (correlations ≈ 0.001); scripts in `eda/`.

### 2.2 Solution Strategy
**Approach Type:** Blocking + gradient-boosted classifier + constrained assignment (hybrid)  
**Core Innovation:** error-analysis-driven design: (1) bidirectional blocking that exploits the
one-owner property, (2) dependency-free transliteration of all Indian scripts via the shared Unicode
block layout plus a phonetic skeleton, (3) house-number relationship features that encode the
dataset's rule separating "same business, noisy number" from "different business, nearby number".

---

## 3. Candidate Generation (Blocking)
- **Normalisation:** lowercase; Indian scripts → Latin (one offset table for Devanagari, Bengali,
  Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam); accent stripping; letter/digit splits
  (`7704D` → `7704 d`); leading-zero removal; `<NULL>` dropped. No hand-written dictionaries.
- **Keys (hashed, 2^22 buckets per kind):** name words + squashed name (handles/domains); order-free
  name word pairs; address words; adjacent address word pairs within each comma component; phonetic
  skeleton words/pairs (`super impex private limited` → `spr mpks prvt lmt`). Skeleton and prefix
  keys are only added to Source 1 (queries) and to Indian-script S2/S3 names, so ordinary candidates
  keep an undiluted representation.
- **Weighting:** IDF per key kind within the country (keys seen once or > 5,000 times dropped), L2 norm.
- **Retrieval:** sparse cosine in row chunks. Forward: top-10 S2 and top-10 S3 per S1 (kept if the
  pair is among the S2/S3 record's top-3 S1). Reverse: every S2/S3 record's top-3 S1 records. The union is
  the final candidate set, exactly what the model scores (`candidate_pairs.tsv`).
- **Candidate pairs generated (test):** 32,633,478 (France 4.61M, India 15.46M, US 12.57M), about 18.8
  per S1 record. The narrower v10 setting (top-8 / top-2) gives 21,536,072 pairs (12.4 per S1) for a
  0.001 lower leaderboard score (0.967 vs 0.968), a candidate-size vs accuracy trade-off we report openly.
- **How we ensured true matches were not lost:** every change was measured on training data:

| Blocking version | India recall | US recall | Cands / S1 | Macro-F0.5 ceiling |
|---|---|---|---|---|
| v1 single words, df cap 1,000 | 0.656 | 0.678 | ~6 | 0.815 |
| v3 + word pairs, df cap 5,000 | 0.887 | 0.923 | ~7 | 0.963 |
| v6 + transliteration/skeleton + reverse search (k=8, top-2) | 0.948 | 0.966 | ~10 | 0.985 |
| **v12: wider search (k=10, reverse top-3)** | **0.955** | **0.972** | ~15.5 | **0.9875** |

  Diagnostics that drove v6: 81% of v3's missed India pairs were "crowded out" of the forward top-8
  (fixed by reverse search) and 19% shared no key, almost all Indian-script names (fixed by
  transliteration). Sampled India study: forward 0.923 → forward + reverse top-2 0.960 pair recall.

---

## 4. Matching Model
**Features (57):**
- *Name:* rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler; word Jaccard and
  containment; squashed-name equality, containment and ratio; phonetic-skeleton ratio, token-set and
  Jaccard; word counts; non-Latin flag.
- *Address:* ratio / token-set / partial; word Jaccard; share of S2/S3 words found in S1; empty flags;
  word counts; share of S1 name words inside the S2/S3 address.
- *House numbers (v9):* for each S2/S3 number, its relation to the S1 numbers (exact, truncated
  prefix/suffix, off-by-one, or unrelated) as counts and fractions, plus distances.
- *Name frequency (v10):* how many S1 / S2+S3 records share each normalised name (decides records with
  empty addresses).
- *Competition (from blocking):* retrieval score; rank among the S1 record's candidates; rank among the
  S2/S3 record's S1 candidates; best/second-best scores, gaps and candidate counts.

**Model type:** LightGBM binary classifier (MIT), 127 leaves, learning rate 0.05, feature and bagging
fraction 0.8; best iteration selected on validation macro F0.5.  
**Training data:** candidates of 300k S1 entities per training country (9.3M pairs with the final blocking); a disjoint set of
50k S1 entities per country is held out for validation (entity-level split).  
**Threshold selection:** grid search maximising the official macro F0.5 on validation (singletons and
blocking misses included) after the one-owner rule (each S2/S3 record goes to its most probable S1 only).

---

## 5. Results & Error Analysis

| Version | Main change | Val macro F0.5 | Precision | Recall | Leaderboard |
|---|---|---|---|---|---|
| v1 | baseline blocking + LightGBM | 0.7932 | 0.9832 | 0.6470 | 0.813 |
| v3 | word-pair keys, df cap 5,000 | 0.9363 | 0.9845 | 0.8722 | 0.925 |
| v6 | transliteration + reverse search | 0.9648 | 0.9894 | 0.9261 | 0.953 |
| v7 | 2× training data (not submitted) | 0.9608 | 0.9879 | 0.9193 | — |
| v9 | house-number relationship features | 0.9707 | 0.9929 | 0.9342 | 0.964 |
| v10 | + name-frequency features | 0.9727 | 0.9940 | 0.9371 | 0.967 |
| v11 | second-stage (stacked) model on v10 features (not submitted) | 0.9729 | — | — | — |
| **v12 (final)** | **wider blocking (k=10, reverse top-3) + v10 model** | **0.9741** | **0.9942** | **0.9398** | **0.968** |

- **F_0.5 Score (macro):** 0.9741 validation / 0.968 public leaderboard (v12, final submission).
- **Decision rule check:** a per-entity expected-F0.5 decision rule was also evaluated on validation
  and automatically rejected because the tuned global threshold (0.725) scored higher.
- **Model errors on validation (v7 → v9 → v10):** rejected true pairs 13,113 → 7,938 → 6,926;
  wrong merges accepted 3,889 → 2,315 → 1,963. The stacked second stage (v11) added only +0.0002
  and was not submitted.
- **Common false positives (wrong merges):** different businesses on the same street with nearby but
  unrelated house numbers (3004 vs 3013, 84D vs 88D), largely fixed by v9 (−40% wrong merges).
  Remaining: look-alike names with truncated numbers, and identical names without an address that
  belong to a namesake; random generated names sharing an address with a different business.
- **Common false negatives (missed matches):** records with an empty S2/S3 address (35% of v9's model
  misses in v9, 37% in v10: many are genuinely ambiguous namesakes), unrelated generated names ("Halogild") with truncated numbers, heavy typos,
  plus ~1.5% of pairs never retrieved by blocking.

---

## 6. Conclusion
A scalable, CPU-only, dictionary-free pipeline (bidirectional IDF-weighted word, word-pair and phonetic
blocking; a LightGBM matcher over string, competition, house-number and name-frequency features; and a
one-owner assignment with an F0.5-tuned threshold) resolves 24M noisy records across three sources and
an unseen country. The key lesson: each large gain came from inspecting real errors and fixing that
exact pattern (crowding → reverse search; Indian scripts → transliteration; house numbers → relation
features), while generic changes such as simply adding training data did not help.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `normalize.py`, `translit.py` (text and script
normalisation), `blocking.py` (keys, IDF, forward/reverse retrieval, cache), `features.py` (pair
features), `train.py` (candidates → features → LightGBM; saves features), `predict.py` (test
candidates and features), `augment.py` (house-number and name-frequency features, final model, output
files), `evaluation.py` (macro F0.5, one-owner decision), `submission.py` (output writer), and
diagnostics `eval_blocking.py`, `eval_reverse.py`, `analyze_misses.py`, `analyze_model.py`,
`stage2.py` (stacking experiment). Entry points and exact commands are in `README.md`; pinned versions
in `requirements.txt`. Hardware: 16 GB RAM laptop, CPU only.

### B. Additional Results
Blocking grids are in `experiments/blocking_results.txt`; per-run parameters and validation metrics are
in `experiments/experiment_log.csv`.
