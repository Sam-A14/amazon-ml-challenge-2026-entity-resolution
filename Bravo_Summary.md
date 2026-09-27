# Amazon ML Challenge 2026 — Business Entity Resolution
**Team Bravo** · Samriddhi Srivastava (team lead), Ayushi Tiwari · 27 September 2026  
**Final result:** public leaderboard macro F0.5 **0.968** (validation 0.9741)

## 1. ML approach
Entity resolution as **blocking → pairwise classification → constrained assignment**, run per
country (open set; France, unseen in training, handled identically). Only the provided data is used —
no external data, APIs or pretrained models; everything runs on a 16 GB CPU laptop.

**Key findings from EDA (24M records)** that shaped the design: matches never cross countries; no
Source 2/3 record belongs to more than one Source 1 entity; names are very noisy (typos, shuffled words,
website handles, names written phonetically in Indian scripts, unrelated generated names) while
addresses are more stable; true pairs often have truncated or off-by-one house numbers.

**Blocking (candidate generation).** Records become IDF-weighted hashed keys: name words, order-free
name word pairs, address words, adjacent address word pairs, and phonetic skeletons of names after a
dependency-free transliteration of all Indian scripts (e.g. सुपर इम्पेक्स प्राइवेट लिमिटेड →
"spr mpks prvt lmt" = Super Impex Private Limited). Sparse cosine retrieval runs in **both directions**:
each Source 1 record takes its top-10 Source 2 and Source 3 records, and each Source 2/3 record takes
its top-3 Source 1 records. Result: ~19 candidates per Source 1 record, recall 0.955 (India) / 0.972 (US),
macro-F0.5 ceiling 0.9875.

## 2. ML model and features
**LightGBM** binary classifier (MIT licence), 127 leaves, learning rate 0.05, best iteration chosen on
validation macro F0.5; trained on 9.3M candidate pairs (300k Source 1 entities per training country).
**57 features:** name similarity (rapidfuzz ratios, Jaro-Winkler, token sets, squashed-name / handle
checks, phonetic-skeleton similarity); address similarity; **house-number relationships** (exact,
truncated, off-by-one or unrelated — separates "same business, noisy number" from "neighbour at a nearby
number"); **name frequency** (how many records share a name — decides records with no address); and
competition features (a pair's rank among its Source 1 record's and its Source 2/3 record's candidates).
**Decision:** probability ≥ 0.725 (tuned on the official macro F0.5, singletons included), then each
Source 2/3 record is assigned to at most one Source 1 entity.

## 3. Experiments (validation macro F0.5 / public leaderboard)
| Version | Change (each driven by measured error analysis) | Val | LB |
|---|---|---|---|
| v1 | word keys + LightGBM baseline | 0.793 | 0.813 |
| v3 | word-pair keys; fewer common-word cut-offs (blocking recall 0.66 → 0.89) | 0.936 | 0.925 |
| v6 | reverse search (fixes 81% of misses: "crowded out") + Indian-script transliteration | 0.965 | 0.953 |
| v9 | house-number relationship features (−40% wrong merges) | 0.971 | 0.964 |
| v10 | name-frequency features | 0.973 | 0.967 |
| **v12** | **wider search (top-10, reverse top-3)** | **0.974** | **0.968** |

Tried and rejected on validation: 2× training data (0.961), stacked second-stage model (+0.0002), a
per-entity expected-F0.5 decision rule (threshold was better). Integrity checks confirmed no train/test
overlap and no information in row order or IDs.

## 4. Conclusion
A scalable, reproducible, CPU-only pipeline raised the score from 0.813 to 0.968. The biggest gains came
from **inspecting real errors and fixing the exact pattern** — crowding → reverse search, Indian scripts →
transliteration, house numbers → relation features — rather than generic tuning. Remaining errors are
mostly ambiguous (namesakes without addresses, generated names sharing an address); closing them would
likely need GPU-based neural matchers, the main limitation of our hardware. Full details:
`Documentation_template.md`; code and exact commands: `code/business_entity_resolution/`.
