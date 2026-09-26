"""Measure blocking quality on the TRAIN split.

Retrieves top-K S2 and S3 candidates for a random sample of Source 1 records (or all of
them with --sample 0), then evaluates a grid of k (per source) and reverse-m filters:
  pair_recall   share of true pairs kept as candidates
  cands/S1      average candidate-list size per Source 1 entity (smaller ranks higher)
  ceiling_F05   macro F0.5 of a perfect matcher restricted to these candidates
NOTE: with a sample, the reverse filter (m) only sees sampled competitors, so its rows are
optimistic; the m=None rows are exact.
Usage:
  python -m src.eval_blocking --data-dir dataset --countries India --sample 100000
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np

from .blocking import filter_candidates, max_df_dict, rank_within, retrieve_all
from .data_loader import countries_of, index_of, load_ground_truth, load_split


def ceiling_f05(n_s1, gold_s1, hit_s1, rows):
    """Macro F0.5 over `rows` if the matcher picked exactly the true candidate pairs."""
    g = np.bincount(gold_s1, minlength=n_s1)[rows]
    h = np.bincount(hit_s1, minlength=n_s1)[rows]
    r = np.divide(h, g, out=np.zeros(rows.size), where=g > 0)
    f = np.where(g == 0, 1.0, np.where(h == 0, 0.0, 1.25 * r / (0.25 + r)))
    return f.mean()


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=100_000, help="S1 queries per country, 0 = all")
    ap.add_argument("--max-k", type=int, default=20)
    ap.add_argument("--max-df", type=int, default=5000, help="cap for single-word keys")
    ap.add_argument("--max-df-pair", type=int, default=5000, help="cap for word-pair keys")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="experiments/blocking_results.txt")
    args = ap.parse_args()

    t0 = time.time()
    s1_all, s2_all, s3_all = load_split(args.data_dir, "train")
    gt = load_ground_truth(args.data_dir)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    countries = args.countries or countries_of(s1_all, s2_all, s3_all)
    # keep only the requested countries in memory
    s1_all, s2_all, s3_all = (f[f["country"].isin(countries)] for f in (s1_all, s2_all, s3_all))
    gt = gt[gt["source1_entity_id"].isin(s1_all["entity_id"])]
    max_df = max_df_dict(args.max_df, args.max_df_pair)

    lines = [f"\n##### max_df={args.max_df} max_df_pair={args.max_df_pair} sample={args.sample}"]
    for country in countries:
        tc = time.time()
        s1 = s1_all[s1_all["country"] == country].reset_index(drop=True)
        s2 = s2_all[s2_all["country"] == country].reset_index(drop=True)
        s3 = s3_all[s3_all["country"] == country].reset_index(drop=True)
        print(f"\n== {country}: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}", flush=True)
        rows = np.arange(len(s1))
        if 0 < args.sample < len(s1):
            rows = np.sort(np.random.default_rng(0).choice(len(s1), args.sample, replace=False))

        s1i, rid, score, src = retrieve_all(s1, s2, s3, args.max_k, max_df, args.chunk,
                                            f"{args.cache_dir}/train_{safe(country)}", rows)
        fwd = rank_within(s1i * 2 + src, score)

        g = gt[gt["source1_entity_id"].isin(s1["entity_id"])]
        g1 = index_of(s1["entity_id"], g["source1_entity_id"])
        rid_ids = np.concatenate([s2["entity_id"].to_numpy(), s3["entity_id"].to_numpy()])
        g2 = index_of(rid_ids, g["rid"])
        ok = (g1 >= 0) & (g2 >= 0)
        g1, g2 = g1[ok], g2[ok]
        n_r = len(rid_ids)
        in_rows = np.isin(g1, rows)
        gold_keys = g1[in_rows].astype(np.int64) * n_r + g2[in_rows]

        lines.append(f"\n== {country}  queries={rows.size:,}  true_pairs={in_rows.sum():,}  "
                     f"time {time.time() - tc:.0f}s")
        lines.append(f"{'k':>3} {'m':>4} {'pair_recall':>12} {'cands/S1':>9} {'ceiling_F05':>12}")
        print(lines[-2] + "\n" + lines[-1], flush=True)
        for k in [3, 5, 8, 10, 15, 20]:
            if k > args.max_k:
                continue
            for m in [None, 1, 2, 3]:
                fs, fr, _ = filter_candidates(s1i, rid, score, fwd, k, m)
                hit = np.isin(fs.astype(np.int64) * n_r + fr, gold_keys)
                line = (f"{k:>3} {str(m):>4} {hit.sum() / max(gold_keys.size, 1):>12.4f} "
                        f"{fs.size / rows.size:>9.2f} "
                        f"{ceiling_f05(len(s1), g1, fs[hit], rows):>12.4f}")
                lines.append(line)
                print(line, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nappended to {args.out}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
