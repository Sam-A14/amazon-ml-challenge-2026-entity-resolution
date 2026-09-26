"""Measure reverse search on the TRAIN split (compare with forward-only blocking).

Samples true pairs through their S2/S3 record, then checks whether the pair is found by
  forward  : the S1 record's top-k S2 / S3 candidates
  reverse  : the S2/S3 record's top-m S1 candidates (m = 1, 2, 3)
  union    : either
Reverse candidates per S1 record are estimated as m * (#S2+#S3) / #S1.
Usage:
  python -m src.eval_reverse --data-dir dataset --country India --cache-dir E:\\ml_cache
"""
import argparse
import time

import numpy as np

from .blocking import build_matrices, max_df_dict, retrieve
from .data_loader import index_of, load_ground_truth, load_split
from .eval_blocking import safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--country", required=True)
    ap.add_argument("--sample", type=int, default=40000, help="true pairs sampled per source")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-df", type=int, default=5000)
    ap.add_argument("--max-df-pair", type=int, default=5000)
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()
    t0 = time.time()

    s1a, s2a, s3a = load_split(args.data_dir, "train")
    s1 = s1a[s1a["country"] == args.country].reset_index(drop=True)
    s2 = s2a[s2a["country"] == args.country].reset_index(drop=True)
    s3 = s3a[s3a["country"] == args.country].reset_index(drop=True)
    del s1a, s2a, s3a
    gt = load_ground_truth(args.data_dir)
    gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])]
    mats = build_matrices({"s1": s1, "s2": s2, "s3": s3},
                          max_df_dict(args.max_df, args.max_df_pair),
                          f"{args.cache_dir}/train_{safe(args.country)}")
    rng = np.random.default_rng(3)
    n_r = len(s2) + len(s3)
    print(f"\n== {args.country}: S1={len(s1):,} S2+S3={n_r:,}  "
          f"reverse candidates per S1 ~ {n_r / len(s1):.2f} x m")

    for name, df in (("s2", s2), ("s3", s3)):
        g = gt[gt["rid"].str.startswith(name.upper())]
        g = g.iloc[rng.choice(len(g), min(args.sample, len(g)), replace=False)]
        i = index_of(s1["entity_id"], g["source1_entity_id"])
        j = index_of(df["entity_id"], g["rid"])
        ok = (i >= 0) & (j >= 0)
        i, j = i[ok], j[ok]
        # forward: top-k of the owners
        owners = np.unique(i)
        a, b, _ = retrieve(mats["s1"], mats[name], args.k, 250, name.upper(), owners)
        fwd = set(zip(a.tolist(), b.tolist()))
        f_hit = np.array([(x, y) in fwd for x, y in zip(i, j)])
        # reverse: top-3 S1 of each sampled S2/S3 record
        recs = np.unique(j)
        ra, rb, rc = retrieve(mats[name], mats["s1"], 3, 1000, name.upper() + "-rev", recs)
        order = np.lexsort((-rc, ra))
        ra, rb = ra[order], rb[order]
        first = np.searchsorted(ra, ra, side="left")
        rank = np.arange(ra.size) - first
        rev = {(int(r), int(s)): int(k) for r, s, k in zip(ra, rb, rank)}
        rrank = np.array([rev.get((y, x), 99) for x, y in zip(i, j)])
        print(f"\n{name.upper()}: {i.size:,} sampled true pairs")
        print(f"  forward top-{args.k}      recall {f_hit.mean():.4f}")
        for m in (1, 2, 3):
            r_hit = rrank < m
            print(f"  reverse top-{m}         recall {r_hit.mean():.4f}   "
                  f"union with forward {(f_hit | r_hit).mean():.4f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
