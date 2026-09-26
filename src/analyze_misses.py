"""Why does blocking miss true pairs?  (diagnostic, train split)

For a sample of Source 1 records of one country, retrieve the top-k S2/S3 candidates and look at
every true pair that was NOT retrieved:
  zero   - the pair shares no usable key at all (score 0): needs a new signal
  crowd  - it shares keys but k other records scored higher: more k / better weighting helps
Prints the split and example pairs of each kind.
Usage:
  python -m src.analyze_misses --data-dir dataset --country India --cache-dir E:\\ml_cache
"""
import argparse

import numpy as np
import pandas as pd

from .blocking import build_matrices, max_df_dict, retrieve
from .data_loader import index_of, load_ground_truth, load_split
from .eval_blocking import safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--country", required=True)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-df", type=int, default=5000)
    ap.add_argument("--max-df-pair", type=int, default=5000)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--n-show", type=int, default=25)
    args = ap.parse_args()

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
    rows = np.sort(np.random.default_rng(1).choice(len(s1), args.sample, replace=False))
    found, kth = set(), {}
    for src, name in ((0, "s2"), (1, "s3")):
        a, b, c = retrieve(mats["s1"], mats[name], args.k, 250, name.upper(), rows)
        found.update(zip(a.tolist(), [src] * a.size, b.tolist()))
        d = pd.DataFrame({"a": a, "c": c}).groupby("a")["c"].agg(["min", "size"])
        for r, (mn, sz) in d.iterrows():
            kth[(r, src)] = mn if sz >= args.k else 0.0

    g1 = index_of(s1["entity_id"], gt["source1_entity_id"])
    keep = np.isin(g1, rows)
    g = gt[keep].copy()
    g["i"] = g1[keep]
    g["src"] = np.where(g["rid"].str.startswith("S2"), 0, 1)
    g["j"] = np.where(g["src"] == 0, index_of(s2["entity_id"], g["rid"]),
                      index_of(s3["entity_id"], g["rid"]))
    g = g[g["j"] >= 0]
    g["hit"] = [(i, s, j) in found for i, s, j in zip(g["i"], g["src"], g["j"])]
    miss = g[~g["hit"]].copy()
    print(f"\n{args.country}: sampled S1={rows.size:,}  true pairs={len(g):,}  "
          f"retrieved={g['hit'].mean():.4f}  missed={len(miss):,}")

    score = np.zeros(len(miss), dtype=np.float32)
    for src, name in ((0, "s2"), (1, "s3")):
        m = (miss["src"] == 0 if src == 0 else miss["src"] == 1).to_numpy()
        if m.any():
            q = mats["s1"][miss["i"].to_numpy()[m]]
            r = mats[name][miss["j"].to_numpy()[m]]
            score[m] = np.asarray(q.multiply(r).sum(axis=1)).ravel()
    miss["score"] = score
    miss["kth"] = [kth.get((i, s), 0.0) for i, s in zip(miss["i"], miss["src"])]
    miss["kind"] = np.where(miss["score"] <= 0, "zero", "crowd")
    print(miss["kind"].value_counts(normalize=True).round(3).to_string())
    print("share of missed pairs from S2 / S3:",
          miss["src"].value_counts(normalize=True).round(3).to_dict())
    crowd = miss[miss["kind"] == "crowd"]
    if len(crowd):
        print("crowded: median own score", round(float(crowd["score"].median()), 3),
              "vs median k-th score", round(float(crowd["kth"].median()), 3))

    rec = {0: s2, 1: s3}
    for kind in ("zero", "crowd"):
        sub = miss[miss["kind"] == kind]
        print(f"\n=== {kind.upper()} examples ===")
        for _, x in sub.sample(min(args.n_show, len(sub)), random_state=0).iterrows():
            a, b = s1.iloc[x["i"]], rec[x["src"]].iloc[x["j"]]
            print(f"S1 | {a.business_name} | {a.business_address}")
            print(f"{b.entity_id[:2]} | {b.business_name} | {b.business_address}"
                  f"   (score {x['score']:.3f}, k-th {x['kth']:.3f})\n")


if __name__ == "__main__":
    main()
