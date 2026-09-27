"""Where does the MATCHING MODEL lose points?  (uses saved features + trained model, minutes)

On the validation pairs (true pairs that blocking DID find):
  FN-low    true pair, probability below threshold              -> model not convinced
  FN-owner  true pair above threshold but its S2/S3 record went to another S1 (one-owner rule)
  FP        wrong pair accepted
Prints shares by country / source / script and real examples of each kind.
Usage:
  python -m src.analyze_model --data-dir dataset --cache-dir E:\\ml_cache
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data_loader import load_split
from .evaluation import decide
from .train import KEY_R, KEY_S1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--n-show", type=int, default=20)
    args = ap.parse_args()

    z = np.load(Path(args.cache_dir) / "features" / "train_features.npz", allow_pickle=True)
    Xva, yva, kva, rva = z["Xva"], z["yva"].astype(bool), z["kva"], z["rva"]
    countries = [str(c) for c in z["countries"]]
    cfg = json.loads((Path(args.model_dir) / "config.json").read_text())
    booster = lgb.Booster(model_file=str(Path(args.model_dir) / "lgbm.txt"))
    p = booster.predict(Xva)
    ci = kva // KEY_S1
    thr = np.array([cfg.get("thresholds", {}).get(countries[c], cfg["threshold"]) for c in ci])
    acc = decide(kva, rva, p, thr)

    kind = np.full(len(p), "", dtype=object)
    kind[yva & ~acc & (p < thr)] = "FN-low"
    kind[yva & ~acc & (p >= thr)] = "FN-owner"
    kind[~yva & acc] = "FP"
    print(f"validation pairs {len(p):,}: true {yva.sum():,}  accepted {acc.sum():,}")
    for k in ("FN-low", "FN-owner", "FP"):
        print(f"  {k:9s} {np.sum(kind == k):,}")

    s1a, s2a, s3a = load_split(args.data_dir, "train")
    rows = []
    for c_idx, country in enumerate(countries):
        m = (ci == c_idx) & (kind != "")
        if not m.any():
            continue
        s1 = s1a[s1a["country"] == country].reset_index(drop=True)
        rec = pd.concat([s2a[s2a["country"] == country], s3a[s3a["country"] == country]],
                        ignore_index=True)
        i = (kva[m] - c_idx * KEY_S1).astype(np.int64)
        j = (rva[m] - c_idx * KEY_R).astype(np.int64)
        d = pd.DataFrame({
            "kind": kind[m], "p": p[m], "country": country,
            "s1_name": s1["business_name"].to_numpy()[i], "s1_addr": s1["business_address"].to_numpy()[i],
            "r_id": rec["entity_id"].to_numpy()[j], "r_name": rec["business_name"].to_numpy()[j],
            "r_addr": rec["business_address"].to_numpy()[j]})
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["src"] = d["r_id"].str[:2]
    d["indic_name"] = d["r_name"].map(lambda s: any(0x0900 <= ord(ch) < 0x0D80 for ch in str(s)))
    d["r_addr_empty"] = d["r_addr"].str.strip().isin(["", "<NULL>", "null"])
    print("\nshare of each error kind by country / source / Indian-script name / empty address:")
    for col in ("country", "src", "indic_name", "r_addr_empty"):
        print(pd.crosstab(d["kind"], d[col], normalize="index").round(3).to_string(), "\n")
    print("probability quantiles of FN-low:",
          d.loc[d["kind"] == "FN-low", "p"].quantile([.1, .5, .9]).round(3).tolist())

    for k in ("FN-low", "FN-owner", "FP"):
        sub = d[d["kind"] == k]
        print(f"\n=== {k} examples ===")
        for _, x in sub.sample(min(args.n_show, len(sub)), random_state=0).iterrows():
            print(f"S1 | {x.s1_name} | {x.s1_addr}")
            print(f"{x.src} | {x.r_name} | {x.r_addr}   (p={x.p:.3f}, {x.country})\n")


if __name__ == "__main__":
    main()
