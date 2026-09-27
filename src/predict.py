"""Run the trained pipeline on the test set and write the two submission files.

Usage:
  python -m src.predict --data-dir dataset --split test --output output
Every country present in the test files is processed (open set, e.g. France), and every
Source 1 test entity gets exactly one row in both output files.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import generate_candidates, max_df_dict
from .eval_blocking import safe
from .data_loader import countries_of, load_split
from .evaluation import decide
from .features import build_features
from .submission import write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--output", default="output")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--feat-dir", default=None,
                    help="save test candidates+features here; with --reuse-features load them "
                         "(re-scoring with a new model then takes minutes)")
    ap.add_argument("--reuse-features", action="store_true")
    args = ap.parse_args()
    if args.feat_dir is None:                     # default: keep features next to the key cache
        args.feat_dir = str(Path(args.cache_dir) / "features")
    t0 = time.time()

    cfg = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(Path(args.model_dir) / "lgbm.txt"))
    s1_all, s2_all, s3_all = load_split(args.data_dir, args.split)
    print(f"config: k={cfg['k']} m={cfg['m']} max_df={cfg['max_df']}/{cfg.get('max_df_pair')} "
          f"threshold={cfg['threshold']:.3f}", flush=True)

    cand_s1, cand_r, match_s1, match_r = [], [], [], []
    for country in countries_of(s1_all, s2_all, s3_all):
        s1 = s1_all[s1_all["country"] == country].reset_index(drop=True)
        s2 = s2_all[s2_all["country"] == country].reset_index(drop=True)
        s3 = s3_all[s3_all["country"] == country].reset_index(drop=True)
        rec = pd.concat([s2, s3], ignore_index=True)
        print(f"\n== {country}: S1={len(s1):,} S2+S3={len(rec):,}", flush=True)
        if len(s1) == 0 or len(rec) == 0:
            continue
        ff = Path(args.feat_dir) / f"{args.split}_{safe(country)}.npz" if args.feat_dir else None
        tf = time.time()
        if args.reuse_features and ff is not None and ff.exists():
            z = np.load(ff)
            s1i, rid, X = z["s1i"], z["rid"], z["X"]
            print(f"  loaded saved candidates+features {ff}", flush=True)
        else:
            cand = generate_candidates(s1, s2, s3, cfg["k"], cfg["m"],
                                       max_df_dict(cfg["max_df"], cfg.get("max_df_pair", cfg["max_df"])),
                                       cfg["chunk"], f"{args.cache_dir}/{args.split}_{safe(country)}",
                                       rev_m=cfg.get("rev_m", 0))
            tf = time.time()
            X = build_features(cand, s1, rec)
            s1i, rid = cand["s1i"].to_numpy(), cand["rid"].to_numpy()
            del cand
            if ff is not None:
                ff.parent.mkdir(parents=True, exist_ok=True)
                np.savez(ff, s1i=s1i, rid=rid, X=X)
        prob = booster.predict(X)
        thr = cfg.get("thresholds", {}).get(str(country), cfg["threshold"])
        acc = decide(s1i, rid, prob, thr)
        a = s1["entity_id"].to_numpy()[s1i]
        b = rec["entity_id"].to_numpy()[rid]
        cand_s1.append(a); cand_r.append(b); match_s1.append(a[acc]); match_r.append(b[acc])
        print(f"  {s1i.size:,} candidates ({s1i.size / len(s1):.2f}/S1), {acc.sum():,} matches, "
              f"threshold {thr:.3f}, features+predict {time.time() - tf:.0f}s", flush=True)
        del X

    cat = lambda xs: np.concatenate(xs) if xs else np.array([], dtype=object)
    write_submission(args.output, s1_all["entity_id"].tolist(),
                     cat(cand_s1), cat(cand_r), cat(match_s1), cat(match_r))
    print(f"\nwrote {args.output}/matching_results.tsv and candidate_pairs.tsv "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
