"""Train the matching model.

Steps per training country:
  1. Generate candidates for ALL Source 1 records (the reverse filter needs every competitor).
  2. Split Source 1 entities randomly into train / validation (entity-level, no leakage of an
     entity's pairs across the split) and subsample them to keep feature time manageable.
  3. Label candidates with the ground truth and compute features.
Then train LightGBM (MIT licence), tune the probability threshold on validation macro F0.5
(official metric, singletons and blocking misses included), save model + config.

Usage:
  python -m src.train --data-dir dataset --k 8 --m 2
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import generate_candidates, max_df_dict
from .eval_blocking import safe
from .data_loader import countries_of, index_of, load_ground_truth, load_split
from .evaluation import decide, macro_f05_fast
from .features import FEATURES, build_features

KEY_S1, KEY_R = 10_000_000, 100_000_000        # country-prefixed integer keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--k", type=int, default=8, help="candidates per S1 per source")
    ap.add_argument("--m", type=int, default=2, help="reverse filter: S1s kept per S2/S3 record")
    ap.add_argument("--max-df", type=int, default=5000, help="cap for single-word keys")
    ap.add_argument("--max-df-pair", type=int, default=5000, help="cap for word-pair keys")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--n-train", type=int, default=150_000, help="train S1 entities per country")
    ap.add_argument("--n-val", type=int, default=50_000, help="validation S1 entities per country")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()

    s1_all, s2_all, s3_all = load_split(args.data_dir, "train")
    gt = load_ground_truth(args.data_dir)
    countries = args.countries or countries_of(s1_all, s2_all, s3_all)
    parts = {"train": [], "val": []}
    val_entities, val_gold = [], []
    block_stats = []

    for ci, country in enumerate(countries):
        tc = time.time()
        s1 = s1_all[s1_all["country"] == country].reset_index(drop=True)
        s2 = s2_all[s2_all["country"] == country].reset_index(drop=True)
        s3 = s3_all[s3_all["country"] == country].reset_index(drop=True)
        rec = pd.concat([s2, s3], ignore_index=True)
        print(f"\n== {country}: S1={len(s1):,} S2+S3={len(rec):,}", flush=True)
        cand = generate_candidates(s1, s2, s3, args.k, args.m,
                                   max_df_dict(args.max_df, args.max_df_pair), args.chunk,
                                   f"{args.cache_dir}/train_{safe(country)}")

        g = gt[gt["source1_entity_id"].isin(s1["entity_id"])]
        g1 = index_of(s1["entity_id"], g["source1_entity_id"])
        g2 = index_of(rec["entity_id"], g["rid"])
        ok = (g1 >= 0) & (g2 >= 0)
        g1, g2 = g1[ok], g2[ok]
        n_gold = np.bincount(g1, minlength=len(s1))
        pair_key = cand["s1i"].to_numpy().astype(np.int64) * len(rec) + cand["rid"].to_numpy()
        label = np.isin(pair_key, g1.astype(np.int64) * len(rec) + g2)
        recall = label.sum() / max(g1.size, 1)
        size = len(cand) / len(s1)
        block_stats.append((country, recall, size))
        print(f"  blocking: pair_recall={recall:.4f} cands/S1={size:.2f} "
              f"({time.time() - tc:.0f}s)", flush=True)

        perm = np.random.default_rng(42 + ci).permutation(len(s1))
        n_val = min(args.n_val, len(s1) // 5)
        split = {"val": perm[:n_val], "train": perm[n_val:n_val + args.n_train]}
        for part, idx in split.items():
            mask = np.isin(cand["s1i"].to_numpy(), idx)
            sel = cand[mask]
            tf = time.time()
            X = build_features(sel, s1, rec)
            print(f"  {part}: {len(idx):,} S1, {len(sel):,} pairs, features {time.time() - tf:.0f}s",
                  flush=True)
            parts[part].append((X, label[mask],
                                ci * KEY_S1 + sel["s1i"].to_numpy(),
                                ci * KEY_R + sel["rid"].to_numpy()))
            if part == "val":
                val_entities.append(ci * KEY_S1 + idx)
                val_gold.append(n_gold[idx])
        del cand

    def stack(p):
        return [np.concatenate([x[i] for x in parts[p]]) for i in range(4)]
    Xtr, ytr, _, _ = stack("train")
    Xva, yva, kva, rva = stack("val")
    ents, gold = np.concatenate(val_entities), np.concatenate(val_gold)
    print(f"\ntraining LightGBM on {len(ytr):,} pairs ({ytr.mean():.3f} positive)", flush=True)

    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  num_threads=0, verbose=-1, seed=42)
    dtrain = lgb.Dataset(Xtr, ytr, feature_name=FEATURES)
    booster = lgb.train(params, dtrain,
                        num_boost_round=2000,
                        valid_sets=[lgb.Dataset(Xva, yva, feature_name=FEATURES, reference=dtrain)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    pva = booster.predict(Xva, num_iteration=booster.best_iteration)

    print("\nthreshold  macroF05  precision  recall", flush=True)
    best = (-1, 0.5)
    for t in np.arange(0.10, 0.96, 0.025):
        acc = decide(kva, rva, pva, t)
        f, p, r = macro_f05_fast(ents, gold, kva, yva, acc)
        print(f"  {t:.3f}   {f:.4f}   {p:.4f}   {r:.4f}")
        if f > best[0]:
            best = (f, float(t), p, r)
    f, t, p, r = best
    ceiling = macro_f05_fast(ents, gold, kva, yva, yva.astype(bool))[0]
    print(f"\nBEST validation macro F0.5 = {f:.4f} at threshold {t:.3f} "
          f"(precision {p:.4f}, recall {r:.4f}); blocking ceiling {ceiling:.4f}", flush=True)

    md = Path(args.model_dir); md.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(md / "lgbm.txt"), num_iteration=booster.best_iteration)
    cfg = dict(k=args.k, m=args.m, max_df=args.max_df, max_df_pair=args.max_df_pair, chunk=args.chunk, threshold=t,
               features=FEATURES, val_macro_f05=f, val_precision=p, val_recall=r,
               val_ceiling=ceiling, blocking=block_stats)
    (md / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", ", ".join(n for n, _ in imp[:12]))

    log = Path("experiments/experiment_log.csv"); log.parent.mkdir(exist_ok=True)
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "k", "m", "max_df", "max_df_pair", "n_train", "n_val", "threshold",
                        "val_macro_f05", "precision", "recall", "ceiling", "blocking", "note"])
        w.writerow([datetime.now().isoformat(timespec="minutes"), args.k, args.m, args.max_df, args.max_df_pair,
                    args.n_train, args.n_val, round(t, 3), round(f, 4), round(p, 4), round(r, 4),
                    round(ceiling, 4), block_stats, args.note])
    print(f"saved {md}/lgbm.txt and config.json  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
