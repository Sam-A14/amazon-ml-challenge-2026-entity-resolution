"""Second-stage matcher (stacking), run on SAVED features - takes minutes, not hours.

Stage 1: LightGBM trained with 3-fold cross-fitting grouped by Source 1 entity, giving
         out-of-fold probabilities on train and averaged fold probabilities on val/test.
Stage 2: LightGBM on [original features + competition features derived from stage-1
         probabilities among the SAME Source 1 entity's candidates]: the pair's probability,
         its rank, the best probability, the gap to it, how many candidates look like matches,
         and their probability mass. (Only Source-1-side groups are used: training samples
         contain all candidates of a sampled S1 entity but not all competitors of an S2/S3
         record, so S2/S3-side groups would differ between train and test.)
Thresholds (global + per country) are tuned on validation macro F0.5 as in train.py; the
one-owner rule is applied as before. The result is only kept if validation improves.

Usage (after `src.train ... --cache-dir E:\\ml_cache` saved features):
  python -m src.stage2 --data-dir dataset --cache-dir E:\\ml_cache --output output
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data_loader import countries_of, load_split
from .eval_blocking import safe
from .evaluation import decide, macro_f05_fast
from .submission import write_submission
from .train import KEY_S1

GROUP_FEATURES = ["p1", "p1_rank_s1", "p1_max_s1", "p1_gap_s1", "p1_n50_s1", "p1_sum_s1",
                  "p1_ratio_s1", "p1_second_s1"]


def group_features(s1_key, p):
    """Competition features of stage-1 probability p within each Source 1 entity."""
    df = pd.DataFrame({"g": s1_key, "p": p})
    g = df.groupby("g")["p"]
    mx = g.transform("max").to_numpy()
    rank = g.rank(ascending=False, method="first").to_numpy()
    n50 = df.assign(h=(df["p"] >= 0.5).astype(np.float32)).groupby("g")["h"].transform("sum").to_numpy()
    sm = g.transform("sum").to_numpy()
    second = df.assign(r=rank).query("r == 2").set_index("g")["p"]
    sec = df["g"].map(second).fillna(0.0).to_numpy()
    return np.column_stack([p, rank, mx, mx - p, n50, sm, p / np.maximum(mx, 1e-6), sec]
                           ).astype(np.float32)


def tune(pv, kva, rva, yva, ents, gold, countries):
    """Global and per-country thresholds maximising validation macro F0.5."""
    grid = np.arange(0.10, 0.96, 0.025)
    best = max((macro_f05_fast(ents, gold, kva, yva, decide(kva, rva, pv, t))[0], float(t)) for t in grid)
    f_glob, t_glob = best
    thresholds, row_c, ent_c = {}, kva // KEY_S1, ents // KEY_S1
    for ci, c in enumerate(countries):
        rm, em = row_c == ci, ent_c == ci
        if em.any():
            thresholds[str(c)] = max((macro_f05_fast(ents[em], gold[em], kva[rm], yva[rm],
                                                     decide(kva[rm], rva[rm], pv[rm], t))[0], float(t))
                                     for t in grid)[1]
    t_row = np.array([thresholds.get(str(countries[c]), t_glob) for c in row_c])
    f_pc, p_pc, r_pc = macro_f05_fast(ents, gold, kva, yva, decide(kva, rva, pv, t_row))
    if f_pc > f_glob:
        return f_pc, p_pc, r_pc, t_glob, thresholds
    f, p, r = macro_f05_fast(ents, gold, kva, yva, decide(kva, rva, pv, t_glob))
    return f, p, r, t_glob, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--feat-dir", default=None)
    ap.add_argument("--output", default="output")
    ap.add_argument("--model-dir", default="models/stage2")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--s1-rounds", type=int, default=800)
    ap.add_argument("--s2-rounds", type=int, default=3000)
    ap.add_argument("--baseline", type=float, default=None,
                    help="validation F0.5 to beat (default: models/config.json val_macro_f05)")
    ap.add_argument("--no-predict", action="store_true", help="only report validation")
    args = ap.parse_args()
    feat_dir = Path(args.feat_dir or Path(args.cache_dir) / "features")
    t0 = time.time()

    z = np.load(feat_dir / "train_features.npz", allow_pickle=True)
    Xtr, ytr, ktr = z["Xtr"], z["ytr"], z["ktr"]
    Xva, yva, kva, rva, ents, gold = (z[n] for n in ("Xva", "yva", "kva", "rva", "ents", "gold"))
    countries = [str(c) for c in z["countries"]]
    print(f"train pairs {len(ytr):,}  val pairs {len(yva):,}", flush=True)

    # ---- stage 1: cross-fitted probabilities -------------------------------------------
    p1 = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=0,
              verbose=-1, seed=7)
    fold = (ktr * 2654435761 % 2**32) % args.folds          # grouped by S1 entity
    oof = np.zeros(len(ytr), dtype=np.float32)
    models1 = []
    for f in range(args.folds):
        tr = fold != f
        m = lgb.train(p1, lgb.Dataset(Xtr[tr], ytr[tr]), num_boost_round=args.s1_rounds)
        oof[~tr] = m.predict(Xtr[~tr])
        models1.append(m)
        print(f"  stage-1 fold {f + 1}/{args.folds} done ({time.time() - t0:.0f}s)", flush=True)
    pva1 = np.mean([m.predict(Xva) for m in models1], axis=0)

    f1, *_ = tune(pva1, kva, rva, yva, ents, gold, countries)
    print(f"stage-1 (cross-fitted) validation macro F0.5 = {f1:.4f}", flush=True)

    # ---- stage 2 ----------------------------------------------------------------------
    Gtr, Gva = group_features(ktr, oof), group_features(kva, pva1)
    p2 = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=0,
              verbose=-1, seed=42)
    dtr = lgb.Dataset(np.hstack([Xtr, Gtr]), ytr)
    m2 = lgb.train(p2, dtr, num_boost_round=args.s2_rounds,
                   valid_sets=[lgb.Dataset(np.hstack([Xva, Gva]), yva, reference=dtr)],
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    pva2 = m2.predict(np.hstack([Xva, Gva]), num_iteration=m2.best_iteration)
    f2, pr2, rc2, t_glob, thresholds = tune(pva2, kva, rva, yva, ents, gold, countries)
    base = args.baseline
    if base is None:
        cfgp = Path("models/config.json")
        base = json.loads(cfgp.read_text())["val_macro_f05"] if cfgp.exists() else 0.0
    print(f"\nSTAGE-2 validation macro F0.5 = {f2:.4f} (precision {pr2:.4f}, recall {rc2:.4f}); "
          f"single model = {base:.4f}; thresholds {thresholds or t_glob}", flush=True)

    md = Path(args.model_dir); md.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models1):
        m.save_model(str(md / f"stage1_fold{i}.txt"))
    m2.save_model(str(md / "stage2.txt"), num_iteration=m2.best_iteration)
    (md / "config.json").write_text(json.dumps(dict(
        threshold=t_glob, thresholds=thresholds, val_macro_f05=f2, val_precision=pr2,
        val_recall=rc2, stage1_val_macro_f05=f1, folds=args.folds, s1_rounds=args.s1_rounds,
        best_iteration=m2.best_iteration, group_features=GROUP_FEATURES), indent=2))
    if args.no_predict:
        return
    if f2 <= base:
        print("stage 2 does NOT beat the single model -> not writing test predictions")
        return

    # ---- test: re-score saved candidates+features -------------------------------------
    s1_all, s2_all, s3_all = load_split(args.data_dir, "test")
    cand_s1, cand_r, match_s1, match_r = [], [], [], []
    for country in countries_of(s1_all, s2_all, s3_all):
        s1 = s1_all[s1_all["country"] == country].reset_index(drop=True)
        rec = pd.concat([s2_all[s2_all["country"] == country],
                         s3_all[s3_all["country"] == country]], ignore_index=True)
        ff = feat_dir / f"test_{safe(country)}.npz"
        if len(s1) == 0 or not ff.exists():
            print(f"  {country}: no saved features ({ff}) - run src.predict first")
            continue
        zt = np.load(ff)
        s1i, rid, X = zt["s1i"], zt["rid"], zt["X"]
        pt1 = np.mean([m.predict(X) for m in models1], axis=0)
        pt2 = m2.predict(np.hstack([X, group_features(s1i, pt1)]), num_iteration=m2.best_iteration)
        thr = thresholds.get(str(country), t_glob)
        acc = decide(s1i, rid, pt2, thr)
        a = s1["entity_id"].to_numpy()[s1i]
        b = rec["entity_id"].to_numpy()[rid]
        cand_s1.append(a); cand_r.append(b); match_s1.append(a[acc]); match_r.append(b[acc])
        print(f"  {country}: {s1i.size:,} candidates, {acc.sum():,} matches (threshold {thr:.3f})",
              flush=True)
        del X, zt
    write_submission(args.output, s1_all["entity_id"].tolist(), np.concatenate(cand_s1),
                     np.concatenate(cand_r), np.concatenate(match_s1), np.concatenate(match_r))
    print(f"wrote {args.output}/matching_results.tsv and candidate_pairs.tsv "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
