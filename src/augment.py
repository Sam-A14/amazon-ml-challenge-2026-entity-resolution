"""v9: house-number relationship features on the SAVED candidates (no re-blocking).

Model-error analysis showed the dataset's rule for numbers in addresses:
  same business      -> numbers equal, truncated (171 ~ 1717, 69 ~ 169, 12 ~ 812) or off by one
  different business -> same street/name but an unrelated nearby number (3004 vs 3013, 84D vs 88D)
The v6 features only compared the FIRST number (equal / prefix), so the model confused the two.
For every S2/S3 number we classify its best relation to the S1 numbers (exact / affix /
off-by-one / other) and add counts, fractions and distances as features.

Steps (one command): load saved train/val features -> add number features -> train LightGBM ->
pick the best iteration and threshold on validation macro F0.5 -> if it beats --baseline,
re-score the saved test candidates with the same features and write both output files.
Usage:
  python -m src.augment --data-dir dataset --cache-dir E:\\ml_cache --output output --baseline 0.9648
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
from .features import FEATURES
from .normalize import clean_tokens
from .submission import write_submission
from .train import KEY_R, KEY_S1

NUM_FEATURES = ["r_nums", "s_nums", "n_exact", "n_affix", "n_near1", "n_other", "frac_other",
                "log_min_diff_other", "log_first_absdiff", "first_rel"]
# v10: name frequency - how many records share a name (decides records with no/thin address)
FREQ_FEATURES = ["s1name_in_s1", "rname_in_s1", "rname_in_r", "s1name_in_r", "same_norm_name"]


def norm_name(x):
    return " ".join(clean_tokens(x))


def name_counts(s1_names, rec_names):
    """Normalized names and their frequencies among S1 and among S2+S3 of one country."""
    n1 = pd.Series([norm_name(x) for x in s1_names])
    nr = pd.Series([norm_name(x) for x in rec_names])
    c1, cr = n1.value_counts(), nr.value_counts()
    return n1.to_numpy(), nr.to_numpy(), c1, cr


def freq_features(nc, i, j):
    n1, nr, c1, cr = nc
    a, b = n1[i], nr[j]
    return np.column_stack([
        pd.Series(a).map(c1).fillna(0).to_numpy(), pd.Series(b).map(c1).fillna(0).to_numpy(),
        pd.Series(b).map(cr).fillna(0).to_numpy(), pd.Series(a).map(cr).fillna(0).to_numpy(),
        (a == b).astype(np.float32)]).astype(np.float32)


def numbers(addr):
    return [t for t in clean_tokens(addr) if t.isdigit()][:8]


def _affix(x, y):
    s, l = (x, y) if len(x) <= len(y) else (y, x)
    return len(s) >= 2 and (l.startswith(s) or l.endswith(s))


def _rel(x, a, sa):
    """Best relation of S2/S3 number x to the S1 numbers a: 0 exact, 1 affix, 2 off-by-one, 3 other."""
    if x in sa:
        return 0, 0
    if any(_affix(x, y) for y in a):
        return 1, 0
    if len(x) < 12:
        d = [abs(int(x) - int(y)) for y in a if len(y) < 12]
        if d:
            m = min(d)
            return (2, m) if m <= 1 else (3, m)
    return 3, -1


def num_feats(a, b):
    """a: S1 numbers, b: S2/S3 numbers."""
    sa = set(a)
    cnt = [0, 0, 0, 0]
    mdo = -1
    for x in b:
        r, d = _rel(x, a, sa)
        cnt[r] += 1
        if r == 3 and d >= 0:
            mdo = d if mdo < 0 else min(mdo, d)
    if a and b:
        fr = _rel(b[0], a[:1], {a[0]})[0]
        fa = abs(int(a[0]) - int(b[0])) if len(a[0]) < 12 and len(b[0]) < 12 else -1
    else:
        fr, fa = -1, -1
    return (len(b), len(a), cnt[0], cnt[1], cnt[2], cnt[3], cnt[3] / max(len(b), 1),
            np.log1p(mdo) if mdo >= 0 else -1.0, np.log1p(fa) if fa >= 0 else -1.0, fr)


def extra_features(s1_addr, rec_addr, i, j, nc=None):
    """Number (+ name-frequency) features for pairs (i, j) of one country."""
    ui, ii = np.unique(i, return_inverse=True)
    uj, jj = np.unique(j, return_inverse=True)
    na = [numbers(x) for x in s1_addr[ui]]
    nb = [numbers(x) for x in rec_addr[uj]]
    out = np.empty((i.size, len(NUM_FEATURES)), dtype=np.float32)
    for n, (p, q) in enumerate(zip(ii, jj)):
        out[n] = num_feats(na[p], nb[q])
    if nc is not None:
        out = np.hstack([out, freq_features(nc, i, j)])
    return out


def add_for_keys(X, k1, kr, countries, frames):
    """Append number features to X, rows identified by country-prefixed keys."""
    E = np.zeros((X.shape[0], len(NUM_FEATURES) + len(FREQ_FEATURES)), dtype=np.float32)
    ci = k1 // KEY_S1
    for c, country in enumerate(countries):
        m = ci == c
        if not m.any():
            continue
        s1a, reca, nc = frames[country]
        E[m] = extra_features(s1a, reca, (k1[m] - c * KEY_S1).astype(np.int64),
                              (kr[m] - c * KEY_R).astype(np.int64), nc)
    return np.hstack([X, E])


def country_frames(data_dir, split, countries):
    s1a, s2a, s3a = load_split(data_dir, split)
    out = {}
    for c in countries:
        s1 = s1a[s1a["country"] == c].reset_index(drop=True)
        rec = pd.concat([s2a[s2a["country"] == c], s3a[s3a["country"] == c]], ignore_index=True)
        out[c] = (s1["business_address"].to_numpy(), rec["business_address"].to_numpy(),
                  name_counts(s1["business_name"].to_numpy(), rec["business_name"].to_numpy()))
    return s1a, s2a, s3a, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--output", default="output")
    ap.add_argument("--model-dir", default="models/v10")
    ap.add_argument("--baseline", type=float, required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--no-name-freq", action="store_true",
                    help="v9 variant: house-number features only (no name-frequency features)")
    args = ap.parse_args()
    fdir = Path(args.cache_dir) / "features"
    t0 = time.time()

    z = np.load(fdir / "train_features.npz", allow_pickle=True)
    countries = [str(c) for c in z["countries"]]
    _, _, _, frames = country_frames(args.data_dir, "train", countries)
    Xtr = add_for_keys(z["Xtr"], z["ktr"], z["rtr"], countries, frames)
    Xva = add_for_keys(z["Xva"], z["kva"], z["rva"], countries, frames)
    n_keep = z["Xtr"].shape[1] + len(NUM_FEATURES)
    if args.no_name_freq:
        Xtr, Xva = Xtr[:, :n_keep], Xva[:, :n_keep]
    ytr, yva, kva, rva, ents, gold = (z[n] for n in ("ytr", "yva", "kva", "rva", "ents", "gold"))
    del frames, z
    names = FEATURES + NUM_FEATURES + ([] if args.no_name_freq else FREQ_FEATURES)
    print(f"number features added ({time.time() - t0:.0f}s); train {len(ytr):,} val {len(yva):,}",
          flush=True)

    params = dict(objective="binary", learning_rate=args.lr, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=0,
                  verbose=-1, seed=42)
    dtr = lgb.Dataset(Xtr, ytr, feature_name=names)
    booster = lgb.train(params, dtr, num_boost_round=args.rounds,
                        valid_sets=[lgb.Dataset(Xva, yva, feature_name=names, reference=dtr)],
                        callbacks=[lgb.log_evaluation(200)])
    del Xtr, dtr

    grid = np.arange(0.30, 0.96, 0.025)
    best = (-1.0, 0, 0.5, 0, 0)
    for it in sorted({500, 1000, 1500, args.rounds}):
        if it > args.rounds:
            continue
        p = booster.predict(Xva, num_iteration=it)
        for t in grid:
            f, pr, rc = macro_f05_fast(ents, gold, kva, yva, decide(kva, rva, p, t))
            if f > best[0]:
                best = (f, it, float(t), pr, rc)
        print(f"  iteration {it}: best so far F0.5 {best[0]:.4f}", flush=True)
    f, it, t, pr, rc = best
    imp = sorted(zip(names, booster.feature_importance("gain")), key=lambda x: -x[1])
    print(f"\nV10 validation macro F0.5 = {f:.4f} (precision {pr:.4f}, recall {rc:.4f}) "
          f"at iteration {it}, threshold {t:.3f}; baseline {args.baseline:.4f}")
    print("new-feature importance ranks:",
          {n: r + 1 for r, (n, _) in enumerate(imp) if n in NUM_FEATURES + FREQ_FEATURES})

    md = Path(args.model_dir); md.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(md / "lgbm.txt"), num_iteration=it)
    (md / "config.json").write_text(json.dumps(dict(threshold=t, iteration=it, val_macro_f05=f,
        val_precision=pr, val_recall=rc, features=names), indent=2))
    if f <= args.baseline:
        print("v10 does NOT beat the baseline -> not writing test predictions")
        return

    s1a, s2a, s3a = load_split(args.data_dir, "test")
    cand_s1, cand_r, match_s1, match_r = [], [], [], []
    for country in countries_of(s1a, s2a, s3a):
        s1 = s1a[s1a["country"] == country].reset_index(drop=True)
        rec = pd.concat([s2a[s2a["country"] == country], s3a[s3a["country"] == country]],
                        ignore_index=True)
        ff = fdir / f"test_{safe(country)}.npz"
        if len(s1) == 0 or not ff.exists():
            print(f"  {country}: missing {ff}")
            continue
        zt = np.load(ff)
        s1i, rid = zt["s1i"], zt["rid"]
        nc = name_counts(s1["business_name"].to_numpy(), rec["business_name"].to_numpy())
        X = np.hstack([zt["X"], extra_features(s1["business_address"].to_numpy(),
                                               rec["business_address"].to_numpy(), s1i, rid,
                                               None if args.no_name_freq else nc)])
        prob = booster.predict(X, num_iteration=it)
        acc = decide(s1i, rid, prob, t)
        a = s1["entity_id"].to_numpy()[s1i]
        b = rec["entity_id"].to_numpy()[rid]
        cand_s1.append(a); cand_r.append(b); match_s1.append(a[acc]); match_r.append(b[acc])
        print(f"  {country}: {s1i.size:,} candidates, {acc.sum():,} matches", flush=True)
        del X, zt
    write_submission(args.output, s1a["entity_id"].tolist(), np.concatenate(cand_s1),
                     np.concatenate(cand_r), np.concatenate(match_s1), np.concatenate(match_r))
    print(f"wrote {args.output}/matching_results.tsv and candidate_pairs.tsv "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
