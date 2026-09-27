"""Official metric (macro F0.5 per Source 1 entity, singletons included) and match decisions."""
import numpy as np
import pandas as pd


def entity_f05(pred: set, gold: set) -> float:
    """F0.5 for one Source 1 entity, exactly as defined in the problem statement."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 1.25 * p * r / (0.25 * p + r)


def decide(s1_key, r_key, prob, threshold):
    """Boolean mask of accepted pairs: prob >= threshold, and each S2/S3 record is given
    to at most one Source 1 entity (the most probable one). The one-owner rule follows from
    the training data: no S2/S3 record belongs to more than one Source 1 entity."""
    keep = prob >= threshold          # threshold: scalar or one value per pair
    idx = np.flatnonzero(keep)
    if idx.size:
        df = pd.DataFrame({"r": r_key[idx], "p": prob[idx], "i": idx})
        best = df.sort_values("p", ascending=False).drop_duplicates("r")["i"].to_numpy()
        keep = np.zeros_like(keep)
        keep[best] = True
    return keep


def macro_f05_fast(entity_keys, n_gold, s1_key, is_true, accepted):
    """Vectorised macro F0.5 over `entity_keys` (all evaluated S1 entities, with their
    number of true matches n_gold, including matches missed by blocking)."""
    pos = pd.Index(entity_keys).get_indexer(s1_key)
    n = len(entity_keys)
    p = np.bincount(pos[accepted], minlength=n)
    h = np.bincount(pos[accepted & is_true], minlength=n)
    g = np.asarray(n_gold)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(p > 0, h / np.maximum(p, 1), 0.0)
        rec = np.where(g > 0, h / np.maximum(g, 1), 0.0)
        f = np.where(h > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where((g == 0) & (p == 0), 1.0, f)
    return f.mean(), (h.sum() / max(p.sum(), 1)), (h.sum() / max(g.sum(), 1))


def decide_expected_f(s1_key, r_key, prob, c=0.2, pmin=0.1):
    """Per-entity decision maximising the EXPECTED F0.5 (the metric is averaged per S1 entity).

    1. one-owner: each S2/S3 record keeps only its most probable S1 pair;
    2. per S1 entity, candidates sorted by probability; for k = 1..n the expected F0.5 of
       predicting the top-k is approximated by 1.25*sum(p_top_k) / (0.25*E|gold| + k), with
       E|gold| = sum of the entity's probabilities + c (c = expected true matches blocking missed);
    3. predicting nothing scores 1 only if the entity has no match: P = prod(1 - p) * exp(-c);
    4. choose the option with the highest expected F0.5. Pairs below pmin are never selected.
    """
    n = len(prob)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-prob, r_key))
    first = np.r_[True, r_key[order][1:] != r_key[order][:-1]]
    own = np.zeros(n, dtype=bool)
    own[order[first]] = True
    p = np.where(own & (prob >= pmin), prob, 0.0).astype(np.float64)

    o = np.lexsort((-p, s1_key))
    s, ps = s1_key[o], p[o]
    start = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(start) - 1
    starts = np.flatnonzero(start)
    k = np.arange(n) - starts[gid] + 1
    csum = np.cumsum(ps)
    base = np.r_[0.0, csum][starts][gid]
    S = csum - base                                   # sum of top-k probabilities
    tot = np.bincount(gid, weights=ps)[gid]           # sum of all probabilities in the entity
    F = 1.25 * S / (0.25 * (tot + c) + k)
    F[ps <= 0] = -1.0
    logq = np.bincount(gid, weights=np.log1p(-np.clip(ps, 0, 1 - 1e-9)))
    p_empty = np.exp(logq - c)
    ng = starts.size
    best_f = np.full(ng, -1.0)
    np.maximum.at(best_f, gid, F)
    is_best = F >= best_f[gid] - 1e-12
    kbest = np.full(ng, 0)
    np.maximum.at(kbest, gid, np.where(is_best, k, 0))  # largest k attaining the max
    take = (best_f > p_empty)[gid] & (k <= kbest[gid]) & (ps > 0)
    out = np.zeros(n, dtype=bool)
    out[o[take]] = True
    return out
