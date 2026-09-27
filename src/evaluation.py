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
