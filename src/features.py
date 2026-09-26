"""Pairwise features for (Source 1, Source 2/3) candidate pairs.

String features are computed with rapidfuzz on normalized text. They were chosen from the
noise seen in the EDA: word-order shuffles (token_sort/token_set), partial names (partial,
containment), typos (ratio, Jaro-Winkler), website/handle names (squashed-name features),
altered or truncated house numbers (number features), partial/missing addresses.
Context features (retrieval score, ranks, gaps) come from blocking.add_context.
"""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import clean_tokens, name_skeleton

STRING_FEATURES = [
    "name_ratio", "name_tsort", "name_tset", "name_partial", "name_jw", "name_jacc",
    "name_contain", "sq_eq", "sq_contain", "sq_ratio", "n_tok1", "n_tok2", "name2_nonascii",
    "addr_ratio", "addr_tset", "addr_partial", "addr_jacc", "addr_contain2", "addr1_empty",
    "addr2_empty", "num_shared", "first_num_eq", "first_num_prefix", "a_tok1", "a_tok2",
    "name_in_addr2", "sk_ratio", "sk_tset", "sk_jacc",
]
CONTEXT_FEATURES = ["src", "score", "fwd_rank", "rev_rank", "s1_best", "n_cands_s1", "r_best",
                    "n_cands_r", "r_second", "s1_gap", "s1_ratio", "r_gap", "r_margin"]
FEATURES = CONTEXT_FEATURES + STRING_FEATURES


def _pair(n1, a1, n2, a2):
    t1, t2 = clean_tokens(n1), clean_tokens(n2)
    s1, s2 = " ".join(t1), " ".join(t2)
    q1, q2 = "".join(t1), "".join(t2)
    set1, set2 = set(t1), set(t2)
    u1, u2 = clean_tokens(a1), clean_tokens(a2)
    b1, b2 = " ".join(u1), " ".join(u2)
    aset1, aset2 = set(u1), set(u2)
    num1 = [x for x in u1 if x.isdigit()]
    num2 = [x for x in u2 if x.isdigit()]
    inter_n = len(set1 & set2)
    inter_a = len(aset1 & aset2)
    short, long_ = (q1, q2) if len(q1) <= len(q2) else (q2, q1)
    k1, k2 = name_skeleton(t1), name_skeleton(t2)
    ks1, ks2 = " ".join(k1), " ".join(k2)
    kset1, kset2 = set(k1), set(k2)
    f1 = num1[0] if num1 else ""
    f2 = num2[0] if num2 else ""
    return (
        fuzz.ratio(s1, s2), fuzz.token_sort_ratio(s1, s2), fuzz.token_set_ratio(s1, s2),
        fuzz.partial_ratio(s1, s2), JaroWinkler.similarity(s1, s2) if s1 and s2 else 0.0,
        inter_n / max(len(set1 | set2), 1), inter_n / max(min(len(set1), len(set2)), 1),
        float(bool(q1) and q1 == q2), float(len(short) >= 4 and short in long_),
        fuzz.ratio(q1, q2), len(t1), len(t2), float(not s2.isascii()),
        fuzz.ratio(b1, b2), fuzz.token_set_ratio(b1, b2), fuzz.partial_ratio(b1, b2),
        inter_a / max(len(aset1 | aset2), 1), inter_a / max(len(aset2), 1),
        float(not u1), float(not u2),
        len(set(num1) & set(num2)), float(bool(f1) and f1 == f2),
        float(bool(f1) and bool(f2) and f1 != f2 and (f1.startswith(f2) or f2.startswith(f1))),
        len(u1), len(u2), len(set1 & aset2) / max(len(set1), 1),
        fuzz.ratio(ks1, ks2), fuzz.token_set_ratio(ks1, ks2),
        len(kset1 & kset2) / max(len(kset1 | kset2), 1),
    )


def string_features(n1, a1, n2, a2, chunk=200_000):
    """Feature matrix (float32) for aligned lists of names/addresses."""
    n = len(n1)
    out = np.empty((n, len(STRING_FEATURES)), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = np.array([_pair(*x) for x in zip(n1[s:e], a1[s:e], n2[s:e], a2[s:e])],
                            dtype=np.float32).reshape(e - s, len(STRING_FEATURES))
    return out


def build_features(cand, s1, rec):
    """cand: candidate DataFrame (from generate_candidates); s1/rec: record frames
    (rec = concat of S2 and S3 for the country, aligned with cand.rid)."""
    i, j = cand["s1i"].to_numpy(), cand["rid"].to_numpy()
    sf = string_features(s1["business_name"].to_numpy()[i].tolist(),
                         s1["business_address"].to_numpy()[i].tolist(),
                         rec["business_name"].to_numpy()[j].tolist(),
                         rec["business_address"].to_numpy()[j].tolist())
    ctx = cand[CONTEXT_FEATURES].to_numpy(dtype=np.float32)
    return np.hstack([ctx, sf])
