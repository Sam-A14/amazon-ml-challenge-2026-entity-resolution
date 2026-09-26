"""Candidate generation (blocking), v3.

Per country (records never match across countries in the training data):
  1. Each record gets four kinds of hashed keys:
       nu  name words (+ squashed name, squashed name without last word -> handles/domains)
       np  unordered pairs of name words (order-invariant, much rarer than single words)
       au  address words
       ab  adjacent address word pairs inside each comma-separated component
     Keys are IDF weighted over all three sources of the country; keys seen once, or more
     often than a cap (per kind), are dropped so the vectors stay sparse.
  2. Cosine similarity S1 x S2 and S1 x S3 in row chunks (sparse matrix products), keeping
     the top-k per source for every S1 row.
  3. Reverse filter: each S2/S3 record stays a candidate only for its top-m S1 records
     (EDA: no S2/S3 record belongs to more than one S1 entity).
Raw hashed matrices are cached on disk so repeated experiments skip tokenization.
"""
import gc
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize

from .normalize import clean_tokens, name_skeleton

N_FEATURES = 2 ** 22
KINDS = ("nu", "np", "au", "ab", "sk")
KEY_VERSION = "v4"          # bump when key generation changes (invalidates cache)
MAX_NAME_WORDS_FOR_PAIRS = 6


def record_keys(name, addr):
    """Blocking keys of one record, by kind."""
    nt = clean_tokens(name)
    seen, uniq = set(), []
    for t in nt:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    nu = set(uniq)
    if len(nt) >= 2:
        nu.add("".join(nt))
    if len(nt) >= 3:
        nu.add("".join(nt[:-1]))
        nu.add("".join(nt[:2]))                      # prefix squash: "pkmedia.com"
    if len(nt) >= 4:
        nu.add("".join(nt[:3]))
    npairs = {"|".join(sorted(p)) for p in combinations(uniq[:MAX_NAME_WORDS_FOR_PAIRS], 2)}
    # phonetic skeleton keys: link transliterated (Indian-script) names and typos
    sk_words = list(dict.fromkeys(name_skeleton(nt)))
    sk = set(sk_words)
    sk.update("|".join(sorted(p)) for p in combinations(sk_words[:MAX_NAME_WORDS_FOR_PAIRS], 2))
    if len(sk_words) >= 2:
        sk.add("#" + "".join(sk_words))
    au, ab = set(), set()
    for comp in (addr or "").split(","):
        ct = clean_tokens(comp)
        au.update(ct)
        ab.update(f"{a}_{b}" for a, b in zip(ct, ct[1:]))
    return nu, npairs, au, ab, sk


def _raw_matrices(df, batch=100_000):
    """Binary hashed matrices (one per key kind) for a frame of records.
    Keys are generated in batches so only `batch` records' key sets are in memory at once."""
    h = FeatureHasher(n_features=N_FEATURES, input_type="string",
                      alternate_sign=False, dtype=np.float32)
    names = df["business_name"].to_numpy()
    addrs = df["business_address"].to_numpy()
    parts = {kind: [] for kind in KINDS}
    for s in range(0, len(df), batch):
        keys = [record_keys(n, a) for n, a in zip(names[s:s + batch], addrs[s:s + batch])]
        for i, kind in enumerate(KINDS):
            parts[kind].append(h.transform(k[i] for k in keys).tocsr())
        del keys
    out = {}
    for kind in KINDS:
        m = sp.vstack(parts[kind], format="csr") if parts[kind] else \
            sp.csr_matrix((0, N_FEATURES), dtype=np.float32)
        m.data[:] = 1.0
        out[kind] = m
    return out


def _cached_raw(df, cache_path):
    if cache_path is None:
        return _raw_matrices(df)
    paths = {k: Path(f"{cache_path}_{KEY_VERSION}_{k}.npz") for k in KINDS}
    if all(p.exists() for p in paths.values()):
        mats = {k: sp.load_npz(p).tocsr() for k, p in paths.items()}
        if all(m.shape[0] == len(df) for m in mats.values()):
            return mats
    mats = _raw_matrices(df)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    for k, p in paths.items():
        sp.save_npz(p, mats[k], compressed=False)
    return mats


def build_matrices(recs_by_source, max_df, cache_prefix=None, weights=None):
    """recs_by_source: dict src -> DataFrame. max_df: dict kind -> df cap.
    Returns dict src -> L2-normalized IDF-weighted sparse matrix."""
    weights = weights or {}
    t0 = time.time()
    raw = {s: _cached_raw(df, None if cache_prefix is None else f"{cache_prefix}_{s}")
           for s, df in recs_by_source.items()}
    print(f"  keys ready {time.time() - t0:.0f}s", flush=True)
    n_docs = sum(len(df) for df in recs_by_source.values())
    idf = {}
    for kind in KINDS:
        dfc = np.zeros(N_FEATURES, dtype=np.int64)
        for s in raw:
            dfc += np.bincount(raw[s][kind].indices, minlength=N_FEATURES)
        w = np.log((n_docs + 1) / (dfc + 1)).astype(np.float32) * weights.get(kind, 1.0)
        w[(dfc < 2) | (dfc > max_df[kind])] = 0.0
        idf[kind] = w
    out = {}
    for s in list(raw):
        m = sp.hstack([raw[s][k] @ sp.diags(idf[k]) for k in KINDS], format="csr")
        m.eliminate_zeros()
        out[s] = normalize(m, norm="l2", copy=False).astype(np.float32)
        del raw[s]
    return out


def _topk_rows(c, k):
    """Top-k entries of every row of CSR matrix c -> (row, col, score) arrays."""
    if c.nnz == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32)
    rows = np.repeat(np.arange(c.shape[0]), np.diff(c.indptr))
    order = np.lexsort((-c.data, rows))
    rank = np.arange(order.size) - c.indptr[rows[order]]
    keep = order[rank < k]
    return rows[keep], c.indices[keep].astype(np.int64), c.data[keep]


def retrieve(q, r, k, chunk=250, label="", rows_idx=None):
    """Top-k columns of q @ r.T for every row of q (or only the rows in rows_idx)."""
    rt = r.T.tocsr()
    del r
    rows_idx = np.arange(q.shape[0]) if rows_idx is None else np.asarray(rows_idx)
    out_r, out_c, out_s = [], [], []
    t0 = time.time()
    for n, start in enumerate(range(0, rows_idx.size, chunk)):
        idx = rows_idx[start:start + chunk]
        c = (q[idx] @ rt).tocsr()
        a, b, s = _topk_rows(c, k)
        out_r.append(idx[a]); out_c.append(b); out_s.append(s)
        if n % 200 == 0:
            print(f"    {label} {start + idx.size:,}/{rows_idx.size:,} rows  {time.time() - t0:.0f}s",
                  flush=True)
    return np.concatenate(out_r), np.concatenate(out_c), np.concatenate(out_s)


def rank_within(groups, scores):
    """Rank (0 = best) of each score within its group."""
    order = np.lexsort((-scores, groups))
    g = groups[order]
    first = np.r_[True, g[1:] != g[:-1]]
    start_pos = np.maximum.accumulate(np.where(first, np.arange(g.size), 0))
    rank = np.empty(g.size, dtype=np.int64)
    rank[order] = np.arange(g.size) - start_pos
    return rank


def retrieve_all(s1, s2, s3, k, max_df, chunk=250, cache_prefix=None, rows_idx=None):
    """Top-k S2 and top-k S3 candidates per S1 row. rid indexes concat(s2, s3)."""
    mats = build_matrices({"s1": s1, "s2": s2, "s3": s3}, max_df, cache_prefix)
    q = mats.pop("s1")
    a2, b2, c2 = retrieve(q, mats.pop("s2"), k, chunk, "S2", rows_idx)   # pop: free after use
    gc.collect()
    a3, b3, c3 = retrieve(q, mats.pop("s3"), k, chunk, "S3", rows_idx)
    del q, mats
    gc.collect()
    return (np.concatenate([a2, a3]), np.concatenate([b2, b3 + len(s2)]),
            np.concatenate([c2, c3]).astype(np.float32),
            np.concatenate([np.zeros(a2.size, np.int8), np.ones(a3.size, np.int8)]))


def generate_candidates(s1, s2, s3, k, m, max_df, chunk=250, cache_prefix=None):
    """Final candidate set for one country (DataFrame with context features)."""
    import pandas as pd
    s1i, rid, score, src = retrieve_all(s1, s2, s3, k, max_df, chunk, cache_prefix)
    if m is not None and s1i.size:
        keep = rank_within(rid, score) < m
        s1i, rid, score, src = s1i[keep], rid[keep], score[keep], src[keep]
    return add_context(pd.DataFrame({"s1i": s1i, "rid": rid, "src": src, "score": score}))


def add_context(df):
    """Competition features: how a pair ranks among its S1's and its S2/S3 record's candidates."""
    df["fwd_rank"] = rank_within(df["s1i"].to_numpy() * 2 + df["src"].to_numpy(), df["score"].to_numpy())
    df["rev_rank"] = rank_within(df["rid"].to_numpy(), df["score"].to_numpy())
    g1 = df.groupby("s1i")["score"]
    df["s1_best"] = g1.transform("max")
    df["n_cands_s1"] = g1.transform("size").astype(np.float32)
    gr = df.groupby("rid")["score"]
    df["r_best"] = gr.transform("max")
    df["n_cands_r"] = gr.transform("size").astype(np.float32)
    second = df.loc[df["rev_rank"] == 1].set_index("rid")["score"]
    df["r_second"] = df["rid"].map(second).fillna(0.0).astype(np.float32)
    df["s1_gap"] = df["s1_best"] - df["score"]
    df["s1_ratio"] = df["score"] / df["s1_best"].clip(lower=1e-6)
    df["r_gap"] = df["r_best"] - df["score"]
    df["r_margin"] = df["r_best"] - df["r_second"]
    return df


def filter_candidates(s1, rid, score, fwd_rank, k, m):
    """Keep pairs with forward rank < k (per S1, per source) and reverse rank < m."""
    keep = fwd_rank < k
    s1, rid, score = s1[keep], rid[keep], score[keep]
    if m is not None and s1.size:
        keep = rank_within(rid, score) < m
        s1, rid, score = s1[keep], rid[keep], score[keep]
    return s1, rid, score


def max_df_dict(uni, pair):
    """Per-kind df caps: single words vs word pairs (skeleton keys use the pair cap)."""
    return {"nu": uni, "au": uni, "np": pair, "ab": pair, "sk": pair}
