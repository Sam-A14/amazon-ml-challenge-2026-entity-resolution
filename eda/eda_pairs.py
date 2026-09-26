"""
EDA part 2 (fast): analyses a random sample of TRUE pairs and writes a small upload sample.
Loads only the train files (not test) to save RAM.
Run:  python eda_pairs.py --data-dir dataset
Outputs: eda_report2.txt and sample/*.tsv
"""
import argparse
import random
import re
from pathlib import Path

import pandas as pd

TOK = re.compile(r"[^\w]+", re.UNICODE)       # split on anything that is not a letter/digit
POSTAL = re.compile(r"\b\d{3}\s?\d{2,3}\b")   # rough ZIP / PIN / code postal


def load(path):
    """Read a challenge TSV as plain strings."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def toks(s):
    """Lowercased word tokens as a set."""
    return {t for t in TOK.split(str(s).lower()) if t}


def postal(s):
    m = POSTAL.findall(str(s))
    return m[-1].replace(" ", "") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--n-pairs", type=int, default=200_000)
    ap.add_argument("--n-sample", type=int, default=2000)
    ap.add_argument("--n-distractors", type=int, default=20000)
    args = ap.parse_args()
    d = Path(args.data_dir) / "train"
    random.seed(42)

    lines = []
    def out(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    print("loading train files ...", flush=True)
    s1 = load(d / "train_source1.tsv")
    rec = pd.concat([load(d / "train_source2.tsv"), load(d / "train_source3.tsv")], ignore_index=True)
    gt = load(d / "train_ground_truth.tsv")
    print("loaded", flush=True)

    # ground truth -> one row per (S1, matched id)
    pairs = gt.assign(rid=gt["matched_entity_ids"].str.split(",")).explode("rid")
    pairs = pairs[pairs["rid"].notna() & (pairs["rid"] != "")][["source1_entity_id", "rid"]]
    ps = pairs.sample(min(args.n_pairs, len(pairs)), random_state=42)

    m = (ps.merge(s1, left_on="source1_entity_id", right_on="entity_id")
           .merge(rec, left_on="rid", right_on="entity_id", suffixes=("_1", "_2")))
    m["src"] = m["rid"].str[:2]

    n1, n2 = m["business_name_1"].map(toks), m["business_name_2"].map(toks)
    a1, a2 = m["business_address_1"].map(toks), m["business_address_2"].map(toks)
    m["same_country"] = m["country_1"] == m["country_2"]
    m["name_exact_lower"] = m["business_name_1"].str.lower().str.strip() == m["business_name_2"].str.lower().str.strip()
    m["name_shares_token"] = [len(x & y) > 0 for x, y in zip(n1, n2)]
    m["name_jaccard"] = [len(x & y) / max(len(x | y), 1) for x, y in zip(n1, n2)]
    m["addr_shares_token"] = [len(x & y) > 0 for x, y in zip(a1, a2)]
    m["name_or_addr_shares_token"] = m["name_shares_token"] | m["addr_shares_token"]
    m["addr2_empty"] = m["business_address_2"].str.strip() == ""
    pc1, pc2 = m["business_address_1"].map(postal), m["business_address_2"].map(postal)
    both = pc1.notna() & pc2.notna()
    m["postal_both"] = both
    m["postal_equal_when_both"] = (pc1 == pc2).where(both)

    cols = ["same_country", "name_exact_lower", "name_shares_token", "name_jaccard",
            "addr_shares_token", "name_or_addr_shares_token", "addr2_empty",
            "postal_both", "postal_equal_when_both"]
    out(f"=== TRUE-PAIR STATS on {len(m):,} sampled pairs (means)")
    out(m[cols].astype(float).mean().round(4).to_string())
    out("\n--- by S1 country")
    out(m.groupby("country_1")[cols].agg(lambda s: s.astype(float).mean()).round(3).T.to_string())
    out("\n--- by source (S2 vs S3)")
    out(m.groupby("src")[cols].agg(lambda s: s.astype(float).mean()).round(3).T.to_string())
    out("\nname_jaccard quantiles [.05,.25,.5,.75]:",
        m["name_jaccard"].quantile([.05, .25, .5, .75]).round(3).tolist())

    out("\n=== 40 RANDOM TRUE PAIRS (S1 line, then matched record)")
    for _, r in m.sample(40, random_state=1).iterrows():
        out(f"{r.source1_entity_id} | {r.business_name_1} | {r.business_address_1} | {r.country_1}")
        out(f"{r.rid} | {r.business_name_2} | {r.business_address_2} | {r.country_2}\n")

    out("=== 20 HARDEST-LOOKING TRUE PAIRS (no shared name token)")
    hard = m[~m["name_shares_token"]]
    for _, r in hard.sample(min(20, len(hard)), random_state=2).iterrows():
        out(f"{r.source1_entity_id} | {r.business_name_1} | {r.business_address_1} | {r.country_1}")
        out(f"{r.rid} | {r.business_name_2} | {r.business_address_2} | {r.country_2}\n")

    Path("eda_report2.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote eda_report2.txt", flush=True)

    # ---- small sample: keep true matches intact, add random distractors ----
    keep = set(random.sample(list(gt["source1_entity_id"]), args.n_sample))
    gts = gt[gt["source1_entity_id"].isin(keep)]
    matched = set(pairs.loc[pairs["source1_entity_id"].isin(keep), "rid"])
    o = Path("sample"); o.mkdir(exist_ok=True)
    s1[s1["entity_id"].isin(keep)].to_csv(o / "train_source1.tsv", sep="\t", index=False)
    rest = rec[~rec["entity_id"].isin(matched)].sample(args.n_distractors, random_state=42)
    both_src = pd.concat([rec[rec["entity_id"].isin(matched)], rest])
    for k in ["S2", "S3"]:
        both_src[both_src["entity_id"].str.startswith(k)].to_csv(
            o / f"train_source{k[1]}.tsv", sep="\t", index=False)
    gts.to_csv(o / "train_ground_truth.tsv", sep="\t", index=False)
    print("wrote sample/*.tsv  -- done", flush=True)


if __name__ == "__main__":
    main()