"""
EDA + sampling script for Amazon ML Challenge 2026 (Business Entity Resolution).

Run from the unzipped student_resource folder:
    python eda_and_sample.py --data-dir dataset

Outputs:
    eda_report.txt        -> full statistics (upload this)
    sample/*.tsv          -> small training sample with true matches kept intact (upload these)
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

POSTAL = re.compile(r"\b\d{3}\s?\d{2,3}\b")  # rough: US ZIP, FR code postal, IN PIN (incl. "560 001")


def load(path: Path) -> pd.DataFrame:
    """Read a challenge TSV as plain strings (no NaN conversion, no quote handling)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def parse_ids(s: str) -> list:
    """Split a comma-separated ID list, dropping empties."""
    return [x.strip() for x in s.split(",") if x.strip()]


def postal(addr: str):
    m = POSTAL.findall(addr or "")
    return m[-1].replace(" ", "") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--n-sample", type=int, default=2000, help="S1 entities to keep in sample")
    ap.add_argument("--n-distractors", type=int, default=5000, help="extra unmatched S2/S3 rows per source")
    args = ap.parse_args()
    d = Path(args.data_dir)
    random.seed(42)

    lines = []
    def out(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    src = {}
    for split in ["train", "test"]:
        for k in [1, 2, 3]:
            p = d / split / f"{split}_source{k}.tsv"
            df = load(p)
            src[(split, k)] = df
            out(f"\n=== {p.name}  rows={len(df):,}  cols={list(df.columns)}")
            out("  duplicate entity_id:", df["entity_id"].duplicated().sum())
            out("  exact duplicate (name,address,country):",
                df.duplicated(["business_name", "business_address", "country"]).sum())
            for c in ["business_name", "business_address", "country"]:
                empty = (df[c].str.strip() == "").sum()
                out(f"  {c}: empty={empty:,}  unique={df[c].nunique():,}")
            out("  country counts:", dict(Counter(df["country"]).most_common(10)))
            out("  name length (chars) quantiles:",
                df["business_name"].str.len().quantile([.05, .5, .95]).round(0).tolist())
            out("  address length quantiles:",
                df["business_address"].str.len().quantile([.05, .5, .95]).round(0).tolist())
            has_pc = df["business_address"].map(postal).notna()
            out("  share with postal-like code, by country:",
                df.assign(pc=has_pc).groupby("country")["pc"].mean().round(3).to_dict())

    # ---- ground truth ----
    gt = load(d / "train" / "train_ground_truth.tsv")
    gt["ids"] = gt["matched_entity_ids"].map(parse_ids)
    s1, s2, s3 = src[("train", 1)], src[("train", 2)], src[("train", 3)]
    out(f"\n=== GROUND TRUTH rows={len(gt):,}  S1 rows={len(s1):,}")
    out("  S1 ids missing from GT:", len(set(s1.entity_id) - set(gt.source1_entity_id)))
    n2 = gt["ids"].map(lambda l: sum(x.startswith("S2-") for x in l))
    n3 = gt["ids"].map(lambda l: sum(x.startswith("S3-") for x in l))
    tot = n2 + n3
    out("  singleton rate (no matches):", round((tot == 0).mean(), 4))
    out("  total matches per S1:", dict(sorted(Counter(tot.clip(upper=6)).items())), "(6 = 6+)")
    out("  S2 matches per S1:", dict(sorted(Counter(n2.clip(upper=6)).items())))
    out("  S3 matches per S1:", dict(sorted(Counter(n3.clip(upper=6)).items())))

    owner = Counter(x for l in gt["ids"] for x in l)
    out("  S2/S3 ids matched to >1 S1 (key check for 1-owner rule):",
        sum(v > 1 for v in owner.values()))
    all_ids = set(s2.entity_id) | set(s3.entity_id)
    out("  GT ids not found in S2/S3 files:", len(set(owner) - all_ids))
    out("  S2 rows never matched:", round(1 - len(set(s2.entity_id) & set(owner)) / len(s2), 4))
    out("  S3 rows never matched:", round(1 - len(set(s3.entity_id) & set(owner)) / len(s3), 4))

    # ---- properties of true pairs ----
    rec = pd.concat([s2, s3]).set_index("entity_id")
    s1i = s1.set_index("entity_id")
    pairs = [(a, b) for a, l in zip(gt.source1_entity_id, gt["ids"]) for b in l if b in rec.index]
    out(f"  positive pairs: {len(pairs):,}")
    same_c = sum(s1i.at[a, "country"] == rec.at[b, "country"] for a, b in pairs)
    out("  true pairs with same country:", round(same_c / max(len(pairs), 1), 4))
    both = same_pc = 0
    for a, b in pairs:
        p1, p2 = postal(s1i.at[a, "business_address"]), postal(rec.at[b, "business_address"])
        if p1 and p2:
            both += 1
            same_pc += p1 == p2
    out(f"  true pairs where both have postal code: {both:,}, of which equal: {same_pc / max(both, 1):.4f}")

    out("\n=== 25 RANDOM TRUE PAIRS")
    for a, b in random.sample(pairs, min(25, len(pairs))):
        out(f"  {a} | {s1i.at[a, 'business_name']} | {s1i.at[a, 'business_address']} | {s1i.at[a, 'country']}")
        out(f"  {b} | {rec.at[b, 'business_name']} | {rec.at[b, 'business_address']} | {rec.at[b, 'country']}\n")

    Path("eda_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---- sample: keep true matches intact, add random distractors ----
    keep_s1 = set(random.sample(list(gt.source1_entity_id), min(args.n_sample, len(gt))))
    gts = gt[gt.source1_entity_id.isin(keep_s1)]
    matched = {x for l in gts["ids"] for x in l}
    o = Path("sample"); o.mkdir(exist_ok=True)
    s1[s1.entity_id.isin(keep_s1)].to_csv(o / "train_source1.tsv", sep="\t", index=False)
    for k, df in [(2, s2), (3, s3)]:
        rest = df[~df.entity_id.isin(matched)]
        extra = rest.sample(min(args.n_distractors, len(rest)), random_state=42)
        pd.concat([df[df.entity_id.isin(matched)], extra]).to_csv(
            o / f"train_source{k}.tsv", sep="\t", index=False)
    gts[["source1_entity_id", "matched_entity_ids"]].to_csv(
        o / "train_ground_truth.tsv", sep="\t", index=False)
    print("\nWrote eda_report.txt and sample/*.tsv")


if __name__ == "__main__":
    main()
