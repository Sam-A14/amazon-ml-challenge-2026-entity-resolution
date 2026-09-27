import pandas as pd
def rd(p, cols):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=cols)
for k in (1, 2, 3):
    tr = rd(f"dataset/train/train_source{k}.tsv", ["entity_id"])["entity_id"]
    te = rd(f"dataset/test/test_source{k}.tsv", ["entity_id"])["entity_id"]
    print(f"S{k}: test IDs also in train = {te.isin(set(tr)).mean():.4f}  ({len(te):,} test rows)")
    del tr, te
tr = rd("dataset/train/train_source1.tsv", ["business_name", "business_address"])
te = rd("dataset/test/test_source1.tsv", ["business_name", "business_address"])
key = lambda d: (d["business_name"].str.lower().str.strip() + "|" + d["business_address"].str.lower().str.strip())
print("S1: test name+address exactly also in train =", round(key(te).isin(set(key(tr))).mean(), 4))
print("S1: test name also in train =", round(te["business_name"].str.lower().str.strip().isin(set(tr["business_name"].str.lower().str.strip())).mean(), 4))
