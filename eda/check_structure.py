import numpy as np, pandas as pd
d = "dataset/train/"
rd = lambda f: pd.read_csv(d + f, sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=[0])["entity_id"]
s1, s2, s3 = rd("train_source1.tsv"), rd("train_source2.tsv"), rd("train_source3.tsv")
gt = pd.read_csv(d + "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
p = gt.assign(r=gt["matched_entity_ids"].str.split(",")).explode("r")
p = p[p["r"].notna() & (p["r"] != "")].sample(500000, random_state=0)
i1 = pd.Index(s1)
for name, s in [("S2", s2), ("S3", s3)]:
    q = p[p["r"].str.startswith(name)]
    a = pd.Series(i1.get_indexer(q["source1_entity_id"]))
    b = pd.Series(pd.Index(s).get_indexer(q["r"]))
    print(name, "row-order correlation (S1 row vs match row):", round(a.corr(b, method="spearman"), 4))
    na = q["source1_entity_id"].str[3:].astype(np.int64).reset_index(drop=True)
    nb = q["r"].str[3:].astype(np.int64).reset_index(drop=True)
    print(name, "ID-number correlation:", round(na.corr(nb, method="spearman"), 4))
    g = pd.DataFrame({"a": a, "b": b}).groupby("a")["b"].agg(["min", "max", "size"])
    g = g[g["size"] > 1]
    print(name, "median row distance between records of the same business:",
          int((g["max"] - g["min"]).median()), "| random would be about", len(s) // 3)
