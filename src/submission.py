"""Writing matching_results.tsv and candidate_pairs.tsv in the exact required format."""
from pathlib import Path

import pandas as pd


def _write(path, col2, all_s1_ids, s1_ids, r_ids):
    """One row per Source 1 entity (in input order), comma-joined unique IDs, no quoting."""
    df = pd.DataFrame({"s1": s1_ids, "r": r_ids}).drop_duplicates()
    lists = df.groupby("s1", sort=False)["r"].agg(lambda x: ",".join(sorted(x)))
    lists = lists.reindex(pd.Index(all_s1_ids)).fillna("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col2}\n")
        for s1, ids in zip(lists.index, lists.to_numpy()):
            f.write(f"{s1}\t{ids}\n")


def write_submission(out_dir, all_s1_ids, cand_s1, cand_r, match_s1, match_r):
    """Write both files. Matches must be a subset of candidates (asserted)."""
    cand = set(zip(cand_s1, cand_r))
    assert all(p in cand for p in zip(match_s1, match_r)), "match outside candidate set"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "candidate_pairs.tsv", "candidate_entity_ids", all_s1_ids, cand_s1, cand_r)
    _write(out / "matching_results.tsv", "matched_entity_ids", all_s1_ids, match_s1, match_r)
