"""Loading the challenge TSV files."""
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def load_tsv(path) -> pd.DataFrame:
    """Read a challenge TSV as strings (tab separated, no quote handling, no NaN)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_split(data_dir, split):
    """Return (s1, s2, s3) DataFrames for split 'train' or 'test'."""
    d = Path(data_dir) / split
    return tuple(load_tsv(d / f"{split}_source{k}.tsv")[COLS] for k in (1, 2, 3))


def load_ground_truth(data_dir) -> pd.DataFrame:
    """Ground truth as one row per true (source1_entity_id, rid) pair."""
    gt = load_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    pairs = gt.assign(rid=gt["matched_entity_ids"].str.split(",")).explode("rid")
    pairs = pairs[pairs["rid"].notna() & (pairs["rid"] != "")]
    return pairs[["source1_entity_id", "rid"]].reset_index(drop=True)


def countries_of(*frames):
    """All country labels present (open set, never hard-coded)."""
    return sorted(set().union(*[set(f["country"].unique()) for f in frames]))


def index_of(ids: pd.Series, lookup: pd.Series) -> np.ndarray:
    """Positions of `lookup` values inside `ids` (-1 if absent)."""
    return pd.Index(ids).get_indexer(lookup)
