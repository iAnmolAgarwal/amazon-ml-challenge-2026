"""Data loading helpers. All reads use sep='\t', dtype=str, keep_default_na=False."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, engine="c")


def load_split(split):
    d = os.path.join(DATA, split)
    s = {k: read_tsv(os.path.join(d, f"{split}_source{k}.tsv")) for k in (1, 2, 3)}
    gt = read_tsv(os.path.join(d, f"{split}_ground_truth.tsv")) if split == "train" else None
    return s[1], s[2], s[3], gt


def gt_pairs(gt):
    """Explode ground truth to a (s1, rid) pair frame."""
    g = gt[gt.matched_entity_ids != ""].copy()
    g["rid"] = g.matched_entity_ids.str.split(",")
    g = g.explode("rid")[["source1_entity_id", "rid"]].rename(columns={"source1_entity_id": "s1"})
    return g.reset_index(drop=True)
