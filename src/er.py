"""Locations of the organisers' files: the dataset folder and the validator, searched under config.yaml data_dir."""
from pathlib import Path


def raw_dir(cfg):
    base = Path(cfg["data_dir"])
    hits = sorted(base.rglob("train_source1.tsv"), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit(f"train_source1.tsv not found under {base}; unzip the dataset into data/")
    return hits[0].parent.parent


def validator_path(cfg):
    hits = sorted(Path(cfg["data_dir"]).rglob("validate_submission.py"), key=lambda p: len(p.parts))
    return hits[0] if hits else None
