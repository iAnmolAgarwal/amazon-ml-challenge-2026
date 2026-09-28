import argparse
import re
import subprocess
import sys

from common import ROOT, load_attempt, load_config
from harness_util import load_test, load_train
from er import raw_dir, validator_path


def next_sub_n(subdir):
    nums = [int(m.group(1)) for p in subdir.glob("*.tsv") if (m := re.match(r"(\d+)_", p.name))]
    return max(nums, default=0) + 1


def write_lists(path, ids, lists, col):
    with open(path, "w", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i, lst in zip(ids, lists):
            seen = list(dict.fromkeys(x for x in lst if x[:3] in ("S2-", "S3-")))
            f.write(f"{i}\t{','.join(seen)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt_id")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["pool"] = "test"
    mod = load_attempt(a.attempt_id)
    train, test = load_train(cfg), load_test(cfg)
    out = mod.fit_predict(train, test, cfg)
    matches = out["matches"] if isinstance(out, dict) else out
    cands = out.get("candidates", matches) if isinstance(out, dict) else matches
    subdir = ROOT / "submissions"
    subdir.mkdir(exist_ok=True)
    n = next_sub_n(subdir)
    mpath = subdir / f"{n:02d}_{a.attempt_id}.tsv"
    cpath = subdir / f"{n:02d}_{a.attempt_id}.candidates.tsv"
    ids = test[cfg["id_col"]].to_list()
    write_lists(mpath, ids, matches, "matched_entity_ids")
    write_lists(cpath, ids, cands, "candidate_entity_ids")
    print(f"wrote {mpath.relative_to(ROOT)} and {cpath.name}; {sum(1 for m in matches if m)} / {len(ids)} entities with matches")
    v = validator_path(cfg)
    if v is None:
        print("WARNING: utils/validate_submission.py not found; not validated")
        return
    r = subprocess.run([sys.executable, str(v), "--matching", str(mpath), "--candidate", str(cpath), "--test-dir", str(raw_dir(cfg) / "test")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("VALIDATION FAILED - do not upload")


if __name__ == "__main__":
    main()
