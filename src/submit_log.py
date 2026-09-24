import argparse
import csv
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, read_json, sha256

LOG = ROOT / "submissions" / "log.csv"
FIELDS = ["n", "timestamp_utc", "attempt_id", "file", "sha256", "cv_mean", "cv_std", "metric", "git_commit", "public_lb", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_csv", help="submissions/NN_<attempt>.csv")
    ap.add_argument("--lb", default="", help="public LB score once known")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-tag", action="store_true")
    a = ap.parse_args()
    p = Path(a.submission_csv)
    m = re.match(r"(\d+)_(.+)\.csv$", p.name)
    if not m:
        raise SystemExit("expected submissions/NN_<attempt_id>.csv")
    n, attempt_id = int(m.group(1)), m.group(2)
    res = read_json(ROOT / "attempts" / attempt_id / "results.json", {})
    row = {
        "n": n,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempt_id": attempt_id,
        "file": str(p.relative_to(ROOT)) if p.is_absolute() else str(p),
        "sha256": sha256(p),
        "cv_mean": res.get("mean", ""),
        "cv_std": res.get("std", ""),
        "metric": res.get("metric", ""),
        "git_commit": res.get("git_commit", ""),
        "public_lb": a.lb,
        "notes": a.notes,
    }
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    tag = f"sub-{n:02d}"
    if not a.no_tag:
        subprocess.run(["git", "tag", "-f", tag, "-m", f"{attempt_id} cv={row['cv_mean']}"], cwd=ROOT, check=True)
    print(f"logged #{n} {attempt_id} sha={row['sha256'][:12]} tag={tag if not a.no_tag else '-'}")


if __name__ == "__main__":
    main()
