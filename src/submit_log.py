import argparse
import csv
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import ROOT, read_json, sha256

LOG = ROOT / "submissions" / "log.csv"
FIELDS = ["n", "timestamp_utc", "day_ist", "attempt_id", "file", "sha256", "code_zip_sha256", "cv_mean", "cv_std", "metric", "git_commit", "public_lb", "notes"]
IST = timezone(timedelta(hours=5, minutes=30))
PER_DAY, TOTAL = 5, 15


def read_log():
    if not LOG.exists():
        return []
    with open(LOG, newline="") as f:
        return list(csv.DictReader(f))


def write_log(rows):
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def today_ist():
    return datetime.now(IST).date().isoformat()


def budget(rows):
    used_today = sum(1 for r in rows if r.get("day_ist") == today_ist())
    return used_today, PER_DAY - used_today, TOTAL - len(rows)


def main():
    ap = argparse.ArgumentParser(description="log an uploaded submission, or set its public LB score afterwards")
    ap.add_argument("submission_tsv", nargs="?", help="submissions/NN_<attempt>.tsv (the file uploaded)")
    ap.add_argument("--lb", default="", help="public LB score if already known")
    ap.add_argument("--set-lb", nargs=2, metavar=("N", "SCORE"), help="fill public LB for submission N")
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true", help="log even if the budget says none left")
    ap.add_argument("--no-tag", action="store_true")
    a = ap.parse_args()
    rows = read_log()
    if a.set_lb:
        n, score = a.set_lb
        hit = [r for r in rows if int(r["n"]) == int(n)]
        if not hit:
            raise SystemExit(f"submission {n} not in log")
        hit[0]["public_lb"] = score
        write_log(rows)
        print(f"#{n} {hit[0]['attempt_id']}: public LB = {score} (CV {hit[0]['cv_mean']})")
        return
    if not a.submission_tsv:
        raise SystemExit("give submissions/NN_<attempt>.tsv or --set-lb N SCORE")
    p = Path(a.submission_tsv)
    m = re.match(r"(\d+)_(.+?)\.tsv$", p.name)
    if not m or p.name.endswith(".candidates.tsv"):
        raise SystemExit("expected submissions/NN_<attempt_id>.tsv")
    n, attempt_id = int(m.group(1)), m.group(2)
    if any(int(r["n"]) == n for r in rows):
        raise SystemExit(f"submission {n} already logged")
    _, left_today, left_total = budget(rows)
    if (left_today <= 0 or left_total <= 0) and not a.force:
        raise SystemExit(f"budget exhausted: {left_today} left today, {left_total} total. Use --force only if the portal really accepted it.")
    res = read_json(ROOT / "attempts" / attempt_id / "results.json", {})
    z = p.with_name(p.stem + "_code.zip")
    rows.append({
        "n": n,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "day_ist": today_ist(),
        "attempt_id": attempt_id,
        "file": str(p.relative_to(ROOT)) if p.is_absolute() else str(p),
        "sha256": sha256(p),
        "code_zip_sha256": sha256(z) if z.exists() else "",
        "cv_mean": res.get("mean", ""),
        "cv_std": res.get("std", ""),
        "metric": res.get("metric", ""),
        "git_commit": res.get("git_commit", ""),
        "public_lb": a.lb,
        "notes": a.notes,
    })
    write_log(rows)
    tag = f"sub-{n:02d}"
    if not a.no_tag:
        subprocess.run(["git", "tag", "-f", tag, "-m", f"{attempt_id} cv={res.get('mean', '')}"], cwd=ROOT, check=True)
    used, left_today, left_total = budget(rows)
    print(f"logged #{n} {attempt_id} tag={tag if not a.no_tag else '-'}; today {used}/{PER_DAY} used, {left_total} of {TOTAL} left overall")
    if not a.lb:
        print(f"when the score shows: python src/submit_log.py --set-lb {n} <score>")


if __name__ == "__main__":
    main()
