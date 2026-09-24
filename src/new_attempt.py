import argparse
import re
import shutil
from datetime import datetime, timezone

from common import ROOT

TEMPLATE = """# {id}

- parent: {parent}
- worker: {worker}
- created: {ts}

## Idea
(one paragraph: what changes vs parent and why it should help)

## Evidence from history
(which attempts/results.json support this; cite numbers)

## Why not a repeat
(which earlier attempts are closest and how this differs)

## Expected gain / risk
(expected CV delta in metric units and std; what could go wrong)

## Kill condition
(measurable: e.g. "no significant paired gain after 2 iterations" or "runtime > 30 min/fold")

## Result
(fill after evaluate: CV mean/std vs parent, what it shows, next idea)
"""


def next_id(attempts):
    nums = [int(m.group(1)) for p in attempts.iterdir() if (m := re.match(r"(\d{3})_", p.name))]
    return max(nums, default=-1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--worker", required=True, choices=["W1", "W2", "L"])
    ap.add_argument("--name", default="", help="short slug appended to the id")
    a = ap.parse_args()
    attempts = ROOT / "attempts"
    parent = attempts / a.parent
    if not (parent / "train.py").exists():
        raise SystemExit(f"parent {a.parent} has no train.py")
    slug = re.sub(r"[^a-z0-9]+", "-", a.name.lower()).strip("-")
    new_id = f"{next_id(attempts):03d}_{a.worker.lower()}" + (f"_{slug}" if slug else "")
    d = attempts / new_id
    d.mkdir()
    shutil.copy(parent / "train.py", d / "train.py")
    (d / "proposal.md").write_text(TEMPLATE.format(id=new_id, parent=a.parent, worker=a.worker, ts=datetime.now(timezone.utc).isoformat(timespec="minutes")))
    print(new_id)


if __name__ == "__main__":
    main()
