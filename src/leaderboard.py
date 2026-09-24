import argparse
import re

from common import ROOT, read_json


def parent_of(d):
    p = d / "proposal.md"
    if not p.exists():
        return None
    m = re.search(r"^- parent:\s*(\S+)", p.read_text(), re.M)
    return None if not m or m.group(1) in ("none", "-", "null") else m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include attempts without results.json")
    a = ap.parse_args()
    dirs = sorted(p for p in (ROOT / "attempts").iterdir() if p.is_dir() and re.match(r"\d{3}_", p.name))
    info = {}
    for d in dirs:
        r = read_json(d / "results.json")
        if r is None and not a.all:
            continue
        info[d.name] = {"parent": parent_of(d), "res": r}
    children = {}
    for k, v in info.items():
        children.setdefault(v["parent"] if v["parent"] in info else None, []).append(k)
    def better(r, b):
        return r["mean"] > b["mean"] if r["higher_is_better"] else r["mean"] < b["mean"]

    best = None
    for k, v in info.items():
        if v["res"] and (best is None or better(v["res"], info[best]["res"])):
            best = k

    def row(k, depth):
        v, r = info[k], info[k]["res"]
        pad = "  " * depth + ("└─ " if depth else "")
        if not r:
            print(f"{pad}{k}  (no results)")
        else:
            pr = info.get(v["parent"], {}).get("res") if v["parent"] else None
            delta = ""
            if pr:
                sign = 1 if r["higher_is_better"] else -1
                dlt = sign * (r["mean"] - pr["mean"])
                ref_std = max(pr["std"], 1e-12)
                flag = "  ** >1 std" if dlt > ref_std else ("  xx worse" if dlt < -ref_std else "  ~ noise")
                delta = f"  Δ={dlt:+.5f}{flag}"
            star = "  <BEST>" if k == best else ""
            print(f"{pad}{k}  {r['metric']}={r['mean']:.5f}±{r['std']:.5f}  n={r['n_rows']}  {r['runtime_s']:.0f}s  @{r['git_commit']}{delta}{star}")
        for c in children.get(k, []):
            row(c, depth + 1)

    for root in children.get(None, []):
        row(root, 0)
    if best:
        print(f"\nbest: {best}  {info[best]['res']['metric']}={info[best]['res']['mean']:.5f}±{info[best]['res']['std']:.5f}")


if __name__ == "__main__":
    main()
