import numpy as np

from submit_log import PER_DAY, TOTAL, budget, read_log, today_ist
from common import ROOT, paired_gain, read_json


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def best_cv():
    best = None
    for d in (ROOT / "attempts").iterdir():
        r = read_json(d / "results.json")
        if r and r.get("metric") == "f05_er" and (best is None or r["mean"] > best[1]):
            best = (d.name, r["mean"], r["std"])
    return best


def main():
    rows = read_log()
    used, left_today, left_total = budget(rows)
    print(f"IST day {today_ist()}: {used}/{PER_DAY} used today, {left_today} left today; {len(rows)}/{TOTAL} used overall, {left_total} left")
    days = {}
    for r in rows:
        days.setdefault(r["day_ist"], 0)
        days[r["day_ist"]] += 1
    if days:
        print("per day: " + ", ".join(f"{d}={c}" for d, c in sorted(days.items())))
    print(f"\n{'#':>3} {'attempt':<32} {'CV mean':>8} {'CV std':>7} {'public LB':>9} {'LB-CV':>7}  notes")
    for r in rows:
        cv, sd, lb = fnum(r["cv_mean"]), fnum(r["cv_std"]), fnum(r["public_lb"])
        gap = f"{lb - cv:+.4f}" if cv is not None and lb is not None else "-"
        print(f"{r['n']:>3} {r['attempt_id']:<32} {cv if cv is not None else '-':>8.8} {sd if sd is not None else '-':>7.7} {lb if lb is not None else 'pending':>9.9} {gap:>7}  {r['notes']}")
    pts = [(fnum(r["cv_mean"]), fnum(r["public_lb"])) for r in rows if fnum(r["cv_mean"]) is not None and fnum(r["public_lb"]) is not None]
    if len(pts) >= 2:
        cv, lb = np.array(pts).T
        gaps = lb - cv
        print(f"\nLB-CV gap: mean {gaps.mean():+.4f}, spread {gaps.std():.4f} over {len(pts)} scored submissions")
        if len(pts) >= 3:
            rc, rl = np.argsort(np.argsort(cv)), np.argsort(np.argsort(lb))
            print(f"rank agreement CV vs LB (Spearman): {np.corrcoef(rc, rl)[0, 1]:+.2f}  (below ~0.5 means CV is not tracking LB: investigate before trusting either)")
        order = np.argsort([int(r["n"]) for r in rows if fnum(r["cv_mean"]) is not None and fnum(r["public_lb"]) is not None])
        cvs, lbs = cv[order], lb[order]
        disagree = [i for i in range(1, len(cvs)) if np.sign(cvs[i] - cvs[i - 1]) != np.sign(lbs[i] - lbs[i - 1])]
        if disagree:
            print(f"CV and LB moved in opposite directions at {len(disagree)} step(s): check those submissions for leakage or country shift (France)")
    b = best_cv()
    if b:
        print(f"\nbest CV so far: {b[0]} {b[1]:.4f} ± {b[2]:.4f}")
        sub = [r for r in rows if fnum(r["cv_mean"]) is not None]
        if sub:
            top = max(sub, key=lambda r: fnum(r["cv_mean"]))
            g = paired_gain(read_json(ROOT / "attempts" / b[0] / "results.json"), read_json(ROOT / "attempts" / top["attempt_id"] / "results.json", {}))
            print(f"best submitted: {top['attempt_id']} {fnum(top['cv_mean']):.4f}")
            if b[0] != top["attempt_id"] and g and g["significant"]:
                print(f"-> {b[0]} shows a significant paired gain over it (Δ={g['mean']:+.4f}, wins {g['wins']}/{g['n']}): candidate for the next submission (audit first)")
        else:
            print("nothing submitted yet")


if __name__ == "__main__":
    main()
