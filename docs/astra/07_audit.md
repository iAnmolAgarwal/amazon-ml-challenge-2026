# Gate 7 audit: Astra "submission compliance" answer (07_submission_compliance_feedback.md)

Auditor: independent (Opus), 2026-09-25. I read the code at HEAD 03e1775, both uploaded code zips, the official README and validator, and the EC2 launch scripts for uploads 1 and 2 (scratchpad `a_submit1.sh`, `a_pred2.sh`). I tested runtime behaviour on the Mac with synthetic data only: scratchpad `audit07/{t_polars,t_big,t_mo,t_lgb,mk}.py`, plus a synthetic end-to-end root `audit07/root/` holding the current and patched `er_predict`/`er_package`. EC2 was not touched. Nothing under src/ or attempts/ was edited. The patches below are proposals and need the human's approval.

## 0. Bottom line

- Astra's verdict ("fix required, no invalid scored TSV") is right. **None of the findings changes a live-leaderboard score or causes a portal rejection.** All of them matter for the final package review (after 27 Sep 23:59 IST). Finding 3 also affects the safety of every future upload.
- Finding 3 (the validator gate misses warnings) is the most concrete. I reproduced it: with a candidate file that contains none of the matches, the official validator prints a WARNING, then `PASS`, and exits 0.
- Finding 4 is overstated for the pinned versions. On polars 1.44.2 and LightGBM 4.7.0, same-order training is bit-identical and the joins and `unique()` keep a stable order. The one real nondeterminism is the byte order of `candidate_pairs.tsv`, which changes between identical runs; the sets are identical. Astra's fix "sort training pairs after joins" should be **rejected**. It would change the model relative to the uploads. `maintain_order="left"` pins today's behaviour without changing any output.
- Finding 1 is true of the uploaded zips, but the uploads themselves were produced with `--train-frac 0.2` (the EC2 scripts show it). Nothing can be done to the uploaded zips 01 and 02, and nothing needs to be. Fix the default in er_predict for the future.
- A latent bug Astra missed: `er_predict` ignores an attempt's `PARAMS` and `ROUNDS` (`er_eval` honours them). No current attempt sets them, so there is no impact yet.

## 1. Findings table

| # | Astra finding | Verdict | Evidence | Impact | Priority |
|---|---|---|---|---|---|
| 1 | README in the uploaded zips omits `--train-frac`; the predictor default is 0.3 and the attempt says 0.2 | **VERIFIED** (zips); resolved for the TSVs | zip 01 README.md:21 and zip 02 README.md:23 have no `--train-frac`. The packaged er_predict.py:41 has `default=0.3`. Both attempts have `TRAIN_FRAC = 0.2`. EC2 scripts: `er_predict.py 001_baseline --train-frac 0.2` and `er_predict.py 002_namefreq --train-frac 0.2`, so the uploaded TSVs used 0.2. At HEAD: the README template has `--train-frac {frac}` (er_package.py:62), but the er_predict default is still 0.3 (er_predict.py:41) and the er_package fallback is 0.3 (er_package.py:123), while `M.TRAIN_FRAC` is 0.2 (er_model.py:16) | live LB: none. Final review: only if the final README or default is wrong. Uploaded zips 01/02: no action possible or needed (see §3) | P2 |
| 1b | (new) er_predict ignores an attempt's PARAMS/ROUNDS | **VERIFIED** | er_predict.py:69 calls `get("fit")(X, yv, cfg)`, which resolves to `M.fit` and uses module `PARAMS`/`ROUNDS` (er_model.py:52). er_eval.py:68 passes `get("PARAMS")`, `get("ROUNDS")` | none today (no attempt sets them). A CV/test mismatch the moment one does | P2 (goes in the same patch) |
| 2 | `--final` overwrites NN_code.zip, silently omits the doc, accepts an unfilled template, copies today's workspace | **VERIFIED** | er_package.py:113 names the zip `NN_<attempt>_code.zip` even with --final. :115 opens it with "w" before :126-127 add the outputs, so a missing `.candidates.tsv` (none exist on the Mac) raises mid-write and leaves a truncated zip. :129 skips the doc silently. docs/Documentation_template.md is the unfilled organiser template (lines 3, 30, 43 and others; identical to data/student_resource/Documentation_template.md). :117 packages the current workspace, with no link to the run that wrote the TSV | final review: high (wrong zip name, possible missing doc or candidates, or source that does not match the outputs). Live LB: none | **P1** |
| 3 | Validation gate accepts missing candidates or unchecked ids | **VERIFIED** (reproduced) | er_package.py:92 has no `--check-ids` and :94 checks only the return code. validate_submission.py:83 (missing S2/S3 file gives a warning and returns None), :252 (missing candidate file gives a warning), :266 (containment gives a warning). All exit 0 with `PASS`. Reproduced on synthetic data: 6,265 matched S1 absent from the candidates, then "PASS ... Safe to submit", exit 0. The uploads were validated by hand on EC2 with `--check-ids` (per the brief), so the only gap there is whether anyone read the WARNING lines | live LB: an id-existence problem only lowers the score. Portal rejection: no (the format rules are errors, not warnings). Final review: a containment warning would look like a pipeline bug | **P1** |
| 4a | Joins and `unique` have no ordering guarantee, so bagging on reordered rows gives a different model | **PARTLY** | The polars 1.44.2 docs for `join(maintain_order=None)` say "might differ ... between runs". Measured on 1.44.2: `Series.unique()` on UInt32 returns sorted, identical output (2M and 20M rows, unsorted input). The left join keeps left order (2M and 20M rows, 8 threads). `maintain_order="left"` gives the identical frame, and `np.sort(unique)` is a no-op. LightGBM 4.7.0 with the repo PARAMS (bagging 0.8, feature_fraction 0.8, 8 threads), 400k x 30 synthetic: 3 same-order runs are bit-identical. `force_col_wise` and `deterministic=True` give the same hash. A **row permutation changes the model** (mean abs dp 0.014, 1,509/100k flips at 0.5). So the risk is real only if polars changes behaviour. The pinned requirements already cover that; the proposed diff makes it explicit at zero output change | live LB: none. Final review: low (a reviewer rerun on the pinned env reproduces the model) | P2 |
| 4b | Candidate serialisation follows unordered grouping | **VERIFIED** | er_predict.py:84, `concat(...).unique()` then `group_by("entity_id")` with no maintain_order. Two identical runs on synthetic data gave different candidate sha256 and identical sets. The matching TSV was byte-identical across runs | not scored. Final review: checksums of a reproduced candidate file will not match | P2 |
| 4c | Astra's fix "sort training pairs by (s1_idx,pool_idx)", "deterministic histogram config" | **REJECT as stated** | Sorting changes training row order relative to uploads 1-4, which gives a different model, so the uploaded TSVs would no longer be reproducible. `deterministic=True`/`force_col_wise` made no difference with fixed `num_threads`. Use `maintain_order="left"` + `np.sort(unique)` instead (tested output-neutral) | - | - |
| 5 | CV-LB gap is partly country composition, not all France | **VERIFIED (arithmetic)**, framing partly wrong | 002 results.json: us 0.971258, india 0.944383, test-weighted 0.956398, overall 0.960502. Gap 0.9605-0.949 = 0.0115 = 0.0041 (mix: 0.9605-0.9564) + 0.15x(0.9564-F). F = (0.949-0.38x0.9713-0.47x0.9444)/0.15 = **0.9069**. The implied-France figure in STATUS already uses the country-weighted CV, so the composition effect was already priced in; only the phrase "gap attributed to France" is loose. Two uncertainties to add: the LB is shown to 3 dp (±0.0005, so F ±0.0033), and the public subset's country mix is unknown. Upload 1 gives F = 0.883 by the same formula, which fits the France-gated normalization + feat2 helping France (+0.024) | none on the score. Documentation must label F as unvalidated | P3 (checks in §4) |
| D | Documentation advice | **Agree, trimmed** | see §5 | final review | P1 (before 27 Sep) |
| x1 | (new) log.csv `git_commit` = `nogit` for both uploads | VERIFIED | submissions/log.csv:2-3. results.json is written on EC2, where there is no git repo (common.py:73-77) | provenance only | fixed by the manifest (src sha256) |
| x2 | (new) upload 1 zip includes `attempts/001_baseline/proposal.md` (internal notes: "worker: L", docs/strategy.md) | VERIFIED | `unzip -l` of zip 01. No AI/tool names found in either zip (grepped) | none (the zip is not re-uploadable; the current packager only takes .py/.json) | none |
| x3 | (new) er_predict docstring points to `src/er_validate.py`, which does not exist | VERIFIED | er_predict.py:8 | cosmetic | with P2 |

## 2. Proposed patches (tested on a synthetic root, not applied)

Tests run in `audit07/root` (synthetic data: 20k train S1, 10k test S1, 2 extra-feature side-car):
- Patched vs current er_predict: **matching TSV byte-identical** (sha e4281e30...), candidate **sets identical**, S1 row order identical. The candidate file is now byte-stable across runs (the current one is not). The default fraction now comes from the attempt (0.5 with no flag), and the manifest is written.
- Patched er_package: clean case gives PASS and writes `.validation.txt`. A match outside the candidates gives FAIL. A missing `.candidates.tsv` gives FAIL before any zip is opened. A missing test_source3 gives FAIL. `--final` with template placeholders gives FAIL. `--final` on a TSV with no manifest gives FAIL. `--final` after editing er_block.py gives FAIL ("er_block.py"). `--final --team X` with a filled doc builds `X_submission.zip` with output/ and Documentation_template.md, and leaves no .tmp behind.

### 2a. src/er_predict.py (P2; output-neutral for the matching file and model)
Fixes 1, 1b, 4a, 4b, x1, and the manifest. Apply after the in-flight 004 predict finishes, then sync to EC2.

```diff
--- a/src/er_predict.py
+++ b/src/er_predict.py
@@ -1,22 +1,27 @@
 """Test inference: train on train pairs, score test pairs, decide, write the two submission TSVs.
 
-usage: python src/er_predict.py <attempt_id> [--tag ""] [--train-frac 0.3]
+usage: python src/er_predict.py <attempt_id> [--tag ""] [--train-frac F]   (default: the attempt's TRAIN_FRAC)
 
-Writes submissions/NN_<attempt_id>.tsv (matching_results format) and
-submissions/NN_<attempt_id>.candidates.tsv (candidate_pairs format: every candidate the model
-scored, union the matches). One row per test S1 in test_source1 order; ids S2-/S3- only.
+Writes submissions/NN_<attempt_id>.tsv (matching_results format),
+submissions/NN_<attempt_id>.candidates.tsv (candidate_pairs format: exactly the pairs the model scored)
+and submissions/NN_<attempt_id>.json (run manifest: fraction, sample hash, features, params, source hashes).
+One row per test S1 in test_source1 order; ids S2-/S3- only, in a fixed order.
 Run src/er_validate.py on the Mac afterwards (it needs the raw test TSVs).
 """
 import argparse
+import hashlib
+import json
 import re
+import sys
 import time
 from pathlib import Path
 
+import lightgbm as lgb
 import numpy as np
 import polars as pl
 
 import er_model as M
-from common import ROOT, load_attempt, load_config
+from common import ROOT, git_commit, load_attempt, load_config, sha256
 
 
 def next_n(d):
@@ -38,11 +43,12 @@
     ap = argparse.ArgumentParser()
     ap.add_argument("attempt_id")
     ap.add_argument("--tag", default="")
-    ap.add_argument("--train-frac", type=float, default=0.3)
+    ap.add_argument("--train-frac", type=float)
     a = ap.parse_args()
     cfg = load_config()
     mod = load_attempt(a.attempt_id)
     get = lambda name: getattr(mod, name, getattr(M, name))
+    frac = a.train_frac or get("TRAIN_FRAC")
     cache = Path(cfg["cache_dir"])
     t0 = time.time()
     s1 = M.load_s1(cfg, "train")
@@ -52,21 +58,21 @@
     def load_feat(split):
         f = pl.read_parquet(cache / "feat" / f"{split}_feat{a.tag}.parquet")
         for extra in extras:
-            f = f.join(pl.read_parquet(cache / "feat" / f"{split}_{extra}{a.tag}.parquet"), on=["s1_idx", "pool_idx"], how="left")
+            f = f.join(pl.read_parquet(cache / "feat" / f"{split}_{extra}{a.tag}.parquet"), on=["s1_idx", "pool_idx"], how="left", maintain_order="left")
         return f
 
     ftr = load_feat("train")
     cols = [c for c in M.feature_cols(ftr, getattr(mod, "FEATURES", None)) if c not in getattr(mod, "EXCLUDE", [])]
     rng = np.random.default_rng(cfg["seed"])
-    ents = ftr["s1_idx"].unique().to_numpy()
-    keep = rng.choice(ents, size=int(len(ents) * a.train_frac), replace=False)
-    tr = ftr.filter(pl.col("s1_idx").is_in(keep)) if a.train_frac < 1 else ftr
+    ents = np.sort(ftr["s1_idx"].unique().to_numpy())
+    keep = rng.choice(ents, size=int(len(ents) * frac), replace=False)
+    tr = ftr.filter(pl.col("s1_idx").is_in(keep)) if frac < 1 else ftr
     del ftr
-    tr = tr.join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_idx", "pool_idx"], how="left").with_columns(pl.col("y").fill_null(0))
+    tr = tr.join(gold.with_columns(y=pl.lit(1, pl.Int8)), on=["s1_idx", "pool_idx"], how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
     yv = tr["y"].to_numpy()
     X = M.to_matrix(tr.select(cols), cols)
     del tr
-    model = get("fit")(X, yv, cfg)
+    model = get("fit")(X, yv, cfg) if hasattr(mod, "fit") else M.fit(X, yv, cfg, get("PARAMS"), get("ROUNDS"))
     print(f"trained on {len(yv):,} pairs ({int(yv.sum()):,} pos) from {len(keep):,} entities in {time.time() - t0:.0f}s", flush=True)
     del X
     fte = load_feat("test")
@@ -79,9 +85,10 @@
     pred = get("decide")(pairs, cfg)
     ts1 = pl.read_parquet(cache / "norm" / "test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.UInt32))
     tpool = pl.read_parquet(cache / "norm" / "test_pool.parquet", columns=["entity_id"]).with_row_index("pool_idx").with_columns(pl.col("pool_idx").cast(pl.UInt32))
-    name = lambda df: df.join(ts1, on="s1_idx").join(tpool.rename({"entity_id": "pid"}), on="pool_idx")
-    m = name(pred).sort(["s1_idx", "prob"], descending=[False, True]).group_by("entity_id", maintain_order=True).agg(pl.col("pid").alias("ids"))
-    c = name(pl.concat([pairs.select("s1_idx", "pool_idx"), pred.select("s1_idx", "pool_idx")]).unique()).group_by("entity_id").agg(pl.col("pid").alias("ids"))
+    name = lambda df: df.join(ts1, on="s1_idx", maintain_order="left").join(tpool.rename({"entity_id": "pid"}), on="pool_idx", maintain_order="left")
+    m = name(pred.sort(["s1_idx", "prob", "pool_idx"], descending=[False, True, False])).group_by("entity_id", maintain_order=True).agg(pl.col("pid").alias("ids"))
+    assert pred.join(pairs, on=["s1_idx", "pool_idx"], how="anti").is_empty(), "a match is not a scored candidate"
+    c = name(pairs.select("s1_idx", "pool_idx").sort("s1_idx", "pool_idx")).group_by("entity_id", maintain_order=True).agg(pl.col("pid").alias("ids"))
     out = ROOT / "submissions"
     out.mkdir(exist_ok=True)
     n = next_n(out)
@@ -89,6 +96,12 @@
     ids = ts1["entity_id"].to_list()
     write_lists(mp, ids, m, "matched_entity_ids")
     write_lists(cp, ids, c, "candidate_entity_ids")
+    src = {p.name: sha256(p) for p in sorted((ROOT / "src").glob("*.py"))} | {"train.py": sha256(ROOT / "attempts" / a.attempt_id / "train.py")}
+    man = dict(attempt=a.attempt_id, tag=a.tag, train_frac=frac, n_entities=len(keep), sample_sha256=hashlib.sha256(np.sort(keep).tobytes()).hexdigest(), n_train_pairs=len(yv), n_pos=int(yv.sum()),
+               features=cols, params=get("PARAMS"), rounds=get("ROUNDS"), custom_fit=hasattr(mod, "fit"), git_commit=git_commit(), src_sha256=src,
+               versions=dict(python=sys.version.split()[0], polars=pl.__version__, numpy=np.__version__, lightgbm=lgb.__version__),
+               matching_sha256=sha256(mp), candidates_sha256=sha256(cp), n_matched_pairs=len(pred), n_candidate_pairs=len(pairs), runtime_s=round(time.time() - t0))
+    mp.with_suffix(".json").write_text(json.dumps(man, indent=1))
     k = pred.group_by("s1_idx").len()
     print(f"wrote {mp.name} and {cp.name}: {len(ids):,} S1, {len(k):,} with matches, {len(pred):,} matched pairs (mean {len(pred) / len(ids):.2f}/S1), {len(pairs):,} candidates, {time.time() - t0:.0f}s")
 
```
Notes: `frac < 1` for 004 (TRAIN_FRAC 1.0) still trains on all rows. `np.sort(keep)` makes the sample hash independent of draw order. The union with `pred` is replaced by an assertion, which answers Astra's point about not repairing containment by unioning. The x3 docstring line (`er_validate.py`) can be dropped in the same edit.

### 2b. src/er_package.py (P1)
Fixes 2, 3 and the 0.3 fallback. On EC2 (where the raw test TSVs and 64 GB are), run `python src/er_package.py submissions/NN_x.tsv --validate-only [--test-dir valtest]`. That writes `NN_x.validation.txt`. Copy `.tsv`, `.candidates.tsv`, `.json` and `.validation.txt` to the Mac and check the sha256 values. Then run `--skip-validate` (per upload) or `--skip-validate --final --team <team>` (final). On the Mac, `--check-ids` with the ~1.7M-row candidate file would not fit in 8 GB, so validation stays on EC2.

```diff
--- a/src/er_package.py
+++ b/src/er_package.py
@@ -1,15 +1,17 @@
 """Validate a submission with the official validator and build the code zip required by the portal.
 
-usage: python src/er_package.py submissions/NN_<attempt_id>.tsv [--final]
+usage: python src/er_package.py submissions/NN_<attempt_id>.tsv [--validate-only] [--test-dir D] [--skip-validate] [--final --team NAME]
 
-1. Runs the organisers' utils/validate_submission.py on the matching TSV and its .candidates.tsv
-   against the raw test TSVs. Stops on FAIL.
+1. Runs the organisers' utils/validate_submission.py --check-ids on the matching TSV and its .candidates.tsv
+   against the raw test TSVs. Stops on FAIL or on any WARNING (skipped id check, missing candidates, match
+   outside candidates); on success writes NN_<attempt_id>.validation.txt (both sha256 + validator output).
 2. Builds submissions/NN_<attempt_id>_code.zip:
      code/business_entity_resolution/src/          pipeline source (src/ + the attempt's train.py)
      code/business_entity_resolution/README.md     exact end-to-end reproduction steps
      code/business_entity_resolution/requirements.txt  pinned versions
-   With --final it also adds output/matching_results.tsv, output/candidate_pairs.tsv and
-   Documentation_template.md (docs/Documentation_template.md) at the zip root.
+   With --final --team T it builds submissions/T_submission.zip instead, adding output/matching_results.tsv,
+   output/candidate_pairs.tsv and the filled docs/Documentation_template.md. It requires a clean .validation.txt
+   for these exact files, the run manifest NN_<attempt_id>.json with matching source hashes, and no template placeholders.
 Never includes data, caches, credentials or git metadata.
 """
 import argparse
@@ -19,7 +21,8 @@
 import zipfile
 from pathlib import Path
 
-from common import ROOT, load_attempt, load_config
+import er_model as M
+from common import ROOT, load_attempt, load_config, read_json, sha256
 from er import raw_dir, validator_path
 
 REQUIREMENTS = """polars==1.44.2
@@ -82,24 +85,44 @@
         print(f)
 '''
 
+DOC = ROOT / "docs" / "Documentation_template.md"
+PLACEHOLDERS = ("[Your Team Name]", "[List all team members]", "[Date]", "[e.g.,", "[total]", "[]", "[brief description]", "[your best validation score]", "[Brief description", "[Blocking + Classifier", "*Provide a brief", "*Describe how", "*Key insights", "*Outline your", "*Summarize your", "*Include any", "*Your complete")
 SRC_FILES = ["common.py", "er.py", "er_norm.py", "er_block.py", "er_feat.py", "er_feat2.py", "er_feat3.py", "er_model.py", "er_eval.py", "er_predict.py", "er_package.py"]
 
 
-def validate(mt, ct, cfg):
+def validate(mt, ct, cfg, test_dir):
     v = validator_path(cfg)
-    if v is None:
-        raise SystemExit("official validator not found under data/")
-    r = subprocess.run([sys.executable, str(v), "--matching", str(mt), "--candidate", str(ct), "--test-dir", str(raw_dir(cfg) / "test")], capture_output=True, text=True)
+    need = [mt, ct] + [test_dir / f"test_source{i}.tsv" for i in (1, 2, 3)]
+    if v is None or not all(p.exists() for p in need):
+        raise SystemExit(f"validator or inputs missing: {[str(p) for p in need if not p.exists()]}")
+    r = subprocess.run([sys.executable, str(v), "--matching", str(mt), "--candidate", str(ct), "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True)
     print(r.stdout[-4000:], r.stderr[-2000:])
-    if r.returncode != 0:
-        raise SystemExit("VALIDATION FAILED - do not upload")
+    if r.returncode != 0 or "WARNING" in r.stdout or "PASS" not in r.stdout:
+        raise SystemExit("VALIDATION FAILED or WARNED - do not upload")
+    mt.with_suffix(".validation.txt").write_text(f"{sha256(mt)}  {mt.name}\n{sha256(ct)}  {ct.name}\n{r.stdout}")
 
 
+def final_checks(mt, ct, man, attempt):
+    rec = mt.with_suffix(".validation.txt")
+    txt = rec.read_text() if rec.exists() else ""
+    if not ct.exists() or txt.split("\n")[:2] != [f"{sha256(mt)}  {mt.name}", f"{sha256(ct)}  {ct.name}"] or "WARNING" in txt or "PASS" not in txt:
+        raise SystemExit(f"--final needs {ct.name} and a clean {rec.name} for these exact files")
+    if not DOC.exists() or any(p in DOC.read_text() for p in PLACEHOLDERS):
+        raise SystemExit("fill docs/Documentation_template.md first")
+    now = {f: sha256(ROOT / "src" / f) for f in SRC_FILES if f != "er_package.py"} | {"train.py": sha256(ROOT / "attempts" / attempt / "train.py")}
+    stale = [f for f in now if man.get("src_sha256", {}).get(f) != now[f]]
+    if not man or stale or man.get("matching_sha256") != sha256(mt) or man.get("candidates_sha256") != sha256(ct):
+        raise SystemExit(f"outputs or source differ from the run manifest {mt.with_suffix('.json').name}: {stale or 'missing/mismatched manifest'}")
+
+
 def main():
     ap = argparse.ArgumentParser()
     ap.add_argument("matching_tsv")
     ap.add_argument("--final", action="store_true")
     ap.add_argument("--skip-validate", action="store_true")
+    ap.add_argument("--validate-only", action="store_true")
+    ap.add_argument("--test-dir")
+    ap.add_argument("--team")
     a = ap.parse_args()
     cfg = load_config()
     mt = Path(a.matching_tsv).resolve()
@@ -109,10 +132,18 @@
     attempt = m.group(2)
     ct = mt.with_name(mt.stem + ".candidates.tsv")
     if not a.skip_validate:
-        validate(mt, ct, cfg)
-    z = mt.with_name(mt.stem + "_code.zip")
+        validate(mt, ct, cfg, Path(a.test_dir) if a.test_dir else raw_dir(cfg) / "test")
+    if a.validate_only:
+        return
+    man = read_json(mt.with_suffix(".json"), {})
+    if a.final:
+        if not a.team:
+            raise SystemExit("--final needs --team")
+        final_checks(mt, ct, man, attempt)
+    z = mt.with_name(f"{a.team}_submission.zip" if a.final else mt.stem + "_code.zip")
+    tmp = z.with_suffix(".zip.tmp")
     base = "code/business_entity_resolution/"
-    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
+    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
         for f in SRC_FILES:
             zf.write(ROOT / "src" / f, base + "src/" + f)
         zf.writestr(base + "src/prepare_raw.py", PREP)
@@ -120,16 +151,17 @@
         for f in (ROOT / "attempts" / attempt).glob("*"):
             if f.suffix in (".py", ".json") and not f.name.startswith("diag"):
                 zf.write(f, base + f"attempts/{attempt}/{f.name}")
-        zf.writestr(base + "README.md", README.format(attempt=attempt, frac=getattr(load_attempt(attempt), "TRAIN_FRAC", 0.3)))
+        zf.writestr(base + "README.md", README.format(attempt=attempt, frac=man.get("train_frac", getattr(load_attempt(attempt), "TRAIN_FRAC", M.TRAIN_FRAC))))
         zf.writestr(base + "requirements.txt", REQUIREMENTS)
         if a.final:
             zf.write(mt, "output/matching_results.tsv")
             zf.write(ct, "output/candidate_pairs.tsv")
-            doc = ROOT / "docs" / "Documentation_template.md"
-            if doc.exists():
-                zf.write(doc, "Documentation_template.md")
-    names = zipfile.ZipFile(z).namelist()
-    assert not any(re.search(r"(\.pem|credentials|\.env|/cache/|/data/)", n) for n in names), "forbidden file in zip"
+            zf.write(DOC, "Documentation_template.md")
+    names = zipfile.ZipFile(tmp).namelist()
+    if any(re.search(r"(\.pem|credentials|\.env|/cache/|/data/)", n) for n in names):
+        tmp.unlink()
+        raise SystemExit("forbidden file in zip")
+    tmp.replace(z)
     print(f"built {z.name}: {len(names)} files, {z.stat().st_size / 1e3:.0f} KB")
 
 
```
Notes: `er_package.py` is excluded from the stale-source check (editing the packager does not change outputs). The final README fraction comes from the run manifest. The final package needs its TSVs from a patched-er_predict run. See §3, step 4.

## 3. What to do about uploads 01/02 and the final package

1. **Uploads 01/02: no action.** The README says the final `<team_name>_submission.zip` is what gets reproduced and "reviewed in detail". Nothing says the per-upload code zips are audited, and they cannot be replaced. Optionally, add one line to the Documentation appendix ("Submission history"): uploads 1-2 were produced by `er_predict.py <attempt> --train-frac 0.2`; the packaged README of those two zips omitted the flag.
2. **Uploads 1-3 validator logs:** if the EC2 validator stdout was saved, grep it for `WARNING`. If it was not saved, rerun `validate_submission.py --check-ids` for 01/02/03 on A when A is idle (minutes, a few GB) and confirm there are no WARNING lines. Cheap, and it closes finding 3 retroactively.
3. **Upload 4** (running now, pre-patch code): package it with `--skip-validate` as before (per-upload zip). Its README will say `--train-frac 1.0` via TRAIN_FRAC. Fine.
4. **Final package:** once the final attempt is chosen, rerun `er_predict.py <final>` once on EC2 with the patched code, from the pipeline caches built by the packaged source. Assert that its matching **sets** equal the uploaded TSV (the patch is output-neutral, so they should be byte-equal). Then `--validate-only`, copy the four files, fill the doc, and build `--final --team <team>`. This is the "clean reproduction" Astra asks for, at the cost of one predict run. A full raw-to-output rerun (norm, block, feat x3) is better but costs hours on EC2; do it only if time allows on 27 Sep.

## 4. Finding 5: cheap checks (run on EC2 A, label-free on test; ranked by value per minute)

1. **Fraction parity (1 min):** `grep "trained on" ~/predict_00{1,2}.log`. The entity count must equal int(0.2 x n_train_S1_with_candidates). That settles finding 1 definitively.
2. **Predicted-k profile by country (5 min):** from `submissions/02_002_namefreq.tsv` joined to `cache/norm/test_s1.parquet` (country_n): empty-list rate, mean k, share k>=2, S2/S3 split, per country. Compare with `decide()` on `cache/oof/002_namefreq.parquet` per country. US/India should match within about 1 pt; France is the only unknown. Gate 6 already used the sub-02 k profile, so extend that script.
3. **Candidate count by country (5 min):** `pl.scan_parquet("cache/feat/test_feat.parquet").group_by("s1_idx").len()` joined to country, giving the zero-candidate rate and the 10/50/90th-percentile candidate counts. Compare with train. A France zero-candidate rate or a low candidate count directly caps France recall.
4. **Feature health by country (10 min):** for each column of `test_feat`/`test_feat2` (and feat3), the null rate, the non-finite count and the mean, by country, against train (scan, select, group_by). This catches France-specific extraction failures (house number, postal code) that would make features silently missing.
5. **ID round-trip (5 min):** sample 30 matched pairs from the uploaded TSV and print the raw S1 and the matched S2/S3 rows from the test TSVs. Eyeball the pairs, including 10 French ones.
6. Optional: test probabilities are not saved. If a later diagnosis needs max-prob per S1 by country, add `pairs.write_parquet(cache/"oof"/f"test_{attempt}.parquet")` to er_predict (~1 GB). Not included in the patch.

Not worth doing: forcing the test rates to equal the train rates (Astra agrees), or any per-country test tuning.

## 5. Documentation_template.md: what to put in (agreeing with Astra, trimmed)

Fill every template section, and keep the headings. Add, briefly:
- **Data and fair play:** only the organisers' train/test TSVs. Labels are used only from train_ground_truth. Test statistics used are label-free (name-frequency counts, and TF-IDF fitted per split). Hand-written normalization rules (list them, including the France-only street-type, st->saint and BP/CEDEX rules, applied to rows whose country label is France; country is an open set, and blocking runs per country value found). No external data, APIs, geocoding, pretrained weights or LLMs in the pipeline.
- **Producing run:** attempt id, effective train fraction, sample sha256, seed 42, feature list, LightGBM params and rounds, and source sha256s, all taken from the run manifest. Also the sha256 of both output files.
- **Blocking audit:** TF-IDF settings, top-40, exact keys, cap and oversized-block rule, dynamic countries. candidate_pairs.tsv = exactly the scored pairs (asserted), with total pairs and candidates per S1 by country. The retrieval oracle on train (0.994 US / 0.979 India from the proposal) is a useful recall figure.
- **Model and decision:** LightGBM 4.7.0 (MIT), size, and the pool-to-best-S1 assignment and expected-F0.5 k rule.
- **Reproduction:** the README command sequence, Python 3.12, the pinned requirements, 8 threads, and peak RAM/disk/runtime per step (from the EC2 logs).
- **Evidence:** 5-fold CV protocol (stratified by country x match count, entity-disjoint), CV mean ± std, per-country results, public LB. The France estimate (0.907) is labelled as an unvalidated inference from the LB, with the ±0.003 rounding band.
