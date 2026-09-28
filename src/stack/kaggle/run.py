"""Kaggle kernel entry point: python run.py <tag> sets the environment of that run (table in ../README.md, "GPU steps")
and runs its script. Values already set in the environment win.
"""
import os
import runpy
import sys

RUNS = {
    "e5s": ("ce.py", {}),
    "e5i": ("ce.py", {"CE_MODE": "incountry", "CE_TAG": "e5i"}),
    "e5b": ("ce.py", {"CE_MODE": "incountry", "CE_TAG": "e5b", "CE_MODEL": "intfloat/multilingual-e5-base", "CE_REVISION": "d128750597153bb5987e10b1c3493a34e5a4502a",
                      "CE_MAXLEN": "128", "CE_BATCH": "64", "CE_LR": "3e-5", "CE_TRAIN_MIN": "200"}),
    "e5bc": ("ce_france.py", {}),
}

script, env = RUNS[sys.argv[1]]
for k, v in env.items():
    os.environ.setdefault(k, v)
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), script), run_name="__main__")
