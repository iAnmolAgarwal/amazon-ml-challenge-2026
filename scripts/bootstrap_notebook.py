"""Kaggle / Colab bootstrap. Paste into the first cell:

    !curl -sL https://raw.githubusercontent.com/... is NOT possible (private repo), so instead:
    upload this file to the notebook (or paste its body) and run:
        %run bootstrap_notebook.py --job embed --model clip-ViT-B-32 --modality image --input data/train.csv --col image --image-root data/images

Token: Kaggle -> Add-ons > Secrets, name GH_TOKEN. Colab -> key icon (userdata), name GH_TOKEN.
Data:  Kaggle -> attach the competition/dataset, pass --data /kaggle/input/<name>.
       Colab  -> mount Drive (auto), pass --data /content/drive/MyDrive/amlc/data.
Outputs land in <repo>/cache and are copied to the platform's output dir (/kaggle/working/out or Drive) for download.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "iAnmolAgarwal/amazon-ml-2026"


def token():
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("GH_TOKEN"), "kaggle"
    except Exception:
        pass
    try:
        from google.colab import userdata

        return userdata.get("GH_TOKEN"), "colab"
    except Exception:
        pass
    t = os.environ.get("GH_TOKEN")
    if not t:
        raise SystemExit("GH_TOKEN not found in Kaggle Secrets / Colab userdata / env")
    return t, "other"


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["embed", "evaluate", "shell"], default="embed")
    ap.add_argument("--data", help="dir with train.csv/test.csv/images; symlinked as repo/data")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--out", help="where to copy cache outputs; default platform output dir")
    a, rest = ap.parse_known_args()
    tok, platform = token()
    repo = Path("/kaggle/working/amazon-ml-2026" if platform == "kaggle" else "/content/amazon-ml-2026")
    if not repo.exists():
        sh(f"git clone -q -b {a.branch} https://x-access-token:{tok}@github.com/{REPO}.git {repo}")
        sh(f"git -C {repo} remote set-url origin https://github.com/{REPO}.git")
    else:
        sh(f"git -C {repo} pull -q")
    os.chdir(repo)
    sh(f"{sys.executable} -m pip install -q polars pyyaml sentence-transformers lightgbm scikit-learn pyarrow psutil")
    if platform == "colab" and not a.data:
        from google.colab import drive

        drive.mount("/content/drive")
        a.data = "/content/drive/MyDrive/amlc/data"
    if a.data:
        if Path("data").is_symlink() or Path("data").exists():
            shutil.rmtree("data", ignore_errors=True) if not Path("data").is_symlink() else Path("data").unlink()
        Path("data").symlink_to(a.data)
    Path("cache").mkdir(exist_ok=True)
    cfg = Path("config.yaml").read_text().replace("device: mps", "device: cuda").replace("num_workers: 2", "num_workers: 4")
    Path("config.yaml").write_text(cfg)
    if a.job == "embed":
        sh(f"{sys.executable} src/embed.py " + " ".join(rest))
    elif a.job == "evaluate":
        sh(f"{sys.executable} src/evaluate.py " + " ".join(rest))
    out = Path(a.out or ("/kaggle/working/out" if platform == "kaggle" else "/content/drive/MyDrive/amlc/cache"))
    out.mkdir(parents=True, exist_ok=True)
    for p in Path("cache").glob("*.npy"):
        shutil.copy(p, out / p.name)
        print("saved", out / p.name, f"{p.stat().st_size / 1e6:.1f} MB")
    print("download these to <mac repo>/cache/ (Kaggle: Output tab; Colab: from Drive)")


if __name__ == "__main__":
    main()
