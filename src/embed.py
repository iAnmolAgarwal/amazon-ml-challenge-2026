import argparse
import time
from pathlib import Path

import numpy as np

from common import load_config
from harness_util import get_device, read_table


def safe_name(s):
    return s.replace("/", "__")


def load_images(paths, root):
    from PIL import Image

    out = []
    for p in paths:
        try:
            out.append(Image.open(Path(root) / p).convert("RGB"))
        except Exception:
            out.append(Image.new("RGB", (32, 32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="sentence-transformers name, e.g. all-MiniLM-L6-v2 or clip-ViT-B-32")
    ap.add_argument("--modality", choices=["text", "image"], required=True)
    ap.add_argument("--input", required=True, help="csv/parquet with the column to embed")
    ap.add_argument("--col", required=True, help="text column, or image filename column")
    ap.add_argument("--image-root", default="", help="dir prepended to image filenames")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="", help="suffix for the cache file (e.g. train/test)")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = get_device(cfg)
    name = f"{safe_name(a.model)}_{a.modality}" + (f"_{a.tag}" if a.tag else "")
    final = Path(cfg["cache_dir"]) / f"{name}.npy"
    if final.exists():
        print(f"cache hit {final}, skipping")
        return
    items = read_table(a.input)[a.col].to_list()
    if a.limit:
        items = items[: a.limit]
    part_dir = Path(cfg["cache_dir"]) / f"{name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(items) + a.chunk - 1) // a.chunk
    print(f"{len(items)} items, {n_chunks} chunks of {a.chunk}, device={device}, model={a.model}")
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, cfg.get("num_workers", 2)))
    model = SentenceTransformer(a.model, device=device)
    t0, done_items = time.time(), 0
    for i in range(n_chunks):
        part = part_dir / f"{i:05d}.npy"
        if part.exists():
            continue
        batch = items[i * a.chunk : (i + 1) * a.chunk]
        if a.modality == "image":
            batch = load_images(batch, a.image_root)
        emb = model.encode(batch, batch_size=a.batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        np.save(part, emb.astype(np.float32))
        done_items += len(batch)
        rate = done_items / (time.time() - t0)
        print(f"chunk {i + 1}/{n_chunks} done, {rate:.1f} items/s, eta {(len(items) - (i + 1) * a.chunk) / max(rate, 1e-9) / 60:.1f} min", flush=True)
    parts = [np.load(part_dir / f"{i:05d}.npy", mmap_mode="r") for i in range(n_chunks)]
    out = np.concatenate(parts).astype(np.float32)
    np.save(final, out)
    for p in part_dir.iterdir():
        p.unlink()
    part_dir.rmdir()
    print(f"saved {final} shape={out.shape} in {time.time() - t0:.1f}s ({len(items) / (time.time() - t0):.1f} items/s incl. model load)")


if __name__ == "__main__":
    main()
