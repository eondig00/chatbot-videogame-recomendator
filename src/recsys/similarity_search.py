import argparse
from pathlib import Path

import faiss
import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("faiss")


def build_index(emb_dir: Path, idx_dir: Path):
    X = np.load(emb_dir / "embeddings.npy", mmap_mode="r")
    d = X.shape[1]
    log.info(f"Building FAISS index, dim={d}, n={X.shape[0]}")
    index = faiss.IndexFlatIP(d)
    index.add(X)
    idx_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_dir / "faiss.index"))
    log.info(f"Index written to {idx_dir / 'faiss.index'}")


def main(build: bool):
    cfg = load_config()
    emb_dir = Path(cfg["paths"]["embeddings"])
    idx_dir = Path(cfg["paths"]["index"])

    if build:
        build_index(emb_dir, idx_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    main(args.build)
