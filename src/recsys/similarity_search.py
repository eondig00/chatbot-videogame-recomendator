import argparse
from pathlib import Path

import faiss
import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("faiss")


def build_index(emb_dir: Path, idx_dir: Path):
    # Cargamos embeddings
    emb_path = emb_dir / "embeddings.npy"
    log.info(f"Loading embeddings from {emb_path}")
    X = np.load(emb_path)

    if X.dtype != np.float32:
        log.info(f"Converting embeddings to float32 (was {X.dtype})")
        X = X.astype("float32", copy=False)

    n, d = X.shape
    log.info(f"Building FAISS index, n={n}, dim={d}")

    # Normalizamos por seguridad (aunque ya vengan normalizados del encoder)
    faiss.normalize_L2(X)

    index = faiss.IndexFlatIP(d)
    index.add(X)

    idx_dir.mkdir(parents=True, exist_ok=True)
    out_path = idx_dir / "faiss.index"
    faiss.write_index(index, str(out_path))
    log.info(f"Index written to {out_path}")


def main(build: bool):
    cfg = load_config()
    emb_dir = Path(cfg["paths"]["embeddings"])
    idx_dir = Path(cfg["paths"]["index"])

    if build:
        build_index(emb_dir, idx_dir)
    else:
        log.info("Nothing to do. Use --build to create the FAISS index.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build FAISS index from embeddings.npy")
    args = parser.parse_args()
    main(args.build)
