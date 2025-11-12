import argparse, numpy as np, pandas as pd, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("search")

def build_index(emb_path: Path, out_dir: Path):
    X = np.load(emb_path, mmap_mode="r")
    index = faiss.IndexFlatIP(X.shape[1])  # inner product (cosine if normalized)
    index.add(X)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))

def encode_query(model_name: str, text: str) -> np.ndarray:
    v = SentenceTransformer(model_name).encode([text], normalize_embeddings=True)
    return v.astype("float32")

def search(query: str, topk: int):
    cfg = load_config()
    emb_dir = Path(cfg["paths"]["embeddings"])
    idx_dir = Path(cfg["paths"]["index"])
    index = faiss.read_index(str(idx_dir / "faiss.index"))
    v = encode_query(cfg["embedding"]["model"], query)
    D, I = index.search(v, topk)
    meta = pd.read_csv(emb_dir / "meta.csv")
    out = meta.iloc[I[0]].copy(); out["score"] = D[0]
    return out

def main(build: bool, query: str | None, topk: int):
    cfg = load_config()
    emb_dir = Path(cfg["paths"]["embeddings"])
    idx_dir = Path(cfg["paths"]["index"])

    if build:
        build_index(emb_dir / "embeddings.npy", idx_dir)
        log.info("Index built")

    if query:
        df = search(query, topk)
        print(df.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default=None)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    main(args.build, args.query, args.topk)
