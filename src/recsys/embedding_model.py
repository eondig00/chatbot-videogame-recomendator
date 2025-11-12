import argparse, gc, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from numpy.lib.format import open_memmap
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("embed")

def _tokens(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, np.ndarray): return [str(t).strip() for t in x.tolist() if str(t).strip()]
    try:
        if pd.isna(x): return []
    except Exception:
        pass
    s = str(x).strip()
    if not s: return []
    return [t.strip() for t in s.split(",")] if "," in s else [s]

def build_corpus(df: pd.DataFrame) -> list[str]:
    def row_text(r):
        name  = str(r.get("name" , "") or "")
        short = str(r.get("short_description","") or "")
        tags  = " ".join(_tokens(r.get("tags")))
        genres= " ".join(_tokens(r.get("genres")))
        cats  = " ".join(_tokens(r.get("categories")))
        return f"{name}\n{short}\nTags: {tags}\nGenres: {genres}\nCategories: {cats}"
    return [row_text(r) for _, r in df.iterrows()]

def main(data_path: str | None, out_dir: str | None, model_name: str | None, max_rows: int | None):
    cfg = load_config()
    data_path = data_path or str(Path(cfg["paths"]["processed"]) / "games.parquet")
    out_dir = Path(out_dir or cfg["paths"]["embeddings"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or cfg["embedding"]["model"]
    batch = int(cfg["embedding"].get("batch_size", 16))

    log.info(f"Loading {data_path}")
    df = pd.read_parquet(data_path)

    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        log.info(f"Sampled to {len(df)} rows (max_rows={max_rows})")

    texts = build_corpus(df)
    n = len(texts)

    npy  = out_dir / "embeddings.npy"
    meta = out_dir / "meta.csv"
    prog = out_dir / "embeddings.done"

    log.info(f"Loading model {model_name} (CPU)")
    model = SentenceTransformer(model_name)
    d = int(model.encode(texts[:1], normalize_embeddings=True).shape[1])

    # .npy con cabecera válida y acceso memmap
    mm = open_memmap(npy, mode="w+", dtype="float32", shape=(n, d))

    start_i = 0
    if prog.exists():
        try:
            start_i = int(prog.read_text().strip())
            if not (0 <= start_i <= n): start_i = 0
            log.info(f"Resuming from row {start_i}")
        except Exception:
            start_i = 0

    for i in range(start_i, n, batch):
        chunk = texts[i:i+batch]
        X = model.encode(chunk, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        mm[i:i+len(chunk)] = X
        prog.write_text(str(i + len(chunk)))
        if (i // batch) % 10 == 0:
            log.info(f"{i+len(chunk)}/{n} embeddings")
        del X; gc.collect()

    # “flush” del memmap
    del mm; gc.collect()

    df[["appid","name"]].to_csv(meta, index=False)
    log.info(f"Saved embeddings to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()
    main(args.data, args.out, args.model, args.max_rows)
