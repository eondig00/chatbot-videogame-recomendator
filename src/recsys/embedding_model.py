import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("embed")


def build_corpus(df: pd.DataFrame) -> list[str]:
    # Tu implementación actual aquí
    texts = []
    for _, r in df.iterrows():
        parts = [
            str(r.get("name") or ""),
            str(r.get("short_description") or ""),
            str(r.get("about_the_game") or ""),
        ]
        texts.append(" ".join(p for p in parts if p.strip()))
    return texts


def main(data_path: str | None, out_path: str | None, model_name: str | None):
    cfg = load_config()

    processed_dir = Path(cfg["paths"]["processed"])
    default_parquet = processed_dir / "games.parquet"

    data_path = Path(data_path) if data_path else default_parquet
    out_path = Path(out_path) if out_path else Path(cfg["paths"]["embeddings"])
    model_name = model_name or cfg["embedding"]["model"]
    batch_size = int(cfg["embedding"].get("batch_size", 16))
    max_rows = cfg["embedding"].get("max_rows")  # opcional, puede no existir

    log.info(f"Loading {data_path}")
    df = pd.read_parquet(data_path)

    if max_rows:
        df = df.head(int(max_rows))
        log.info(f"Using first {len(df)} rows (max_rows={max_rows})")

    texts = build_corpus(df)
    ids = df["appid"].to_numpy(dtype=np.int64)

    log.info(f"Loading model {model_name}")
    model = SentenceTransformer(model_name)

    log.info(f"Encoding {len(texts)} rows, batch_size={batch_size}")
    X = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "embeddings.npy", X)
    np.save(out_path / "ids.npy", ids)

    # 👇 Guardamos el nombre del modelo para que el recomendador no se equivoque
    (out_path / "model_name.txt").write_text(model_name.strip())

    log.info(f"Saved embeddings, ids and model_name in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    main(args.data, args.out, args.model)
