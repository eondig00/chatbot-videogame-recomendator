import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("embed")


# ===========================
#   CORPUS REAL Y OPTIMIZADO
# ===========================
def build_corpus(df: pd.DataFrame) -> list[str]:
    """
    Construye un texto rico para cada juego.
    Mucho más robusto que el original (no crashea si hay None/NaN).
    """
    texts = []

    for _, r in df.iterrows():
        parts = []

        # Nombre del juego
        name = r.get("name")
        if isinstance(name, str):
            parts.append(name)

        # Descripción corta
        sd = r.get("short_description")
        if isinstance(sd, str):
            parts.append(sd)

        # About
        about = r.get("about_the_game")
        if isinstance(about, str):
            parts.append(about)

        # Géneros
        genres = r.get("genres")
        if isinstance(genres, list):
            parts.append("genres: " + " ".join(str(x) for x in genres))

        # Tags
        tags = r.get("tags")
        if isinstance(tags, list):
            parts.append("tags: " + " ".join(str(x) for x in tags))

        # Categories
        categories = r.get("categories")
        if isinstance(categories, list):
            parts.append("categories: " + " ".join(str(x) for x in categories))

        # Fallback para no dejar textos vacíos
        text = " | ".join(p for p in parts if p.strip()) or "game"
        texts.append(text)

    return texts


# ================================
#           MAIN
# ================================
def main(data_path: str | None, out_path: str | None, model_name: str | None):
    cfg = load_config()

    processed_dir = Path(cfg["paths"]["processed"])
    data_path = Path(data_path) if data_path else processed_dir / "games.parquet"

    out_path = Path(out_path) if out_path else Path(cfg["paths"]["embeddings"])
    out_path.mkdir(parents=True, exist_ok=True)

    # Modelo a usar (viene del config.yml pero puedes pasarlo por --model)
    model_name = model_name or cfg["embedding"]["model"]
    batch_size = int(cfg["embedding"].get("batch_size", 16))
    max_rows = cfg["embedding"].get("max_rows")  # opcional

    log.info(f"Loading {data_path}")
    df = pd.read_parquet(data_path)

    if max_rows:
        df = df.head(int(max_rows))
        log.info(f"Using first {len(df)} rows (max_rows={max_rows})")

    # Corpus enriquecido
    log.info("Building corpus...")
    texts = build_corpus(df)
    ids = df["appid"].to_numpy(dtype=np.int64)

    # MODELO
    log.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # ENCODING
    log.info(f"Encoding {len(texts)} items (batch_size={batch_size})")
    X = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    # SAVE
    np.save(out_path / "embeddings.npy", X)
    np.save(out_path / "ids.npy", ids)

    (out_path / "model_name.txt").write_text(model_name.strip(), encoding="utf-8")

    log.info("Saved embeddings + ids + model name successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    main(args.data, args.out, args.model)
