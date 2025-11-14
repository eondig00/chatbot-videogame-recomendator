import faiss, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config

DATA_PARQUET = Path("data/processed/games.parquet")
EMB_DIR = Path("data/embeddings")
IDX_DIR = Path("data/index")
MODEL_NAME = "all-MiniLM-L6-v2"


# Fix para imagenes
_df = pd.read_parquet(DATA_PARQUET)  # full dataset con header_image y todo
_meta = _df.set_index("appid")       # lookups rápidos por appid

_index = faiss.read_index(str(IDX_DIR / "faiss.index"))
_model = SentenceTransformer(MODEL_NAME)

# Mapear la fila del embeddin -> appid
_emb_ids = np.load(EMB_DIR / "ids.npy") 

def _encode(texts, model_name): 
    return SentenceTransformer(model_name).encode(texts, normalize_embeddings=True).astype("float32")

def _why(row: dict, q: str) -> str:
    parts=[]
    if isinstance(row.get("genres"), list) and row["genres"]:
        parts.append("Géneros: "+", ".join(row["genres"][:3]))
    if isinstance(row.get("tags"), list) and row["tags"]:
        parts.append("Tags: "+", ".join(row["tags"][:3]))
    if pd.notna(row.get("price", np.nan)):
        parts.append(f"Precio aprox: {row['price']}")
    return f"Relacionado con “{q}”. " + " | ".join(parts)

def recommend_by_text(query: str, k: int = 10):
    qv = _model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = _index.search(qv, k)
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        appid = int(_emb_ids[idx])
        row = _meta.loc[appid]
        out.append({
            "appid": appid,
            "name": row.get("name", ""),
            "score": float(score),
            "header_image": row.get("header_image", ""),
        })
    return out