import faiss, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.utils.config import load_config

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

def recommend_by_text(query: str, k: int = 10) -> list[dict]:
    cfg = load_config()
    data = pd.read_parquet(Path(cfg["paths"]["processed"]) / "games.parquet")
    meta = pd.read_csv(Path(cfg["paths"]["embeddings"]) / "meta.csv")
    index = faiss.read_index(str(Path(cfg["paths"]["index"]) / "faiss.index"))
    qv = _encode([query], cfg["embedding"]["model"])
    D, I = index.search(qv, k)

    # acceso rápido por appid
    data = data.set_index("appid", drop=False)
    items=[]
    for score, i in zip(D[0], I[0]):
        appid = int(meta.iloc[i]["appid"])
        row = data.loc[appid].to_dict()
        items.append({
            "appid": appid,
            "name": row.get("name",""),
            "score": float(score),
            "why": _why(row, query)
        })
    return items
