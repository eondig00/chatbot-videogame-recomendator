import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config

# ----------------- Config y rutas -----------------
_cfg = load_config()

DATA_PARQUET = Path(_cfg["paths"]["processed"]) / "games.parquet"
EMB_DIR = Path(_cfg["paths"]["embeddings"])
IDX_DIR = Path(_cfg["paths"]["index"])

# modelo: primero miramos lo que se usó al embebir
model_file = EMB_DIR / "model_name.txt"
if model_file.exists():
    MODEL_NAME = model_file.read_text().strip()
else:
    MODEL_NAME = _cfg["embedding"]["model"]

# ----------------- Carga datos y FAISS -----------------
_df = pd.read_parquet(DATA_PARQUET)
_meta = _df.set_index("appid")

_index = faiss.read_index(str(IDX_DIR / "faiss.index"))
_model = SentenceTransformer(MODEL_NAME)
_emb_ids = np.load(EMB_DIR / "ids.npy")

# ----------------- NSFW / hentai filter -----------------
_NSFWTOK = {
    "hentai", "nsfw", "porn", "porno", "eroge", "erótica", "erotica",
    "sexual", "sexual-content", "sexual content", "nudity",
    "adult only", "adults only", "18+", "r18",
    "sex", "explicit", "ecchi", "yaoi", "yuri"
}

_token_re = re.compile(r"[A-Za-z0-9ñáéíóúüç+#]+")


def _as_tokens(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        toks = []
        for x in value:
            if isinstance(x, str):
                toks += [t.lower() for t in _token_re.findall(x)]
        return toks
    if isinstance(value, str):
        return [t.lower() for t in _token_re.findall(value)]
    return []


def _row_is_nsfw(row: pd.Series) -> bool:
    fields = [
        row.get("tags"),
        row.get("genres"),
        row.get("categories"),
        row.get("short_description"),
        row.get("about_the_game"),
    ]
    toks = set()
    for f in fields:
        toks.update(_as_tokens(f))

    if any(tok in _NSFWTOK for tok in toks):
        return True

    joined = " ".join(sorted(toks)).replace("-", " ")
    for kw in _NSFWTOK:
        if kw in joined:
            return True
    return False


def _safe_list(v):
    """Convierte listas, arrays o strings raros en una lista de str."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, np.ndarray):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        # puede venir como texto tipo '["Action","RPG"]'
        if v.startswith("[") and v.endswith("]"):
            try:
                import json
                return [str(x) for x in json.loads(v)]
            except Exception:
                pass
        return [v]
    return []


def _row_dict(appid: int) -> dict:
    r = _meta.loc[appid]
    d = r.to_dict()
    return {
        "appid": int(appid),
        "name": d.get("name") or "",
        "header_image": d.get("header_image") or "",
        "short_description": d.get("short_description") or "",
        "user_score": d.get("user_score"),
        "genres": _safe_list(d.get("genres")),
        "tags": _safe_list(d.get("tags")),
        "categories": _safe_list(d.get("categories")),
        "price": d.get("price"),
    }


def recommend_by_text(query: str, k: int = 10, exclude_nsfw: bool = True):
    qv = _model.encode([query], normalize_embeddings=True).astype("float32")
    # Pedimos más resultados a FAISS para poder filtrar hentai
    top_k_probe = max(50, 10 * k)
    scores, idxs = _index.search(qv, top_k_probe)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        appid = int(_emb_ids[idx])
        row = _meta.loc[appid]
        if exclude_nsfw and _row_is_nsfw(row):
            continue
        out = _row_dict(appid)
        out["score"] = float(score)
        results.append(out)
        if len(results) >= k:
            break

    return results
