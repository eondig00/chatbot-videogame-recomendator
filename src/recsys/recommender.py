import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any 

from src.utils.config import load_config
from src.recsys.user_prefs import load_user_prefs 
from src.llm.memory import GLOBAL_USER_MEMORY

# ----------------- Config y rutas -----------------
_cfg = load_config()

DATA_PARQUET = Path(_cfg["paths"]["processed"]) / "games.parquet"
EMB_DIR = Path(_cfg["paths"]["embeddings"])
IDX_DIR = Path(_cfg["paths"]["index"])


# modelo usado durante el embedding
model_file = EMB_DIR / "model_name.txt"
if model_file.exists():
    MODEL_NAME = model_file.read_text().strip()
else:
    MODEL_NAME = _cfg["embedding"]["model"]  # fallback

# ----------------- Carga datos + FAISS -----------------
_df = pd.read_parquet(DATA_PARQUET)
_meta = _df.set_index("appid")

_index = faiss.read_index(str(IDX_DIR / "faiss.index"))
_model = SentenceTransformer(MODEL_NAME)
_emb_ids = np.load(EMB_DIR / "ids.npy")

# ----------------- Filtro NSFW -----------------
_NSFWTOK = {
    "hentai","nsfw","porn","porno","eroge","erótica","erotica",
    "sexual","sexual-content","sexual content","nudity",
    "adult only","adults only","18+","r18",
    "sex","explicit","ecchi","yaoi","yuri",
}

_token_re = re.compile(r"[A-Za-z0-9ñáéíóúüç+#]+")

def _as_tokens(value):
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

def _row_is_nsfw(row):
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

    # tokens directos
    if any(tok in _NSFWTOK for tok in toks):
        return True

    # tokens unidos
    joined = " ".join(sorted(toks)).replace("-", " ")
    for kw in _NSFWTOK:
        if kw in joined:
            return True

    return False


# ----------------- Formateo de cada juego -----------------
def _safe_list(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, np.ndarray):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        if v.startswith("[") and v.endswith("]"):
            try:
                import json
                return [str(x) for x in json.loads(v)]
            except:
                return [v]
        return [v]
    return []

def _row_dict(appid: int) -> dict:
    r = _meta.loc[appid]
    d = r.to_dict()

    # Sanitizamos user_score: None si viene NaN o no numérico
    raw_us = d.get("user_score")
    us_clean = None
    if isinstance(raw_us, (int, float)) and not (isinstance(raw_us, float) and np.isnan(raw_us)):
        us_clean = float(raw_us)

    return {
        "appid": int(appid),
        "name": d.get("name") or "",
        "header_image": d.get("header_image") or "",
        "short_description": d.get("short_description") or "",
        "user_score": us_clean,
        "genres": _safe_list(d.get("genres")),
        "tags": _safe_list(d.get("tags")),
        "categories": _safe_list(d.get("categories")),
        "price": d.get("price"),
        # campos extra para ranking más global
        "num_reviews_total": d.get("num_reviews_total"),
    }


def _apply_user_prefs(candidatos: list[dict], prefs) -> list[dict]:
    if not candidatos:
        return []

    liked = set(g.lower() for g in (prefs.liked_genres or []))
    disliked = set(g.lower() for g in (prefs.disliked_genres or []))
    avoid_tags = set(t.lower() for t in (prefs.avoid_tags or []))

    filtrados: list[dict] = []
    for c in candidatos:
        genres = [str(g).lower() for g in (c.get("genres") or [])]
        tags   = [str(t).lower() for t in (c.get("tags") or [])]

        # 1) Filtros duros
        if disliked and any(g in disliked for g in genres):
            continue
        if avoid_tags and any(t in avoid_tags for t in tags):
            continue

        us = c.get("user_score") or 0.0
        if us < prefs.min_user_score:
            continue

        nrev = c.get("num_reviews_total") or 0
        if nrev < prefs.min_num_reviews:
            continue

        price = c.get("price")
        if prefs.max_price is not None and price is not None:
            try:
                if float(price) > float(prefs.max_price):
                    continue
            except Exception:
                pass

        # 2) Bonus por géneros favoritos
        bonus = 0.0
        if liked and any(g in liked for g in genres):
            bonus += 0.05  # +0.05 al score

        c["score"] = float(c.get("score", 0.0) + bonus)
        filtrados.append(c)

    return filtrados

# ----------------- Buscador principal -----------------
def recommend_by_text(query: str, k: int = 10, exclude_nsfw: bool = True):
    """
    Recomendador basado en:
    - FAISS con embeddings normalizados (coseno)
    - Filtro NSFW
    - Reordenación por FAISS + calidad + popularidad + prefs usuario
    """

    # 1) Embedding query
    qv = _model.encode([query], normalize_embeddings=True).astype("float32")

    # 2) Pedimos TOP grandes para filtrar y reordenar bien
    candidate_count = max(100, k * 20)
    scores, idxs = _index.search(qv, candidate_count)

    # 💡 prefs ACTUALES
    prefs = GLOBAL_USER_MEMORY.get_explicit_prefs()
    effective_exclude = exclude_nsfw or prefs.avoid_nsfw

    candidatos = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue

        appid = int(_emb_ids[idx])
        row = _meta.loc[appid]

        if effective_exclude and _row_is_nsfw(row):
            continue

        info = _row_dict(appid)
        info["faiss_score"] = float(score)
        candidatos.append(info)

    if not candidatos:
        return []

    # 3) Reordenar: similitud + calidad + popularidad
    faiss_scores = np.array(
        [c["faiss_score"] for c in candidatos],
        dtype="float32",
    )
    fs_min, fs_max = faiss_scores.min(), faiss_scores.max()
    denom = (fs_max - fs_min) or 1.0

    for c in candidatos:
        fs_norm = (c["faiss_score"] - fs_min) / denom

        user_score = c.get("user_score")
        if user_score is None or user_score <= 0:
            user_norm = 0.0
        else:
            user_norm = float(user_score) / 100.0

        num_reviews = c.get("num_reviews_total")
        if num_reviews is None or num_reviews <= 0:
            num_reviews = 0
        if num_reviews < 20:
            pop = 0.0
        else:
            pop = np.log10(num_reviews + 10) / 3.0

        c["score"] = float(
            0.70 * fs_norm +
            0.30 * (user_norm * pop)
        )

    # 4) Aplicar preferencias del usuario (filtrado + bonus)
    candidatos = _apply_user_prefs(candidatos, prefs)

    if not candidatos:
        return []

    # 5) Orden final
    candidatos.sort(key=lambda c: c["score"], reverse=True)
    return candidatos[:k]

