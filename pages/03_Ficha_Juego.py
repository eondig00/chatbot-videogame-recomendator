import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import html
import urllib.parse
import ollama

from src.utils.config import load_config
from src.llm.likes import LIKES
from src.recsys.recommender import recommend_by_text
from src.llm.memory import GLOBAL_USER_MEMORY

mem = GLOBAL_USER_MEMORY
user_prefs = mem.get_explicit_prefs()

# ================== ESTILO ==================
st.markdown("""
<style>

/* ===========================================================
   FIXES DE LAYOUT — alineación, márgenes, proporciones
   =========================================================== */

/* Centrar verticalmente las dos columnas del hero */
.game-hero [data-testid="column"] > div {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Quitar padding vertical interno de columnas */
.game-hero [data-testid="column"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* Imagen hero consistente */
.game-img,
.game-hero img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}

/* Botones de acción tipo Steam (Like / Fav) */
.action-row {
    margin-top: 0.8rem;
    margin-bottom: 0.4rem;
}
.action-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    border: 1px solid #1e293b;
    background: #020617;
    color: #e5e7eb;
    font-size: 0.9rem;
    margin-right: 0.6rem;
    cursor: pointer;
    text-decoration: none;
}
.action-btn:hover {
    border-color: #38bdf8;
    background: #0b1120;
}

/* ===========================================================
   RESTO DE ESTILO
   =========================================================== */

.game-hero {
    background: #020617;
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid #1e293b;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    margin-top: 0.8rem;
    margin-bottom: 1.5rem;
}
.badge {
    display: inline-block;
    padding: 0.3rem 0.55rem;
    font-size: 0.8rem;
    border-radius: 12px;
    background: #1e293b;
    color: #e2e8f0;
    margin-right: 0.4rem;
    border: 1px solid #334155;
}
.section-box {
    background: #0f172a;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #1e293b;
    margin-bottom: 1.2rem;
}
.meta-chip {
    display: inline-block;
    padding: 0.25rem 0.55rem;
    background: #1e293b;
    border-radius: 12px;
    font-size: 0.75rem;
    color: #cbd5e1;
    margin: 0.2rem;
    border: 1px solid #334155;
}
.sim-card {
    background: #020617;
    border-radius: 14px;
    padding: 0.6rem;
    border: 1px solid #1e293b;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    height: 100%;
}
.sim-title {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.sim-meta {
    font-size: 0.75rem;
    opacity: 0.8;
}

</style>
""", unsafe_allow_html=True)


# ================== TRADUCCIÓN ==================
def _translate_to_es(text: str) -> str:
    if not text:
        return ""
    prompt = (
        "Traduce al español neutro, manteniendo el tono informativo, "
        "sin añadir ni quitar información:\n\n"
        f"{text}"
    )
    try:
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": "Eres un traductor profesional EN→ES."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp["message"]["content"].strip()
    except Exception:
        # Si falla el modelo, devolvemos el original en inglés.
        return text


@st.cache_data(show_spinner=False)
def translate_cached(text: str) -> str:
    return _translate_to_es(text)


# ================== TÍTULO ==================
st.title("📄 Ficha del juego")

# ================== 1) Recuperar appid ==================
appid = st.session_state.get("selected_appid")

if appid is None:
    st.warning("No hay ningún juego seleccionado. Vuelve a **Buscar** o **Librería**.")
    st.stop()

# ================== 2) Cargar datos ==================
cfg = load_config()
data_parquet = Path(cfg["paths"]["processed"]) / "games.parquet"
df = pd.read_parquet(data_parquet)

row = df.loc[df["appid"] == appid]
if row.empty:
    st.error("Juego no encontrado en el dataset.")
    st.stop()

row = row.iloc[0]
name = row.get("name", "Sin nombre")

# Nombre del juego arriba, separado del hero
st.markdown(f"## {html.escape(str(name))}")

# ================== Helpers ==================
def _safe_list(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, np.ndarray):
        return [str(x) for x in v.tolist()]
    if isinstance(v, str):
        if v.startswith("[") and v.endswith("]"):
            import json
            try:
                return [str(x) for x in json.loads(v)]
            except Exception:
                return [v]
        return [v]
    return []


genres = _safe_list(row.get("genres"))
tags = _safe_list(row.get("tags"))
cats = _safe_list(row.get("categories"))

# ================== 3) Cabecera tipo Steam ==================
with st.container():
    st.markdown('<div class="game-hero">', unsafe_allow_html=True)

    cols = st.columns([1.2, 2])
    with cols[0]:
        img = row.get("header_image")
        if isinstance(img, str) and img.startswith("http"):
            st.image(img, width='stretch')

    with cols[1]:
        st.markdown(f"[🔗 Ver en Steam](https://store.steampowered.com/app/{appid})")

        # ⭐ Puntuación + 💰 Precio
        us = row.get("user_score")
        if isinstance(us, (int, float)) and not np.isnan(us):
            st.markdown(
                f'<span class="badge">⭐ {us:.1f}/100</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<span class="badge">⭐ N/D</span>', unsafe_allow_html=True)

        price = row.get("price")
        if isinstance(price, (int, float)) and not np.isnan(price):
            st.markdown(
                f'<span class="badge">💰 {price:.2f} €</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<span class="badge">💰 N/D</span>', unsafe_allow_html=True)

        # ---- Barra de acciones Like / Fav ----
        liked = LIKES.is_liked(int(appid))
        starred = LIKES.is_starred(int(appid))

        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        c_like, c_fav = st.columns([1, 1])

        with c_like:
            label_like = "👍 Me gusta" if not liked else "💔 Quitar me gusta"
            if st.button(label_like, key=f"ficha-like-{appid}"):
                if liked:
                    LIKES.unlike(int(appid))
                else:
                    LIKES.like(int(appid))
                st.rerun()

        with c_fav:
            label_fav = "⭐ Añadir a favoritos" if not starred else "★ Quitar de favoritos"
            if st.button(label_fav, key=f"ficha-star-{appid}"):
                LIKES.toggle_star(int(appid))
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ================== 3.5) Stats rápidos en sección aparte ==================
stats_lines = []
stats_lines.append(f"- 🎭 Géneros: {len(genres) if genres else 'N/D'}")
stats_lines.append(f"- 🏷️ Tags: {len(tags) if tags else 'N/D'}")
stats_lines.append(f"- 📦 Categorías: {len(cats) if cats else 'N/D'}")
if isinstance(row.get("price"), (int, float)):
    stats_lines.append(
        f"- 💵 ¿Free to Play?: {'Sí' if row['price'] == 0 else 'No'}"
    )
us = row.get("user_score")
if isinstance(us, (int, float)) and not np.isnan(us):
    stats_lines.append(
        f"- 📊 Nota por encima del filtro: "
        f"{'Sí' if us >= user_prefs.min_user_score else 'No'}"
    )

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("⚙️ Stats rápidos")
st.markdown("\n".join(stats_lines))
st.markdown("</div>", unsafe_allow_html=True)

# ================== 3.6) Galería de imágenes ==================
screenshots = []
if "screenshots" in df.columns:
    screenshots = _safe_list(row.get("screenshots"))

valid_shots = [
    url for url in screenshots
    if isinstance(url, str) and url.startswith(("http://", "https://"))
]

if not valid_shots and isinstance(img, str) and img.startswith(("http://", "https://")):
    valid_shots = [img]

if valid_shots:
    st.subheader("📸 Galería")
    n = min(len(valid_shots), 4)
    cols_shots = st.columns(n)
    for c, url in zip(cols_shots, valid_shots[:n]):
        with c:
            st.image(url, width='stretch')

# ================== 3.7) Tráilers / vídeos ==================
st.subheader("🎬 Tráilers y vídeos")

yt_query = f"{name} trailer pc game"
yt_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(yt_query)
st.markdown(f"[🔍 Buscar tráiler en YouTube]({yt_url})")
st.caption("Se abrirá una búsqueda de tráilers en YouTube en tu navegador.")

# ================== 4) Descripciones (con traducción a español) ==================
translate_es = st.checkbox("Traducir descripciones al español", value=True)

short_desc = row.get("short_description") or ""
about = row.get("about_the_game") or ""

if short_desc:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📝 Descripción corta")
    if translate_es:
        st.write(translate_cached(short_desc))
    else:
        st.write(short_desc)
    st.markdown("</div>", unsafe_allow_html=True)

if about:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📘 Acerca del juego")
    if translate_es:
        st.write(translate_cached(about))
    else:
        st.write(about)
    st.markdown("</div>", unsafe_allow_html=True)

# ================== 5) Metadatos ==================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
cols2 = st.columns(3)

with cols2[0]:
    st.caption("🎭 Géneros")
    if genres:
        st.markdown(
            "".join(
                f'<span class="meta-chip">{html.escape(g)}</span>' for g in genres
            ),
            unsafe_allow_html=True,
        )
    else:
        st.write("—")

with cols2[1]:
    st.caption("🏷️ Tags")
    if tags:
        st.markdown(
            "".join(
                f'<span class="meta-chip">{html.escape(t)}</span>' for t in tags[:15]
            ),
            unsafe_allow_html=True,
        )
    else:
        st.write("—")

with cols2[2]:
    st.caption("📦 Categorías")
    if cats:
        st.markdown(
            "".join(
                f'<span class="meta-chip">{html.escape(c)}</span>' for c in cats
            ),
            unsafe_allow_html=True,
        )
    else:
        st.write("—")

st.markdown("</div>", unsafe_allow_html=True)

# ================== 6) Juegos similares (FAISS) ==================
st.subheader("🎮 Juegos similares")

q_parts = [
    str(row.get("name", "")),
    row.get("short_description") or "",
]
query = ". ".join([p for p in q_parts if p]).strip()

recs = []
if query:
    recs = recommend_by_text(
        query,
        k=5,
        exclude_nsfw=user_prefs.avoid_nsfw,
    )

if not recs:
    st.info("No se han encontrado juegos similares.")
else:
    cols_sim = st.columns(5)
    for col, r in zip(cols_sim, recs[:5]):
        with col:
            with st.container():
                st.markdown('<div class="sim-card">', unsafe_allow_html=True)
                if r.get("header_image"):
                    st.image(r["header_image"], width='stretch')

                st.markdown(
                    f'<div class="sim-title">{html.escape(str(r["name"]))}</div>',
                    unsafe_allow_html=True,
                )

                us2 = r.get("user_score")
                meta_txt = ""
                if isinstance(us2, (int, float)) and not np.isnan(us2):
                    meta_txt += f"⭐ {us2:.1f}/100"

                st.markdown(
                    f'<div class="sim-meta">{meta_txt}</div>',
                    unsafe_allow_html=True,
                )

                if st.button("📁 Ficha", key=f"sim-ficha-{r['appid']}"):
                    st.session_state["selected_appid"] = int(r["appid"])
                    st.switch_page("pages/03_Ficha_Juego.py")

                st.markdown("</div>", unsafe_allow_html=True)
