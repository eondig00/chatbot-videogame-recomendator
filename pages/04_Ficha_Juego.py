# pages/03_Ficha_Juego.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import load_config

st.title("📄 Ficha del juego")

# 1) Recuperar appid seleccionado
appid = st.session_state.get("selected_appid")

if appid is None:
    st.warning("No hay ningún juego seleccionado. Vuelve a **Buscar** o **Librería** y pulsa en 'Ver ficha'.")
    st.stop()

# 2) Cargar datos
cfg = load_config()
data_parquet = Path(cfg["paths"]["processed"]) / "games.parquet"
df = pd.read_parquet(data_parquet)

row = df.loc[df["appid"] == appid]
if row.empty:
    st.error(f"No se ha encontrado el juego con appid={appid} en el dataset.")
    st.stop()

row = row.iloc[0]

# 3) Cabecera con imagen + título
cols = st.columns([1, 2])
with cols[0]:
    img = row.get("header_image")
    if isinstance(img, str) and img.startswith("http"):
        st.image(img, width="stretch")
with cols[1]:
    st.markdown(f"## {row.get('name', 'Sin nombre')}")
    st.markdown(f"[Ver en Steam](https://store.steampowered.com/app/{appid})")

    us = row.get("user_score")
    if us is not None and not (isinstance(us, float) and np.isnan(us)):
        st.markdown(f"⭐ **Puntuación usuarios:** {us:.1f}/100")
    else:
        st.markdown("⭐ **Puntuación usuarios:** N/D")

    price = row.get("price")
    if price is not None and not (isinstance(price, float) and np.isnan(price)):
        st.markdown(f"💰 **Precio aproximado:** {price} €")

# 4) Detalles tipo Steam
st.markdown("---")

short_desc = row.get("short_description") or ""
about = row.get("about_the_game") or ""

if short_desc:
    st.subheader("Descripción corta")
    st.write(short_desc)

if about:
    st.subheader("Acerca del juego")
    st.write(about)

# 5) Metadatos: géneros, tags, categorías…
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
                pass
        return [v]
    return []

genres = _safe_list(row.get("genres"))
tags = _safe_list(row.get("tags"))
cats = _safe_list(row.get("categories"))

cols2 = st.columns(3)
with cols2[0]:
    st.caption("🎭 Géneros")
    if genres:
        st.write(", ".join(genres))
    else:
        st.write("—")

with cols2[1]:
    st.caption("🏷️ Tags")
    if tags:
        st.write(", ".join(tags[:15]))
    else:
        st.write("—")

with cols2[2]:
    st.caption("📦 Categorías")
    if cats:
        st.write(", ".join(cats))
    else:
        st.write("—")
