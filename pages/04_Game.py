# pages/04_Juego.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.title("🗂️ Ficha del juego")
appid = st.query_params.get("appid", None)

path = Path("data/processed/games.parquet")
if not path.exists():
    st.error("No existe data/processed/games.parquet")
    st.stop()

df = pd.read_parquet(path).set_index("appid")

if not appid:
    st.info("Abre esta página desde 'Buscar' o introduce un appid:")
    appid = st.text_input("AppID")
    if not appid:
        st.stop()

try:
    appid = int(appid)
except:
    st.error("appid inválido")
    st.stop()

if appid not in df.index:
    st.error(f"No existe appid={appid} en el dataset.")
    st.stop()

r = df.loc[appid].to_dict()

# Cabecera
cols = st.columns([1,2])
with cols[0]:
    img = r.get("header_image")
    if isinstance(img, str) and img.startswith(("http://","https://")):
        st.image(r["header_image"], width="stretch")
with cols[1]:
    st.markdown(f"## {r.get('name','(Sin título)')}")
    # score del público
    pos, neg = r.get("positive"), r.get("negative")
    user_score = None
    if "user_score" in r and pd.notna(r["user_score"]):
        user_score = float(r["user_score"])
        if user_score <= 10: user_score *= 10
    elif pos is not None and neg is not None:
        tot = float(pos)+float(neg)
        user_score = 100.0*float(pos)/tot if tot>0 else None
    if user_score is not None:
        st.metric("Score del público", f"{user_score:.1f}%")

    # enlaces
    steam_url = f"https://store.steampowered.com/app/{appid}"
    links = [f"[Steam]({steam_url})"]
    if r.get("website"):
        links.append(f"[Web oficial]({r['website']})")
    st.markdown(" · ".join(links))

# metadatos breves
def _fmt_list(x):
    if isinstance(x, list): return ", ".join(map(str,x))
    return str(x) if pd.notna(x) else ""

st.write("**Géneros:**", _fmt_list(r.get("genres","")))
st.write("**Categorías:**", _fmt_list(r.get("categories","")))
st.write("**Etiquetas:**", _fmt_list(r.get("tags","")))

# descripciones
st.subheader("Descripción breve")
st.write(r.get("short_description","(sin descripción)"))

st.subheader("Acerca del juego")
st.write(r.get("about_the_game","(sin contenido)"))

# galería
shots = r.get("screenshots")
if isinstance(shots, list) and shots:
    st.subheader("Galería")
    for x in shots[:6]:
        url = x.get("path_full") if isinstance(x, dict) else str(x)
        if isinstance(url, str) and url.startswith(("http://","https://")):
            st.image(url, use_column_width=True)
