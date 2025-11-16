# pages/03_Libreria.py
import streamlit as st
import duckdb as ddb
from pathlib import Path

st.set_page_config(page_title="🎮 Catálogo de Juegos", layout="wide")
st.title("🎮 Catálogo de Videojuegos")

PARQUET = Path("data/processed/games.parquet")
if not PARQUET.exists():
    st.error("No existe data/processed/games.parquet")
    st.stop()


@st.cache_resource
def get_con():
    con = ddb.connect(database=":memory:")
    con.execute(
        f"CREATE OR REPLACE VIEW games AS "
        f"SELECT * FROM read_parquet('{PARQUET.as_posix()}');"
    )
    return con


con = get_con()

# 🔎 Barra de búsqueda
q = st.text_input("Buscar por nombre, género o etiqueta…", "")
limit = st.slider("Resultados mostrados", 20, 200, 50, 10)

base_query = """
SELECT appid,
       name,
       header_image,
       short_description,
       genres,
       categories,
       tags,
       user_score
FROM games
"""

if q.strip():
    q_sql = q.replace("'", "''").lower()
    where = f"""
        WHERE LOWER(CAST(name AS VARCHAR)) LIKE '%{q_sql}%'
        OR LOWER(CAST(genres AS VARCHAR)) LIKE '%{q_sql}%'
        OR LOWER(CAST(tags AS VARCHAR)) LIKE '%{q_sql}%'
        OR LOWER(CAST(categories AS VARCHAR)) LIKE '%{q_sql}%'
    """
else:
    where = ""

df = con.execute(base_query + where + f" LIMIT {int(limit)}").fetchdf()

if df.empty:
    st.info("No se encontraron juegos.")
    st.stop()

# 🧱 Cuadrícula de tarjetas
cols_per_row = 5
for i in range(0, len(df), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, (_, r) in zip(cols, df.iloc[i : i + cols_per_row].iterrows()):
        with col:
            with st.container(border=True):
                img = r.get("header_image")
                if isinstance(img, str) and img.startswith(("http://", "https://")):
                    st.image(img, width="stretch")

                st.markdown(f"**{r['name']}**")

                # ⭐ Puntuación de usuarios si existe
                us = r.get("user_score")
                if isinstance(us, (int, float)):
                    st.caption(f"⭐ {us:.1f}/100")

                # Descripción corta recortada
                if r.get("short_description"):
                    st.caption((r["short_description"][:90] + "…").strip())

                # 🔗 Botón para ir a la ficha dentro de la app
                if st.button("🗂️ Ver ficha del juego", key=f"lib-{r['appid']}"):
                    st.session_state["selected_appid"] = int(r["appid"])
                    st.switch_page("pages/04_Ficha_Juego.py")
