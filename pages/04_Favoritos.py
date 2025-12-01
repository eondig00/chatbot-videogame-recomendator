# pages/04_Favoritos.py
import streamlit as st
import duckdb as ddb
from pathlib import Path
import html

from src.llm.likes import LIKES

st.set_page_config(page_title="⭐ Favoritos", layout="wide")
st.title("⭐ Tus juegos favoritos")

# ---------- Estilos para cards modernas (mismo que Librería) ----------
st.markdown(
    """
    <style>
    .game-card {
        background: #020617;
        border-radius: 18px;
        padding: 0.6rem 0.7rem 0.7rem 0.7rem;
        border: 1px solid #111827;
        box-shadow: 0 8px 24px rgba(0,0,0,0.45);
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        height: 100%;
    }
    .game-card-img {
        width: 100%;
        border-radius: 14px;
        object-fit: cover;
        max-height: 160px;
    }
    .game-card-title {
        font-weight: 600;
        font-size: 0.98rem;
        margin-top: 0.25rem;
    }
    .game-card-meta {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    .game-card-desc {
        font-size: 0.8rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

liked_ids = LIKES.all_likes()
if not liked_ids:
    st.info("Todavía no has marcado ningún juego con like en la librería.")
    st.stop()

ids_sql = ",".join(str(int(a)) for a in liked_ids)

base_query = f"""
SELECT appid,
       name,
       header_image,
       short_description,
       genres,
       categories,
       tags,
       user_score,
       price
FROM games
WHERE appid IN ({ids_sql})
ORDER BY user_score DESC NULLS LAST
"""

df = con.execute(base_query).fetchdf()

cols_per_row = 4
for i in range(0, len(df), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, (_, r) in zip(cols, df.iloc[i : i + cols_per_row].iterrows()):
        with col:
            appid = int(r["appid"])
            name = html.escape(str(r.get("name", "Sin nombre")))
            img_url = r.get("header_image")

            us = r.get("user_score")
            price = r.get("price")
            desc_raw = (r.get("short_description") or "").strip()
            if len(desc_raw) > 120:
                desc_raw = desc_raw[:117].rstrip() + "…"
            desc = html.escape(desc_raw)

            score_part = (
                f"⭐ {us:.1f}/100" if isinstance(us, (int, float)) else "⭐ N/D"
            )
            price_part = (
                f" · 💰 {price:.2f} €" if isinstance(price, (int, float)) else ""
            )

            starred = LIKES.is_starred(appid)
            title_prefix = "⭐ " if starred else ""

            # ----- Card visual -----
            st.markdown(
                f"""
                <div class="game-card">
                  {'<img src="' + img_url + '" class="game-card-img" />' if isinstance(img_url, str) and img_url.startswith(('http://','https://')) else ''}
                  <div class="game-card-title">{title_prefix}{name}</div>
                  <div class="game-card-meta">{score_part}{price_part}</div>
                  <div class="game-card-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ----- Botones -----
            b1, b2, b3 = st.columns(3)

            with b1:
                if st.button("💔 Quitar", key=f"fav-unlike-{appid}"):
                    LIKES.unlike(appid)
                    st.rerun()

            with b2:
                label = "⭐ Fav" if starred else "☆ Fav"
                if st.button(label, key=f"fav-star-{appid}"):
                    LIKES.toggle_star(appid)
                    st.rerun()

            with b3:
                if st.button("📁 Ficha", key=f"fav-{appid}"):
                    st.session_state["selected_appid"] = appid
                    st.switch_page("pages/03_Ficha_Juego.py")
