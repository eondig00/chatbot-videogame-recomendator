# pages/01_Buscar.py
import numpy as np
import streamlit as st
from src.recsys.recommender import recommend_by_text

st.set_page_config(page_title="🔎 Buscar juegos", layout="wide")
st.title("🔎 Buscar juegos por descripción")

# -------------------------
# Entrada del usuario
# -------------------------
q = st.text_input(
    "¿Qué te apetece?",
    placeholder="Ej.: roguelike deckbuilder de partidas cortas, cooperativo, pixel art…"
)
k = st.number_input("Número de resultados", 3, 50, 10, 1)

# -------------------------
# Búsqueda
# -------------------------
f_excluir = st.checkbox("Excluir contenido adulto (NSFW)", value=True)

if st.button("Buscar", type="primary") and q.strip():
    recs = recommend_by_text(q.strip(), int(k), exclude_nsfw=f_excluir)

    if not recs:
        st.info("Sin resultados.")
    else:
        for r in recs:
            with st.container(border=True):
                cols = st.columns([1, 3])

                # -------- IMAGEN --------
                with cols[0]:
                    img = r.get("header_image")
                    if isinstance(img, str):
                        st.image(img, use_container_width=True)

                # -------- TEXTO --------
                with cols[1]:

                    # Título
                    st.markdown(f"### {r['name']}")

                    # ⭐ Puntuación usuarios
                    us = r.get("user_score")
                    if isinstance(us, (int, float)) and not np.isnan(us):
                        st.markdown(f"⭐ **Puntuación usuarios:** {us:.1f}/100")
                    else:
                        st.markdown("⭐ **Puntuación usuarios:** N/D")

                    # 💰 Precio (si existe)
                    price = r.get("price")
                    if isinstance(price, (int, float)) and not np.isnan(price):
                        st.markdown(f"💰 **Precio:** {price:.2f} €")

                    # 🧠 Similitud FAISS
                    st.markdown(f"🧠 **Similitud (FAISS):** {r['score']:.3f}")

                    # Enlace externo (opcional)
                    st.markdown(
                        f"[Ver en Steam](https://store.steampowered.com/app/{r['appid']})",
                        unsafe_allow_html=True,
                    )

                    # Botón ver ficha interna
                    if st.button("🗂️ Ver ficha", key=f"btn-ficha-{r['appid']}"):
                        st.session_state["selected_appid"] = int(r["appid"])
                        st.switch_page("pages/03_Ficha_Juego.py")
