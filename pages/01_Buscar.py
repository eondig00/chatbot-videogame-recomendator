# pages/01_Buscar.py
import numpy as np
import streamlit as st
from src.recsys.recommender import recommend_by_text

st.set_page_config(page_title="🔎 Buscar juegos", layout="wide")
st.title("🔎 Buscar juegos por descripción")

# -------------------------
# Entrada del usuario
# -------------------------
q = st.text_input("¿Qué te apetece?", placeholder="Ej.: roguelike deckbuilder de partidas cortas, cooperativo, pixel art…")
k = st.number_input("Número de resultados", 3, 50, 10, 1)

# -------------------------
# Búsqueda
# -------------------------
f_excluir = st.checkbox("Excluir contenido adulto (NSFW)", value=True)

if st.button("Buscar", type="primary") and q.strip():
    recs = recommend_by_text(q.strip(), int(k), exclude_nsfw=f_excluir)
    if not recs:
        st.info("Sin resultados.")
    for r in recs:
        with st.container(border=True):
            cols = st.columns([1, 3])
            with cols[0]:
                if r.get("header_image"):
                    st.image(r["header_image"], width="stretch")
            with cols[1]:
                st.markdown(f"### {r['name']}")

                # ---- SCORE DE USUARIOS ----
                us = r.get("user_score")
                if us is not None and not (isinstance(us, float) and np.isnan(us)):
                    st.markdown(f"⭐ **Puntuación usuarios:** {us:.1f}/100")
                else:
                    st.markdown("⭐ **Puntuación usuarios:** N/D")

                # ---- SIMILITUD FAISS ----
                st.markdown(f"🧠 **Similitud (FAISS):** {r['score']:.3f}")

                # Enlace externo a Steam (por si quieres mantenerlo)
                st.markdown(
                    f"[Ver en Steam](https://store.steampowered.com/app/{r['appid']})",
                    unsafe_allow_html=True,
                )

                # Botón para ir a la ficha dentro de tu app (lo vemos en el siguiente punto)
                if st.button("Ver ficha", key=f"btn-ficha-{r['appid']}"):
                    st.session_state["selected_appid"] = int(r["appid"])
                    st.switch_page("pages/04_Ficha_Juego.py")
