# pages/01_Buscar.py
import streamlit as st
from src.recsys.recommender import recommend_by_text

st.title("🔎 Buscar juegos por descripción")

q = st.text_input("¿Qué te apetece?", placeholder="Ej.: roguelike deckbuilder de partidas cortas…")
k = st.number_input("Resultados", 3, 30, 10)

if st.button("Buscar", type="primary") and q.strip():
    recs = recommend_by_text(q.strip(), int(k))
    if not recs:
        st.info("Sin resultados.")
    for r in recs:
        with st.container(border=True):
            cols = st.columns([1,3])
            with cols[0]:
                if r.get("header_image"):
                    st.image(r["header_image"], width="stretch")
            with cols[1]:
                st.markdown(f"### [{r['name']}](https://store.steampowered.com/app/{r['appid']})")
                st.markdown(f"**Similitud (FAISS):** {r['score']:.3f}")
