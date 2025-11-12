import streamlit as st
from src.recsys.recommender import recommend_by_text

st.set_page_config(page_title="🎮 Recomendador", page_icon="🎮")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Cuéntame qué te apetece jugar."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ej: roguelike corto, cooperativo, pixel art..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            recs = recommend_by_text(prompt, k=10)
            resp = "\n\n".join([f"**{r['name']}**\n{r['why']}" for r in recs])
            st.markdown(resp)
            st.session_state.messages.append({"role":"assistant","content":resp})
