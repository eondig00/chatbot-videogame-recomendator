# pages/02_Chat_LLM.py (idea)
import streamlit as st
from src.llm.chat_agent import chat_recommend

st.title("💬 Asistente LLM de videojuegos")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("¿Qué te apetece jugar?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = chat_recommend(prompt)   # aquí se usa lo nuevo
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
