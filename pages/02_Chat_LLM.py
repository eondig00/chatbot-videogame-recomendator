# pages/02_Chat.py
import streamlit as st
from src.llm.chat_agent import chat_recommend

st.title("🧠 Asistente LLM (Ollama)")

if "chat" not in st.session_state:
    st.session_state["chat"] = []

# Mostrar historial
for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("¿Qué tipo de juego buscas?")

if prompt:
    # Usuario
    st.session_state["chat"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LLM (Ollama)
    with st.chat_message("assistant"):
        with st.spinner("Buscando recomendaciones…"):
            answer = chat_recommend(prompt, model="llama3.2:1b")
            st.markdown(answer)

    st.session_state["chat"].append({"role": "assistant", "content": answer})
