# streamlit_app.py
import streamlit as st

from src.llm.chat_agent import chat_recommend

# ================== CONFIG STREAMLIT ==================

st.set_page_config(
    page_title="Chat LLM · Recomendador de videojuegos",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Chat LLM recomendador de videojuegos")
st.caption("Escribe lo que buscas y el asistente te recomendará juegos usando el catálogo FAISS.")


# ================== ESTADO DE SESIÓN ==================

if "chat_messages" not in st.session_state:
    # Guardamos solo lo que se muestra en la UI
    st.session_state.chat_messages = []   # lista de dicts: {"role": "user"/"assistant", "content": "..."}


# ================== INTERFAZ DE CHAT ==================

# Mostramos historial
for msg in st.session_state.chat_messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

# Input del usuario (nuevo mensaje)
user_input = st.chat_input("Escribe aquí tu mensaje sobre juegos (ej. 'Quiero un RPG como Dragon Age pero corto')")

if user_input:
    # Añadimos el mensaje del usuario al historial
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Llamamos a tu agente LLM
    with st.chat_message("assistant"):
        with st.spinner("Pensando recomendaciones..."):
            try:
                answer = chat_recommend(user_input)
            except Exception as e:
                answer = f"Ha habido un error llamando al modelo: {e}"

            st.write(answer)

    # Guardamos respuesta en el historial
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ================== PANEL LATERAL ==================

with st.sidebar:
    st.header("ℹ️ Ayuda rápida")
    st.markdown(
        """
- Pide cosas tipo:
  - *"Quiero un RPG táctico corto"*
  - *"Algo como The Walking Dead pero sin tanto drama"*
  - *"Juegos de granja pero con algo de combate"*
- El modelo usa:
  - embeddings + FAISS para buscar candidatos
  - tu función `chat_recommend()` para generar la respuesta.

Si quieres empezar de cero la conversación, pulsa abajo.
"""
    )

    if st.button("🧹 Reset conversación"):
        st.session_state.chat_messages = []
        st.success("Conversación reiniciada.")
