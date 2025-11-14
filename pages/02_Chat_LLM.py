import streamlit as st

st.title("💬 Chat (LLM – placeholder)")
st.caption("Más adelante, aquí enchufaremos un LLM. Hoy, devolvemos eco + recomendación sugerida.")

if "chat" not in st.session_state:
    st.session_state.chat = []

msg = st.chat_input("Escribe tu mensaje")
if msg:
    st.session_state.chat.append(("user", msg))
    # Placeholder de respuesta (sin LLM aún)
    st.session_state.chat.append(("assistant", f"Entendido: **{msg}**. Próximo paso: integrar LLM + RAG con tu índice."))

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
