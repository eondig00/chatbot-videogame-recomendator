import streamlit as st
from src.llm.chat_agent import chat_recommend

# ================== CONFIG STREAMLIT ==================

st.set_page_config(
    page_title="Chat LLM · Recomendador de videojuegos",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 PixelSage")
st.caption("Escribe lo que buscas y el asistente te recomendará juegos.")

# ================== ESTILOS DE CHAT ==================

st.markdown(
    """
    <style>
    .chat-row {
        display: flex;
        margin: 0.25rem 0;
        width: 100%;
    }
    .chat-bubble {
        padding: 0.6rem 0.8rem;
        border-radius: 0.75rem;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.35;
        word-wrap: break-word;
    }
    .chat-bot {
        background-color: #111827;
        color: #f9fafb;
        border: 1px solid #020617;
        border-bottom-left-radius: 0.1rem;
    }
    .chat-user {
        background-color: #e5f0ff;
        color: #111827;
        border: 1px solid #bfdbfe;
        border-bottom-right-radius: 0.1rem;
    }
    .chat-meta {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-bottom: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== ESTADO DE SESIÓN ==================
# - chat_messages: [{"role": "user"/"assistant", "content": "..."}]
# - last_recs: lista de juegos candidatos de la última respuesta

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    # Mensaje inicial opcional
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": "¡Hola! Dime qué te apetece jugar y te recomiendo algo.",
        }
    )

if "last_recs" not in st.session_state:
    st.session_state.last_recs = []

# ================== HISTORIAL DE CONVERSACIÓN ==================

st.markdown("### 💬 Conversación")

for msg in st.session_state.chat_messages:
    if msg["role"] == "assistant":
        # Bot → izquierda
        st.markdown(
            f"""
            <div class="chat-row" style="justify-content: flex-start;">
              <div class="chat-bubble chat-bot">
                <div class="chat-meta">🤖 Recomendador</div>
                <div>{msg['content']}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Usuario → derecha
        st.markdown(
            f"""
            <div class="chat-row" style="justify-content: flex-end;">
              <div class="chat-bubble chat-user">
                <div class="chat-meta">😊 Usuario</div>
                <div>{msg['content']}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----- Botones "Ver ficha" para los últimos candidatos -----

last_recs = st.session_state.get("last_recs", [])
if last_recs:
    st.markdown("### 🎮 Juegos de esta recomendación")
    for r in last_recs[:3]:  # máximo 3 botones
        name = r.get("name", "Juego sin nombre")
        appid = r.get("appid")
        if appid is None:
            continue
        if st.button(f"🗂️ Ver ficha: {name}", key=f"chat-ficha-{appid}"):
            st.session_state["selected_appid"] = int(appid)
            st.switch_page("pages/03_Ficha_Juego.py")

# ================== INPUT DEL USUARIO ==================

user_input = st.chat_input("Escribe aquí tu mensaje sobre juegos")

if user_input:
    text = user_input.strip()
    if text:
        # 1) MOSTRAR mensaje del usuario INMEDIATAMENTE
        user_placeholder = st.empty()
        user_placeholder.markdown(
            f"""
            <div class="chat-row" style="justify-content: flex-end;">
              <div class="chat-bubble chat-user">
                <div class="chat-meta">😊 Usuario</div>
                <div>{text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2) Guardar mensaje del usuario en la conversación
        st.session_state.chat_messages.append({"role": "user", "content": text})

        # 3) Burbuja de "Escribiendo..." del bot
        bot_typing_placeholder = st.empty()
        bot_typing_placeholder.markdown(
            """
            <div class="chat-row" style="justify-content: flex-start;">
              <div class="chat-bubble chat-bot">
                <div class="chat-meta">🤖 Recomendador</div>
                <div><em>Escribiendo…</em></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4) Llamar al modelo y guardar candidatos
        try:
            result = chat_recommend(text)
            if isinstance(result, dict):
                answer = result.get("answer", "")
                st.session_state.last_recs = result.get("recs", []) or []
            else:
                # fallback por si algo devuelve solo texto
                answer = str(result)
                st.session_state.last_recs = []
        except Exception as e:
            answer = f"Ha habido un error llamando al modelo: {e}"
            st.session_state.last_recs = []

        # 5) Añadir respuesta del bot
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer.strip()}
        )

        # 6) Limpiar placeholders y refrescar
        user_placeholder.empty()
        bot_typing_placeholder.empty()
        st.rerun()

# ================== PANEL LATERAL ==================

with st.sidebar:
    st.header("ℹ️ Ayuda rápida")
    
    st.header("🎤 Tono del asistente")
    tono = st.selectbox(
        "Selecciona el tono:",
        ["normal", "amigo", "entusiasta", "formal", "seco", "periodista", "tiktoker"],
        index=0
    )
    st.session_state["tone"] = tono
    
    st.markdown(
        """
- Pide cosas tipo:
  - *"Quiero un RPG táctico corto"*
  - *"Algo como The Walking Dead pero sin tanto drama"*
  - *"Juegos de granja pero con algo de combate"*
  
Si quieres empezar de cero la conversación, pulsa abajo.
"""
    )

    if st.button("🧹 Reset conversación"):
        st.session_state.chat_messages = []
        st.session_state.last_recs = []
        st.success("Conversación reiniciada.")
        st.rerun()
