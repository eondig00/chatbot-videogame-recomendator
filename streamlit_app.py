import streamlit as st
from pathlib import Path
import pandas as pd

from src.llm.chat_agent import chat_recommend

# ================== CONFIG STREAMLIT ==================

st.set_page_config(
    page_title="Chat LLM · Recomendador de videojuegos",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 PixelSage")
st.caption("Escribe lo que buscas y el asistente te recomendará juegos.")

# ================== CARGA DE DATOS ==================

@st.cache_resource
def load_games_df():
    data_path = Path("data/processed/games.parquet")
    return pd.read_parquet(data_path)

GAMES_DF = load_games_df()

def get_game_row(appid: int):
    df = GAMES_DF
    row = df.loc[df["appid"] == appid]
    if row.empty:
        return None
    return row.iloc[0]

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
# - chat_messages: [{"role": "...", "content": "...", "games": [ {appid, name...}, ...]}]

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": "¡Hola! Dime qué te apetece jugar y te recomiendo algo.",
            "games": [],
        }
    )

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

        # Fichas de juegos mencionados explícitamente en esa respuesta
        games = msg.get("games") or []
        for g in games:
            appid = g.get("appid")
            if appid is None:
                continue
            row = get_game_row(int(appid))
            if row is None:
                continue

            with st.container():
                cols = st.columns([1, 2])
                with cols[0]:
                    img = row.get("header_image")
                    if isinstance(img, str) and img.startswith(("http://", "https://")):
                        st.image(img, width='content')
                with cols[1]:
                    st.markdown(f"**{row.get('name', 'Sin nombre')}**")

                    us = row.get("user_score")
                    if isinstance(us, (int, float)):
                        st.caption(f"⭐ {us:.1f}/100")

                    price = row.get("price")
                    if isinstance(price, (int, float)):
                        st.caption(f"💰 {price:.2f} €")

                    short_desc = row.get("short_description") or ""
                    if short_desc:
                        st.write((short_desc[:180] + "…").strip())

                    if st.button("🗂️ Ver ficha completa", key=f"chat-ficha-{appid}"):
                        st.session_state["selected_appid"] = int(appid)
                        st.switch_page("pages/03_Ficha_Juego.py")

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
        st.session_state.chat_messages.append(
            {"role": "user", "content": text, "games": []}
        )

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

        # 4) Llamar al modelo
        try:
            result = chat_recommend(text)
            if isinstance(result, dict):
                answer = result.get("answer", "") or ""
                candidate_recs = result.get("recs", []) or []
            else:
                answer = str(result)
                candidate_recs = []
        except Exception as e:
            answer = f"Ha habido un error llamando al modelo: {e}"
            candidate_recs = []

        # 5) Detectar qué juegos han sido mencionados explícitamente por nombre
        mentioned_games = []
        ans_lower = answer.lower()
        for r in candidate_recs:
            name = str(r.get("name") or "").strip()
            if name and name.lower() in ans_lower:
                mentioned_games.append(r)

        # 6) Añadir respuesta del bot con las fichas asociadas
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer.strip(),
                "games": mentioned_games,
            }
        )

        # 7) Limpiar placeholders y refrescar
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
        index=0,
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
        st.success("Conversación reiniciada.")
        st.rerun()
