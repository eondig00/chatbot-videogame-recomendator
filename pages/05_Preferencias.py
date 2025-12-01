import streamlit as st
from src.llm.memory import GLOBAL_USER_MEMORY

st.set_page_config(page_title="🎯 Perfil del jugador", layout="centered")
st.title("🎯 Perfil del jugador")

prefs = GLOBAL_USER_MEMORY.get_explicit_prefs()

# ---- Defaults seguros desde memoria ----
liked_default = ", ".join(prefs.liked_genres)
disliked_default = ", ".join(prefs.disliked_genres)
avoid_tags_default = ", ".join(prefs.avoid_tags)

# Nota mínima: guardada como float, pero el slider será int 0–100
min_score_default = int(prefs.min_user_score or 0)

# Reseñas mínimas
min_reviews_default = int(prefs.min_num_reviews or 0)

# max_price puede ser None → mostramos 0 como “sin límite”
max_price_default = float(prefs.max_price) if prefs.max_price is not None else 0.0

with st.form("perfil"):
    st.subheader("🎮 Géneros")

    liked = st.text_input(
        "Géneros que te gustan",
        liked_default,
        help="Ej.: RPG, Strategy, City Builder"
    )

    disliked = st.text_input(
        "Géneros que NO quieres ver",
        disliked_default,
        help="Ej.: Soulslike, Roguelite"
    )

    st.subheader("🏷️ Tags a evitar")

    avoid_tags = st.text_input(
        "Tags que prefieres evitar",
        avoid_tags_default,
        help="Ej.: Horror, Gore"
    )

    st.subheader("⭐ Filtros de calidad y precio")

    # 🔧 Slider todo en int (min_value, max_value, step, value)
    min_score = st.slider(
        "Nota mínima de usuarios",
        min_value=0,
        max_value=100,
        step=1,
        value=min_score_default,
    )

    min_reviews = st.number_input(
        "Reseñas mínimas",
        min_value=0,
        step=10,
        value=min_reviews_default,
    )

    # Aquí todo float: min_value, step, value
    max_price = st.number_input(
        "Precio máximo (0 = sin límite)",
        min_value=0.0,
        step=1.0,
        value=max_price_default,
    )

    avoid_nsfw = st.checkbox(
        "Evitar contenido adulto (NSFW)",
        value=prefs.avoid_nsfw,
    )

    save = st.form_submit_button("💾 Guardar")

    if save:
        GLOBAL_USER_MEMORY.set_explicit_prefs(
            liked_genres=[s.strip() for s in liked.split(",") if s.strip()],
            disliked_genres=[s.strip() for s in disliked.split(",") if s.strip()],
            avoid_tags=[s.strip() for s in avoid_tags.split(",") if s.strip()],
            min_user_score=float(min_score),
            min_num_reviews=int(min_reviews),
            max_price=float(max_price) if max_price > 0 else None,
            avoid_nsfw=avoid_nsfw,
        )
        st.success("Preferencias guardadas.")
        st.rerun()
