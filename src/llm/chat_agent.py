from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import pprint

import ollama

from src.recsys.recommender import recommend_by_text
from src.llm.memory import UserMemory


# ================== ESTADO GLOBAL ==================

USER_MEMORY = UserMemory()                # gustos largos del usuario
DEFAULT_MODEL = "llama3.2:3b"             # modelo por defecto para Ollama

# Historial de conversación (solo mensajes reales user/assistant)
CHAT_HISTORY: List[Dict[str, str]] = []
MAX_HISTORY_MESSAGES = 10                 # últimos mensajes a incluir (5 turnos user+assistant)


@dataclass
class Plan:
    search_query: str
    top_k: int = 8
    exclude_nsfw: bool = True


# ================== PROMPTS ==================

PLANNER_SYSTEM = """
Eres un planificador de consultas para un recomendador de videojuegos basado en embeddings.

Devuelves SOLO JSON:

{
  "search_query": "...",
  "top_k": 10,
  "exclude_nsfw": true
}

Reglas rápidas:
- top_k entre 5 y 20.
- exclude_nsfw = true por defecto.
- Si no va de videojuegos → search_query = "general" y top_k = 5.
- Si pide hentai/adulto → exclude_nsfw = false.
"""

ANSWER_SYSTEM = """
Eres un asistente experto en videojuegos.  
Responde siempre de forma directa, clara y usando SOLO los juegos del catálogo candidato.

REGLAS PRINCIPALES:
- SOLO puedes recomendar juegos cuyos nombres estén en la lista de títulos permitidos.  
- No inventes juegos, datos, enlaces ni descripciones.  
- Máximo 2 juegos por respuesta.  
- Tono natural, frases cortas, tuteando.  
- No repitas recomendaciones que el usuario ya ha descartado.  
- No muestres el catálogo ni el resumen interno.  
- Responde siempre en el idioma del usuario.

CUANDO EL MENSAJE ES SOBRE VIDEOJUEGOS:
1. Resume en una frase lo que busca el usuario (si es necesario).  
2. Si falta un detalle clave (género, duración, dificultad), pregunta SOLO una cosa breve.  
3. Elige 1–2 juegos del catálogo que mejor encajen.  
4. Explica brevemente:
   - qué tipo de juego es
   - por qué encaja con lo que busca
   - si parece corto/largo/fácil/difícil solo si se puede deducir (no inventes horas)

CUANDO EL MENSAJE NO ES DE VIDEOJUEGOS:
- Responde corto y natural.  
- Si tiene sentido, ofrece seguir con recomendaciones.

SIN CANDIDATOS EN EL CATÁLOGO:
- Di que no tienes buenos candidatos con esa descripción.  
- Pide que concrete más (género, ambientación, duración, tono).

PREFERENCIAS Y DESCARTES:
- Si el usuario descarta un tipo de juego (ej. soulslike, terror, roguelite), evítalo en el resto de la conversación.

GLOSARIO RÁPIDO (para entender intención):
- telltale-like: aventura narrativa con decisiones y poca acción.  
- soulslike: acción exigente estilo Dark Souls.  
- roguelike/roguelite: runs con muerte frecuente y algo de aleatoriedad.  
- CRPG: rol clásico con decisiones y party.  
- metroidvania: exploración y habilidades que desbloquean zonas.  
- walking simulator: narrativa y exploración, casi sin mecánicas.  
- farming sim: gestión de granja/vida en pueblo.

REGLAS DURAS:
- Solo usa juegos del catálogo candidato.  
- No nombres NADA que no esté en la lista permitida.  
- Máximo 2 juegos.  
- No inventes.

Tu prioridad: ser útil, concreto y preciso usando únicamente los juegos del catálogo.
"""


# ================== WRAPPER OLLAMA ==================

def _ollama_chat(model: str, messages: List[Dict[str, str]]) -> str:
    """Wrapper sencillo para ollama.chat() que devuelve sólo el 'content'."""
    # Debug opcional:
    # print("\n=== MENSAJES ENVIADOS A OLLAMA ===")
    # pprint.pprint(messages)
    # print("==================================\n")

    res = ollama.chat(model=model, messages=messages)
    return res["message"]["content"]


# ================== HELPERS ==================

def _build_allowed_titles_text(recs: List[Dict[str, Any]]) -> str:
    """
    Construye un mensaje de sistema que lista explícitamente
    qué juegos puede mencionar el LLM.
    Soporta formatos:
      - {"name": "...", ...}
      - {"game": {"name": "...", ...}, ...}
    """
    titulos: List[str] = []

    for r in recs:
        if "game" in r and isinstance(r["game"], dict):
            name = r["game"].get("name")
        else:
            name = r.get("name")
        if name:
            titulos.append(str(name))

    titulos_unicos = sorted(set(titulos))

    if not titulos_unicos:
        return (
            "IMPORTANTE (regla dura): no tienes ningún juego candidato. "
            "No inventes títulos y di que no tienes buenos candidatos."
        )

    joined = ", ".join(titulos_unicos)

    return (
        "IMPORTANTE (regla dura): "
        "Solo puedes mencionar juegos cuyos nombres estén en esta lista. "
        "NO menciones ningún otro título, aunque lo conozcas: "
        f"{joined}."
    )


# ================== PLANIFICACIÓN ==================

def plan_recommendation(user_message: str, model: str | None = None) -> Plan:
    model = model or DEFAULT_MODEL

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    # Debug:
    print("\n=== MENSAJES ENVIADOS A OLLAMA (PLANNER) ===")
    pprint.pprint(messages)
    print("===========================================\n")

    raw = _ollama_chat(model, messages)

    # Si el modelo se lía, hacemos un fallback razonable
    try:
        data = json.loads(raw)
    except Exception:
        return Plan(search_query=user_message, top_k=8, exclude_nsfw=True)

    search_query = data.get("search_query") or user_message

    try:
        top_k = int(data.get("top_k", 8))
    except Exception:
        top_k = 8
    top_k = max(5, min(top_k, 20))

    exclude = bool(data.get("exclude_nsfw", True))

    return Plan(search_query=search_query, top_k=top_k, exclude_nsfw=exclude)


# ================== ORQUESTACIÓN COMPLETA ==================

def _build_catalogo_text(recs: List[Dict[str, Any]]) -> str:
    """
    Transformamos las recomendaciones crudas de FAISS en un texto compacto
    de contexto para el LLM (NO se enseña al usuario).
    """
    if not recs:
        return "Sin candidatos claros."

    lineas = []
    for r in recs[:6]:   # como mucho 6 candidatos en el contexto
        # Soportar formato {"game": {...}} o plano
        base = r.get("game", r)

        appid = base.get("appid", "desconocido")
        name = base.get("name", "Juego sin nombre")
        genres = ", ".join(base.get("genres", [])[:3]) if base.get("genres") else ""
        score = base.get("user_score")

        parts = [name]
        if genres:
            parts.append(f"géneros: {genres}")
        if score is not None:
            parts.append(f"nota usuarios ~{score:.1f}/10")
        parts.append(f"APPID={appid}")
        lineas.append(" - " + " | ".join(parts))

    return "\n".join(lineas)


def chat_recommend(user_message: str, model: str | None = None) -> str:
    global CHAT_HISTORY

    model = model or DEFAULT_MODEL

    # 1) Actualizamos memoria de gustos con la nueva consulta
    USER_MEMORY.update_from_query(user_message)

    # 2) Planificamos la búsqueda semántica
    plan = plan_recommendation(user_message, model=model)

    # 3) Consultamos FAISS
    recs = recommend_by_text(
        plan.search_query,
        k=plan.top_k,
        exclude_nsfw=plan.exclude_nsfw,
    )

    # 4) Actualizamos memoria con los resultados sugeridos
    USER_MEMORY.update_from_results(recs)

    # 5) Preparamos contexto compacto para el LLM
    catalogo_text = _build_catalogo_text(recs)
    memory_summary = json.dumps(USER_MEMORY.to_prompt(), ensure_ascii=False)

    # Regla dura de títulos permitidos
    allowed_titles_text = _build_allowed_titles_text(recs)

    mensajes: List[Dict[str, str]] = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "system", "content": allowed_titles_text},
        {
            "role": "system",
            "content": f"Metadatos internos: search_query={plan.search_query!r}, exclude_nsfw={plan.exclude_nsfw}"
        },
        {
            "role": "system",
            "content": "PREFERENCIAS DEL USUARIO (RESUMEN INTERNO):\n" + memory_summary
        },
        {
            "role": "system",
            "content": "CATÁLOGO CANDIDATO (NO LO MUESTRES TAL CUAL, ES SOLO CONTEXTO):\n" + catalogo_text
        },
    ]

    # Historial corto (solo conversación real)
    if CHAT_HISTORY:
        mensajes.extend(CHAT_HISTORY[-MAX_HISTORY_MESSAGES:])

    # Mensaje actual del usuario
    mensajes.append({"role": "user", "content": user_message})

    # Debug:
    print("\n=== MENSAJES ENVIADOS A OLLAMA (CHAT) ===")
    pprint.pprint(mensajes)
    print("=========================================\n")

    respuesta = _ollama_chat(model, mensajes)

    # Actualizamos historial
    CHAT_HISTORY.append({"role": "user", "content": user_message})
    CHAT_HISTORY.append({"role": "assistant", "content": respuesta})

    # Recorte de historial para que no crezca infinito
    if len(CHAT_HISTORY) > 4 * MAX_HISTORY_MESSAGES:
        CHAT_HISTORY = CHAT_HISTORY[-2 * MAX_HISTORY_MESSAGES:]

    return respuesta
