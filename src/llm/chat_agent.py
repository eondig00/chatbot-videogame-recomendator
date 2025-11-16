# src/llm/chat_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import ollama

from src.recsys.recommender import recommend_by_text


@dataclass
class Plan:
    search_query: str
    top_k: int = 10
    exclude_nsfw: bool = True


# ------------------ PROMPTS ------------------

PLANNER_SYSTEM = """Eres un planificador de consultas para un recomendador \
de videojuegos basado en embeddings y FAISS.

Tu tarea es transformar la petición del usuario en un JSON con:
{
  "search_query": "...",
  "top_k": número entre 5 y 20,
  "exclude_nsfw": true/false
}

No respondas con texto. Solo devuelve JSON válido.
Si el usuario pide explícitamente hentai/adulto, poner exclude_nsfw = false.
"""


ANSWER_SYSTEM = """Eres un asistente experto en videojuegos. \
Recibirás una lista de juegos recomendados por FAISS.

Tu tarea:
- Explicar por qué esos juegos encajan con lo pedido.
- Ser claro, directo y útil.
- No recomendar NSFW a menos que el usuario lo pida explícitamente.
- Formato: lista numerada con nombre, motivo y enlace de Steam.

Habla siempre en español.
"""


# ----------- HELPER OLLAMA -------------------

def _ollama_chat(model: str, messages: List[Dict[str, str]]) -> str:
    """Wrapper uniforme para ollama.chat()"""
    res = ollama.chat(model=model, messages=messages)
    return res["message"]["content"]


# ---------- 1) Planificación de búsqueda ----------

def plan_recommendation(user_message: str, model: str = "llama3.1:8b") -> Plan:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_message}
    ]

    raw = _ollama_chat(model, messages)

    try:
        data = json.loads(raw)
    except Exception:
        return Plan(search_query=user_message, top_k=10, exclude_nsfw=True)

    search_query = data.get("search_query") or user_message
    top_k = max(5, min(int(data.get("top_k", 10)), 20))
    exclude = bool(data.get("exclude_nsfw", True))

    return Plan(search_query=search_query, top_k=top_k, exclude_nsfw=exclude)


# ---------- 2) Orquestación completa ----------

def chat_recommend(user_message: str, model: str = "llama3.1:8b") -> str:
    plan = plan_recommendation(user_message, model=model)

    # FAISS
    recs = recommend_by_text(
        plan.search_query,
        k=plan.top_k,
        exclude_nsfw=plan.exclude_nsfw
    )

    # Solo enviamos info esencial al LLM
    recs_for_llm = [
        {
            "appid": r["appid"],
            "name": r["name"],
            "short_description": r.get("short_description", ""),
            "user_score": r.get("user_score"),
            "price": r.get("price"),
            "genres": r.get("genres", []),
            "tags": r.get("tags", [])
        }
        for r in recs
    ]

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user_message},
        {"role": "assistant",
         "content": "Juegos candidatos recomendados por FAISS:\n" +
                    json.dumps(recs_for_llm, ensure_ascii=False)}
    ]

    answer = _ollama_chat(model, messages)
    return answer
