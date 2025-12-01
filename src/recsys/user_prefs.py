from __future__ import annotations

from dataclasses import dataclass
from src.llm.memory import GLOBAL_USER_MEMORY


@dataclass
class UserPrefs:
    liked_genres: list[str]
    disliked_genres: list[str]
    avoid_tags: list[str]
    min_user_score: float
    min_num_reviews: int
    max_price: float | None
    avoid_nsfw: bool

    def to_dict(self) -> dict:
        return {
            "liked_genres": self.liked_genres,
            "disliked_genres": self.disliked_genres,
            "avoid_tags": self.avoid_tags,
            "min_user_score": self.min_user_score,
            "min_num_reviews": self.min_num_reviews,
            "max_price": self.max_price,
            "avoid_nsfw": self.avoid_nsfw,
        }

    def to_prompt_summary(self) -> dict:
        return self.to_dict()


def default_prefs() -> UserPrefs:
    """Solo se usa si algo falla muy fuerte; normalmente no debería llamarse."""
    return UserPrefs(
        liked_genres=[],
        disliked_genres=[],
        avoid_tags=["Horror", "Gore"],
        min_user_score=0.0,
        min_num_reviews=0,
        max_price=None,
        avoid_nsfw=True,
    )


def load_user_prefs() -> UserPrefs:
    """
    ⚠️ IMPORTANTE:
    Ahora NO leemos de un JSON propio.
    Delegamos SIEMPRE en GLOBAL_USER_MEMORY.
    """
    try:
        mem_prefs = GLOBAL_USER_MEMORY.get_explicit_prefs()
        # mem_prefs ya es un UserPrefs del módulo de memoria,
        # pero tiene los mismos campos, así que lo “clonamos” aquí.
        return UserPrefs(
            liked_genres=list(mem_prefs.liked_genres),
            disliked_genres=list(mem_prefs.disliked_genres),
            avoid_tags=list(mem_prefs.avoid_tags),
            min_user_score=float(mem_prefs.min_user_score),
            min_num_reviews=int(mem_prefs.min_num_reviews),
            max_price=mem_prefs.max_price,
            avoid_nsfw=bool(mem_prefs.avoid_nsfw),
        )
    except Exception:
        return default_prefs()


def save_user_prefs(prefs: UserPrefs) -> None:
    """
    Si en algún sitio del código se llama a esto, también delegamos en GLOBAL_USER_MEMORY.
    """
    GLOBAL_USER_MEMORY.set_explicit_prefs(
        liked_genres=prefs.liked_genres,
        disliked_genres=prefs.disliked_genres,
        avoid_tags=prefs.avoid_tags,
        min_user_score=prefs.min_user_score,
        min_num_reviews=prefs.min_num_reviews,
        max_price=prefs.max_price,
        avoid_nsfw=prefs.avoid_nsfw,
    )
