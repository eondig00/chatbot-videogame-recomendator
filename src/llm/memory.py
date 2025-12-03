from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import json

from src.utils.config import load_config


# =============== PREFERENCIAS EXPLÍCITAS ===============

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
        # versión compacta para meter en el prompt del LLM
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPrefs":
        return cls(
            liked_genres=list(data.get("liked_genres", [])),
            disliked_genres=list(data.get("disliked_genres", [])),
            avoid_tags=list(data.get("avoid_tags", [])),
            min_user_score=float(data.get("min_user_score", 0.0)),
            min_num_reviews=int(data.get("min_num_reviews", 0)),
            max_price=data.get("max_price", None),
            avoid_nsfw=bool(data.get("avoid_nsfw", True)),
        )


# =============== MEMORIA DEL USUARIO ===============

class UserMemory:
    """
    Resumen de gustos + preferencias explícitas del usuario.
    - ultimas_consultas: queries de texto recientes.
    - generos_frecuentes, tags_frecuentes: contadores ligeros.
    - explicit_prefs: lo que el usuario fija a mano (página de preferencias).
    """

    def __init__(self) -> None:
        cfg = load_config()

        # ❗ USAR SIEMPRE la ruta del config.yml
        #    paths.user_prefs: data/user_prefs/user_prefs.json
        self._prefs_file = Path(cfg["paths"]["user_prefs"])
        self._prefs_file.parent.mkdir(parents=True, exist_ok=True)

        # Inicializamos memoria interna
        self.ultimas_consultas: List[str] = []
        self.generos_frecuentes: List[str] = []
        self.tags_frecuentes: List[str] = []
        self.num_juegos_vistos: int = 0
        
        self.tone: str = "normal"

        # Cargar preferencias guardadas
        self.explicit_prefs: UserPrefs = self._load_prefs_from_disk()
        self.tone: UserPrefs = self._load_prefs_from_disk()

    # ---------- CARGA / GUARDADO ----------

    def _load_prefs_from_disk(self) -> UserPrefs:
        if not self._prefs_file.exists():
            return UserPrefs(
                liked_genres=[],
                disliked_genres=[],
                avoid_tags=["Horror", "Gore"],
                min_user_score=0.0,
                min_num_reviews=0,
                max_price=None,
                avoid_nsfw=True,
            ), "normal"

        try:
            raw = self._prefs_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            return UserPrefs.from_dict(data)
        except Exception:
            return UserPrefs(
                liked_genres=[],
                disliked_genres=[],
                avoid_tags=["Horror", "Gore"],
                min_user_score=0.0,
                min_num_reviews=0,
                max_price=None,
                avoid_nsfw=True,
            )

    def _save_prefs_to_disk(self) -> None:
        self._prefs_file.write_text(
            json.dumps(self.explicit_prefs.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- API PARA EL RECOMENDADOR / LLM ----------

    def update_from_query(self, query: str) -> None:
        if not query:
            return
        self.ultimas_consultas.append(query)
        self.ultimas_consultas = self.ultimas_consultas[-20:]

    def update_from_results(self, recs: List[Dict[str, Any]]) -> None:
        """
        Actualización ligera y segura de géneros y tags.
        Evita meter cadenas mal formadas como "['Indie'" o "[]".
        """
        import json

        for r in recs:
            self.num_juegos_vistos += 1

            # --- géneros ---
            genres = r.get("genres") or []
            if isinstance(genres, str):
                # puede venir como "['RPG', 'Action']"
                if genres.startswith("[") and genres.endswith("]"):
                    try:
                        genres = json.loads(genres)
                    except Exception:
                        genres = [genres]
                else:
                    genres = [genres]

            for g in genres:
                g_str = str(g).strip()
                if g_str and g_str != "[]":
                    self.generos_frecuentes.append(g_str)

            # --- tags ---
            tags = r.get("tags") or []
            if isinstance(tags, str):
                if tags.startswith("[") and tags.endswith("]"):
                    try:
                        tags = json.loads(tags)
                    except Exception:
                        tags = [tags]
                else:
                    tags = [tags]

            for t in tags:
                t_str = str(t).strip()
                if t_str and t_str != "[]":
                    self.tags_frecuentes.append(t_str)

        # limitar memoria para no crecer infinito
        self.generos_frecuentes = self.generos_frecuentes[-40:]
        self.tags_frecuentes = self.tags_frecuentes[-40:]

    
    def to_prompt(self) -> Dict[str, Any]:
        return {
            "ultimas_consultas": self.ultimas_consultas[-5:],
            "generos_frecuentes": self.generos_frecuentes[-8:],
            "tags_frecuentes": self.tags_frecuentes[-8:],
            "num_juegos_vistos": self.num_juegos_vistos,
            "explicit_prefs": self.explicit_prefs.to_prompt_summary(),
        }

    # ---------- API PARA LA PÁGINA DE PREFERENCIAS ----------

    def get_explicit_prefs(self) -> UserPrefs:
        return self.explicit_prefs

    def set_explicit_prefs(
        self,
        liked_genres: list[str],
        disliked_genres: list[str],
        avoid_tags: list[str],
        min_user_score: float,
        min_num_reviews: int,
        max_price: float | None,
        avoid_nsfw: bool,
    ) -> None:
        self.explicit_prefs = UserPrefs(
            liked_genres=liked_genres,
            disliked_genres=disliked_genres,
            avoid_tags=avoid_tags,
            min_user_score=min_user_score,
            min_num_reviews=min_num_reviews,
            max_price=max_price,
            avoid_nsfw=avoid_nsfw,
        )
        self._save_prefs_to_disk()


# Singleton global a importar desde otros módulos
GLOBAL_USER_MEMORY = UserMemory()
