# src/llm/likes.py
from __future__ import annotations

from pathlib import Path
from typing import Set
import json

# Ruta única para los likes del usuario
LIKES_PATH = Path("data/user/likes.json")


class LikesStore:
    def __init__(self) -> None:
        # juegos con like “normal”
        self._liked: Set[int] = set()
        # juegos marcados como “favorito fuerte” (estrella)
        self._starred: Set[int] = set()
        self._load()

    # ---------- Carga / guardado en disco ----------

    def _load(self) -> None:
        if not LIKES_PATH.exists():
            self._liked = set()
            self._starred = set()
            return

        try:
            raw = LIKES_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)

            liked: Set[int] = set()
            starred: Set[int] = set()

            # Formato nuevo: {"liked": [...], "starred": [...]}
            if isinstance(data, dict) and (
                "liked" in data or "starred" in data
            ):
                liked = {int(x) for x in data.get("liked", [])}
                starred = {int(x) for x in data.get("starred", [])}

            # Formato viejo 1: [123, 456, ...]  -> sólo liked
            elif isinstance(data, list):
                liked = {int(x) for x in data}
                starred = set()

            # Formato viejo 2: {"123": {"appid": 123, "starred": true}, ...}
            elif isinstance(data, dict):
                for k, v in data.items():
                    try:
                        appid = int(v.get("appid", k))
                    except Exception:
                        continue
                    liked.add(appid)
                    if bool(v.get("starred", False)):
                        starred.add(appid)

            self._liked = liked
            self._starred = starred

        except Exception:
            # si está roto, empezamos de cero
            self._liked = set()
            self._starred = set()

    def _save(self) -> None:
        LIKES_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "liked": sorted(self._liked),
            "starred": sorted(self._starred),
        }
        LIKES_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- API pública usada por las páginas ----------

    def is_liked(self, appid: int) -> bool:
        return appid in self._liked

    def is_starred(self, appid: int) -> bool:
        return appid in self._starred

    def like(self, appid: int) -> None:
        self._liked.add(appid)
        self._save()

    def unlike(self, appid: int) -> None:
        self._liked.discard(appid)
        # si quitas el like, también tiene sentido quitar el starred
        self._starred.discard(appid)
        self._save()

    def toggle_star(self, appid: int) -> bool:
        """
        Alterna estado de “favorito fuerte”.
        Devuelve True si queda marcado, False si se desmarca.
        """
        if appid in self._starred:
            self._starred.remove(appid)
            self._save()
            return False

        # al marcar como favorito fuerte, nos aseguramos de que tenga like
        self._liked.add(appid)
        self._starred.add(appid)
        self._save()
        return True

    def all_likes(self) -> Set[int]:
        """Conjunto de appids con like (independientemente de star)."""
        return set(self._liked)

    def all_starred(self) -> Set[int]:
        """Conjunto de appids marcados como favorito fuerte."""
        return set(self._starred)

    # Aliases para compatibilidad si en algún sitio usas estos nombres:

    def has(self, appid: int) -> bool:
        return self.is_liked(appid)

    def all(self) -> Set[int]:
        return self.all_likes()


# Singleton global
LIKES = LikesStore()
