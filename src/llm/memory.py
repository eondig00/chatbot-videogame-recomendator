from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import collections
import re
import math


_token_re = re.compile(r"[A-Za-z0-9ñáéíóúüç+#]+")


def _tokens(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t.lower() for t in _token_re.findall(text)]


@dataclass
class UserMemory:
    """
    Memoria muy simple de preferencias del usuario para usar
    como contexto del LLM (no persiste en disco, solo en sesión).
    """
    genre_counts: collections.Counter = field(default_factory=collections.Counter)
    tag_counts: collections.Counter = field(default_factory=collections.Counter)
    last_queries: List[str] = field(default_factory=list)
    seen_appids: List[int] = field(default_factory=list)
    max_queries: int = 10
    max_seen: int = 30

    # --------- Entrada: texto del usuario ---------
    def update_from_query(self, text: str) -> None:
        if not text:
            return
        self.last_queries.append(text)
        if len(self.last_queries) > self.max_queries:
            self.last_queries = self.last_queries[-self.max_queries :]

    # --------- Entrada: resultados de FAISS ---------
    def update_from_results(self, recs: List[Dict[str, Any]]) -> None:
        if not recs:
            return

        for r in recs[:5]:  # solo miramos los primeros candidatos
            appid = int(r.get("appid", 0) or 0)
            if appid and appid not in self.seen_appids:
                self.seen_appids.append(appid)

            # géneros y tags
            for g in r.get("genres", []) or []:
                self.genre_counts[g] += 1
            for t in r.get("tags", []) or []:
                self.tag_counts[t] += 1

        if len(self.seen_appids) > self.max_seen:
            self.seen_appids = self.seen_appids[-self.max_seen :]

    # --------- Salida: resumen para el prompt ---------
    def _top_pref(self, counter: collections.Counter, n: int = 5) -> List[str]:
        return [name for name, _ in counter.most_common(n)]

    def to_prompt(self) -> Dict[str, Any]:
        """
        Devuelve un dict serializable a JSON para que el LLM
        pueda entender los gustos del usuario.
        """
        return {
            "ultimas_consultas": self.last_queries[-self.max_queries :],
            "generos_frecuentes": self._top_pref(self.genre_counts, 6),
            "tags_frecuentes": self._top_pref(self.tag_counts, 8),
            "num_juegos_vistos": len(self.seen_appids),
        }
