# pages/03_Datos.py  — versión ligera con DuckDB
import streamlit as st
import duckdb as ddb
from pathlib import Path

st.title("📊 Datos usados por FAISS (modo ligero)")

PARQUET = Path("data/processed/games.parquet")
if not PARQUET.exists():
    st.error("No existe data/processed/games.parquet")
    st.stop()

@st.cache_resource
def get_con():
    con = ddb.connect(database=":memory:")
    # No carga a RAM: crea una vista sobre el parquet
    con.execute(f"""
        CREATE OR REPLACE VIEW games AS
        SELECT * FROM read_parquet('{PARQUET.as_posix()}', filename=true);
    """)
    return con

con = get_con()

# ---------- MÉTRICAS BÁSICAS (sin cargar el DF completo)
total = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
st.metric("Juegos totales", total)

FIELDS = ["name", "short_description", "tags", "genres", "categories"]

# Cobertura de campos indexados (no vacíos)
coverage_sql = "SELECT " + ", ".join(
    [f"SUM(LENGTH(TRIM(COALESCE(CAST({c} AS VARCHAR), ''))) > 0)::INT AS {c}_nonempty" for c in FIELDS]
) + " FROM games"
cov_row = con.execute(coverage_sql).fetchdf().iloc[0].to_dict()
st.subheader("Cobertura de campos (lo que entra al corpus FAISS)")
st.write({k.replace("_nonempty",""): f"{v} ({(v/total):.1%})" for k, v in cov_row.items()})

# ---------- PREVIEW (limitado)
st.subheader("Vista previa de datos indexados")
n = st.slider("Filas a previsualizar", 5, 100, 20, 5)
preview_cols = ["appid","name","short_description","tags","genres","categories"]
preview_cols = [c for c in preview_cols if con.execute(f"PRAGMA table_info('games')").fetchdf()["name"].str.contains(f"^{c}$").any()]
preview = con.execute(f"""
    SELECT {", ".join(preview_cols)}
    FROM games
    LIMIT {int(n)}
""").fetchdf()
st.dataframe(preview, use_container_width=True)

# ---------- CORPUS (cómo se construye el texto para FAISS) - solo primeras N
def build_corpus_row(r: dict) -> str:
    def toks(x):
        if x is None: return []
        s = str(x).strip()
        if not s: return []
        # si viene como lista en texto, muchas veces está separada por coma
        return [t.strip() for t in s.split(",")] if "," in s else [s]
    name  = str(r.get("name","") or "")
    short = str(r.get("short_description","") or "")
    tags  = " ".join(toks(r.get("tags")))
    gens  = " ".join(toks(r.get("genres")))
    cats  = " ".join(toks(r.get("categories")))
    return f"{name}\n{short}\nTags: {tags}\nGenres: {gens}\nCategories: {cats}"

with st.expander("Ver 'corpus' generado (primeras filas)"):
    corp = con.execute(f"""
        SELECT appid, name, short_description, tags, genres, categories
        FROM games
        LIMIT {int(n)}
    """).fetchdf()
    corp["faiss_text"] = corp.apply(lambda r: build_corpus_row(r), axis=1)
    st.dataframe(corp[["appid","name","faiss_text"]], use_container_width=True)

# ---------- BUSCADOR sobre campos indexados (SQL con LIMIT para no cargar todo)
st.subheader("Buscador en campos indexados")
q = st.text_input("Buscar (en name, short_description, tags, genres, categories)…")
limit = st.slider("Límite de resultados", 50, 500, 200, 50)
if q:
    q_sql = q.replace("'", "''").lower()
    where = " OR ".join([
        f"LOWER(CAST({c} AS VARCHAR)) LIKE '%{q_sql}%'"
        for c in FIELDS
    ])
    res = con.execute(f"""
        SELECT appid, name, short_description, genres, categories, tags
        FROM games
        WHERE {where}
        LIMIT {int(limit)}
    """).fetchdf()
    st.write(f"{len(res)} resultados (mostrando hasta {limit})")
    st.dataframe(res, use_container_width=True)
