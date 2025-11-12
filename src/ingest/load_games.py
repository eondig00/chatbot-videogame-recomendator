import argparse, pandas as pd
from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger("ingest")

def _as_list(x):
    if pd.isna(x): return []
    s = str(x).strip()
    return [t.strip() for t in s.split(",")] if s else []

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["appid","price","user_score","positive","negative","num_reviews_total","discount","peak_ccu"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year
    for c in ("genres","tags","categories","supported_languages","full_audio_languages"):
        if c in df.columns: df[c] = df[c].apply(_as_list)
    if "appid" in df.columns:
        before = len(df); df = df.drop_duplicates(subset=["appid"])
        log.info(f"Removed {before-len(df)} duplicates")
    return df

def main(in_path: str | None, out_path: str | None):
    cfg = load_config()
    in_path = in_path or cfg["data"]["primary_csv"]
    out_path = out_path or str(Path(cfg["paths"]["processed"]) / "games.parquet")

    log.info(f"Loading {in_path}")
    df = pd.read_csv(in_path, low_memory=False, encoding="utf-8")
    df = normalize(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"Wrote {out_path} with {len(df)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=None)
    ap.add_argument("--out", dest="out_path", default=None)
    args = ap.parse_args()
    main(args.in_path, args.out_path)
