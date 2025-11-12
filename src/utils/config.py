from pathlib import Path
import yaml

def load_config(path: str | Path = "config.yml") -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    # asegura carpetas
    for k in ("processed", "embeddings", "index"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    return cfg
