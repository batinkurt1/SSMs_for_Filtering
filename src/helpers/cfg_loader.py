from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_yaml(path):
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return {} if data is None else data