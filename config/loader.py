import yaml
from pathlib import Path

def load_settings():
    path = Path(__file__).resolve().parent / "settings.yaml"
    with open(path,"r") as f:
        return yaml.safe_load(f)