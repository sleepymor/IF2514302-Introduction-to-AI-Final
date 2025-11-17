import yaml
from pathlib import Path

def load_config(path: str = "configs/default_config.yaml"):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)