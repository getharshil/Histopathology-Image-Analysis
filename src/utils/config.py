"""YAML configuration loader."""
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
