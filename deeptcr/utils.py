from pathlib import Path
import yaml
from easydict import EasyDict

def load_config(config_path:str):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = Path(config_path).stem
    return config, config_name