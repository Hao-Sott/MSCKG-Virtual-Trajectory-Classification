from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["config_path"] = str(path)
    config["root"] = str(path.parent.parent)
    return config


def resolve_path(config, key):
    path = Path(config["paths"][key])
    if not path.is_absolute():
        path = Path(config["root"]) / path
    return path.resolve()


def with_overrides(config, **overrides):
    result = deepcopy(config)
    for dotted, value in overrides.items():
        target = result
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return result

