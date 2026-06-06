"""Configuration loading helpers."""

from copy import deepcopy

import yaml


def _set_nested(config, dotted_key, value):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def get_nested(config, dotted_key, default=None):
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_config(config_path, overrides=None):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config = deepcopy(config)
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_nested(config, key, value)
    return config
