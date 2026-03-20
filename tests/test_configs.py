"""Test that all YAML configs load without error and have required keys."""
from pathlib import Path

import pytest
import yaml

REQUIRED_KEYS = {"data", "model", "optimizer", "loss", "training", "output"}

CONFIG_ROOT = Path(__file__).parent.parent / "configs"


def all_configs():
    configs = list(CONFIG_ROOT.glob("*.yaml"))
    configs += list((CONFIG_ROOT / "appendix").glob("*.yaml"))
    return configs


@pytest.mark.parametrize("config_path", all_configs())
def test_config_loads(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"{config_path} did not parse to a dict"


@pytest.mark.parametrize("config_path", all_configs())
def test_config_required_keys(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    missing = REQUIRED_KEYS - set(cfg.keys())
    assert not missing, f"{config_path} missing keys: {missing}"
