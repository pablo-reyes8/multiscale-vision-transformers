import copy

import pytest
import yaml

from famous_vits.config import load_pipeline_config, validate_pipeline_config
from famous_vits.pipeline import run_pipeline


def test_load_pipeline_config_applies_defaults():
    config = load_pipeline_config("configs/smoke_test.yaml")

    assert config["version"] == 1
    assert config["task"] == "train"
    assert config["data"]["dataset"] == "cifar100"
    assert config["runtime"]["smoke_test"] is True


def test_pipeline_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown keys in model"):
        validate_pipeline_config(
            {
                "task": "train",
                "model": {"name": "vit", "typo": 1},
            }
        )


def test_example_configs_are_valid_yaml_and_pipeline_configs():
    for path in (
        "configs/train_cifar100.yaml",
        "configs/arena_cifar100.yaml",
        "configs/analyze_cifar100.yaml",
        "configs/smoke_test.yaml",
    ):
        with open(path, encoding="utf-8") as handle:
            assert isinstance(yaml.safe_load(handle), dict)
        assert load_pipeline_config(path)["version"] == 1


def test_run_smoke_pipeline(tmp_path):
    config = copy.deepcopy(load_pipeline_config("configs/smoke_test.yaml"))
    checkpoint = tmp_path / "smoke.pt"
    config["output"]["checkpoint"] = str(checkpoint)

    result = run_pipeline(config)

    assert result["task"] == "train"
    assert result["checkpoint"] == str(checkpoint)
    assert result["num_classes"] == 2
    assert checkpoint.is_file()
