"""Loading and validation for declarative YAML pipelines."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_VERSION = 1
SUPPORTED_TASKS = {"train", "arena", "analyze"}

COMMON_DEFAULTS: dict[str, Any] = {
    "version": CONFIG_VERSION,
    "runtime": {
        "device": None,
        "seed": 42,
        "smoke_test": False,
    },
}

TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "train": {
        "model": {
            "name": "vit",
            "num_classes": None,
            "in_chans": 3,
            "img_size": 32,
            "kwargs": {},
        },
        "data": {
            "dataset": "cifar100",
            "data_dir": "./data",
            "batch_size": 128,
            "eval_batch_size": 256,
            "num_workers": 2,
            "val_split": 0.1,
            "augment": "matched",
        },
        "training": {
            "epochs": 20,
            "optimizer": "adamw",
            "lr": 5e-4,
            "optimizer_kwargs": {},
            "grad_clip_norm": 1.0,
        },
        "output": {
            "dir": "outputs",
            "checkpoint": None,
        },
    },
    "arena": {
        "arena": {
            "models": ["vit", "hierarchical_vit", "swin"],
            "dataset": "cifar100",
            "data_dir": "./data",
            "output_dir": None,
            "augment": "matched",
            "img_size": 32,
            "num_classes": None,
            "in_chans": 3,
            "batch_size": 128,
            "eval_batch_size": 256,
            "epochs": 20,
            "lr": 5e-4,
            "weight_decay": 0.05,
            "val_split": 0.1,
            "num_workers": 2,
            "autocast_dtype": "fp16",
            "use_amp": True,
            "grad_clip_norm": 1.0,
            "warmup_ratio": 0.05,
            "min_lr": 0.0,
            "label_smoothing": 0.1,
            "print_every": 100,
            "max_train_batches": None,
            "max_eval_batches": None,
        }
    },
    "analyze": {
        "analysis": {
            "checkpoint": None,
            "dataset": "cifar100",
            "data_dir": "./data",
            "batch_size": 256,
            "num_workers": 2,
            "num_bins": 15,
            "output_dir": "outputs/analysis",
        }
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _reject_unknown_keys(payload: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {unknown}")


def validate_pipeline_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and expand a pipeline mapping with task-specific defaults."""

    if not isinstance(payload, dict):
        raise TypeError("Pipeline configuration must be a YAML mapping.")
    _reject_unknown_keys(
        payload,
        {"version", "task", "runtime", "model", "data", "training", "output", "arena", "analysis"},
        "pipeline root",
    )
    version = payload.get("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        raise ValueError(f"Unsupported config version {version}; expected {CONFIG_VERSION}.")
    task = str(payload.get("task", "train")).lower()
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{task}'. Available: {sorted(SUPPORTED_TASKS)}")

    allowed_for_task = {
        "train": {"version", "task", "runtime", "model", "data", "training", "output"},
        "arena": {"version", "task", "runtime", "arena"},
        "analyze": {"version", "task", "runtime", "analysis"},
    }[task]
    _reject_unknown_keys(payload, allowed_for_task, f"task '{task}'")
    if "runtime" in payload:
        if not isinstance(payload["runtime"], dict):
            raise TypeError("'runtime' must be a mapping.")
        _reject_unknown_keys(payload["runtime"], set(COMMON_DEFAULTS["runtime"]), "runtime")
    for section, defaults in TASK_DEFAULTS[task].items():
        if section not in payload:
            continue
        if not isinstance(payload[section], dict):
            raise TypeError(f"'{section}' must be a mapping.")
        _reject_unknown_keys(payload[section], set(defaults), section)

    resolved = _deep_merge(COMMON_DEFAULTS, TASK_DEFAULTS[task])
    resolved = _deep_merge(resolved, payload)
    resolved["task"] = task

    if task == "train":
        for section in ("model", "data", "training", "output", "runtime"):
            if not isinstance(resolved[section], dict):
                raise TypeError(f"'{section}' must be a mapping.")
        if not resolved["model"]["name"]:
            raise ValueError("model.name is required.")
        if resolved["training"]["epochs"] < 1:
            raise ValueError("training.epochs must be >= 1.")
    elif task == "arena":
        models = resolved["arena"]["models"]
        if not isinstance(models, list) or not models:
            raise ValueError("arena.models must be a non-empty list.")
    elif not resolved["analysis"]["checkpoint"]:
        raise ValueError("analysis.checkpoint is required.")
    return resolved


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, expand environment variables and return a resolved config."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = os.path.expandvars(config_path.read_text(encoding="utf-8"))
    payload = yaml.safe_load(raw)
    return validate_pipeline_config(payload)
