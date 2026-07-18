"""Model registry and timm-like construction helpers."""

from __future__ import annotations

import copy
import importlib
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .arena.presets import MODEL_PRESETS

REPO_ROOT = Path(__file__).resolve().parents[1]
_IMPORT_PREFIXES = {"model", "data", "training", "validation", "scripts", "inference"}


@dataclass(frozen=True)
class ModelInfo:
    """Public metadata for a registered architecture preset."""

    name: str
    family: str
    default_input_channels: int
    recommended_input_size: int
    description: str
    dynamic_image_size: bool


_PROJECTS = {
    "vit": "model_zoo/ViT",
    "hierarchical_vit": "model_zoo/HierarchicalViT",
    "swin": "model_zoo/SwinViT",
    "maxvit": "model_zoo/MaxViT",
    "volo": "model_zoo/Volo",
}

_ALIASES = {
    "vision_transformer": "vit",
    "hvit": "hierarchical_vit",
    "hierarchicalvit": "hierarchical_vit",
    "swin_vit": "swin",
    "swin_transformer": "swin",
    "maxvit": "maxvit_tiny",
    "max_vit": "maxvit_tiny",
    "maxvit_t": "maxvit_tiny",
    "maxvit_s": "maxvit_small",
    "maxvit_b": "maxvit_base",
    "volo_flat": "volo",
    "volo_pyramid": "volo_hierarchical",
}


def _canonical_name(name: str) -> str:
    normalized = name.lower().strip().replace("-", "_").replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in MODEL_PRESETS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(list_models())}.")
    return normalized


def list_models(family: str | None = None) -> list[str]:
    """Return canonical preset names, optionally filtered by family."""

    names = list(MODEL_PRESETS)
    if family is None:
        return names
    wanted = family.lower().replace(" ", "_")
    return [
        name
        for name in names
        if MODEL_PRESETS[name]["family"].lower().replace(" ", "_") == wanted
        or MODEL_PRESETS[name]["builder"] == wanted
    ]


def model_info(name: str) -> ModelInfo:
    canonical = _canonical_name(name)
    preset = MODEL_PRESETS[canonical]
    return ModelInfo(
        name=canonical,
        family=preset["family"],
        default_input_channels=3,
        recommended_input_size=32,
        description=preset["description"],
        dynamic_image_size=preset["builder"] in {"vit", "swin", "volo"},
    )


@contextmanager
def _isolated_project_imports(project: str):
    """Temporarily isolate legacy absolute imports such as ``model.*``.

    The original subprojects intentionally remain importable on their own. This
    boundary lets the packaged API reuse them without collisions between their
    identically named ``model`` packages.
    """

    project_root = REPO_ROOT / project
    previous_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.split(".", 1)[0] in _IMPORT_PREFIXES
    }
    for name in list(sys.modules):
        if name.split(".", 1)[0] in _IMPORT_PREFIXES:
            del sys.modules[name]

    sys.path.insert(0, str(project_root))
    model_path = project_root / "model"
    if model_path.exists():
        sys.path.insert(1, str(model_path))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".", 1)[0] in _IMPORT_PREFIXES:
                del sys.modules[name]
        sys.path[:] = previous_path
        sys.modules.update(saved_modules)


def _validate_common(img_size: int | tuple[int, int], in_chans: int, num_classes: int) -> None:
    sizes = (img_size, img_size) if isinstance(img_size, int) else img_size
    if len(sizes) != 2 or min(sizes) < 1:
        raise ValueError("img_size must be a positive integer or (height, width) tuple.")
    if in_chans < 1:
        raise ValueError("in_chans must be a positive integer.")
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")


def _load_checkpoint(model: nn.Module, path: str | Path, strict: bool) -> None:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state dictionary.")
    model.load_state_dict(state, strict=strict)


def _build_maxvit(
    variant: str,
    config: dict[str, Any],
    *,
    in_chans: int,
    num_classes: int,
    overrides: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    with _isolated_project_imports(_PROJECTS["maxvit"]):
        cfg_module = importlib.import_module("model_configurations")
        maxvit_module = importlib.import_module("model.MaxViT")

        builders = {
            "tiny": cfg_module.maxvit_cifar100_tiny,
            "small": cfg_module.maxvit_cifar100_small,
            "base": cfg_module.maxvit_cifar100_base,
        }
        cfg = builders[variant](**config)
        valid_fields = set(asdict(cfg))
        unknown = sorted(set(overrides) - valid_fields)
        if unknown:
            raise TypeError(f"Unsupported MaxViT arguments: {unknown}")
        cfg = replace(cfg, num_classes=num_classes, in_chans=in_chans, **overrides)
        model = maxvit_module.MaxViT(cfg)
    resolved = asdict(cfg)
    resolved["variant"] = variant
    return model, resolved


def create_model(
    name: str,
    *,
    pretrained: bool = False,
    checkpoint_path: str | Path | None = None,
    num_classes: int = 100,
    in_chans: int = 3,
    img_size: int | tuple[int, int] = 32,
    strict: bool = True,
    **model_kwargs: Any,
) -> nn.Module:
    """Create any local ViT through one stable, timm-like entry point.

    ``pretrained=True`` means weights must be supplied through
    ``checkpoint_path``; this repository does not silently download weights.
    Preset values can be overridden with architecture-specific keyword args.
    """

    canonical = _canonical_name(name)
    _validate_common(img_size, in_chans, num_classes)
    if pretrained and checkpoint_path is None:
        raise ValueError("pretrained=True requires checkpoint_path; no remote weights are bundled.")

    preset = MODEL_PRESETS[canonical]
    builder = preset["builder"]
    config = copy.deepcopy(preset["config"])

    if "in_channels" in model_kwargs:
        alias_value = model_kwargs.pop("in_channels")
        if in_chans != 3 and alias_value != in_chans:
            raise ValueError("Use only one value for in_chans/in_channels.")
        in_chans = int(alias_value)

    if builder == "maxvit":
        model, resolved = _build_maxvit(
            preset["variant"],
            config,
            in_chans=in_chans,
            num_classes=num_classes,
            overrides=model_kwargs,
        )
    else:
        config.update(model_kwargs)
        project = _PROJECTS[builder]
        module_name, class_name = {
            "vit": ("model.vision_transformer", "VisionTransformer"),
            "hierarchical_vit": ("model.hierarchical_vit", "HierarchicalViT"),
            "swin": ("model.swin_vision_transformer", "SwinTransformer"),
            "volo": ("model.VOLO", "VOLOClassifier"),
        }[builder]
        with _isolated_project_imports(project):
            model_class = getattr(importlib.import_module(module_name), class_name)
            model = model_class(
                img_size=img_size,
                in_chans=in_chans,
                num_classes=num_classes,
                **config,
            )
        resolved = config

    model.architecture_name = canonical
    model.family = preset["family"]
    model.input_channels = in_chans
    model.num_classes = num_classes
    model.input_size = img_size
    model.model_config = resolved

    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, strict=strict)
    return model
