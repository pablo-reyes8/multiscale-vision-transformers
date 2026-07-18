"""Declarative YAML pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from .arena.engine import compare_models
from .config import load_pipeline_config, validate_pipeline_config
from .data import get_classification_dataloaders, get_dataset_info
from .orchestrator import ViTOrchestrator


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synthetic_loaders(num_classes: int, in_chans: int, img_size: int):
    images = torch.randn(4, in_chans, img_size, img_size)
    targets = torch.arange(4) % num_classes
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)
    return loader, loader


def _run_train(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    runtime = config["runtime"]
    dataset_info = get_dataset_info(data_config["dataset"])
    num_classes = model_config["num_classes"] or dataset_info["num_classes"]

    if runtime["smoke_test"]:
        train_loader, val_loader = _synthetic_loaders(
            num_classes,
            model_config["in_chans"],
            model_config["img_size"],
        )
        class_names = [str(index) for index in range(num_classes)]
        dataset_name = "synthetic"
    else:
        if model_config["in_chans"] != 3:
            raise ValueError("Built-in datasets emit RGB tensors; use model.in_chans=3.")
        if num_classes != dataset_info["num_classes"]:
            raise ValueError("model.num_classes must match the selected built-in dataset.")
        train_loader, val_loader, _ = get_classification_dataloaders(
            dataset_name=data_config["dataset"],
            data_dir=data_config["data_dir"],
            batch_size=data_config["batch_size"],
            eval_batch_size=data_config["eval_batch_size"],
            num_workers=data_config["num_workers"],
            val_split=data_config["val_split"],
            augment=data_config["augment"],
            img_size=model_config["img_size"],
            seed=runtime["seed"],
        )
        class_names = list(getattr(train_loader.dataset, "classes", [])) or None
        dataset_name = data_config["dataset"]

    orchestrator = ViTOrchestrator(
        model_config["name"],
        num_classes=num_classes,
        in_chans=model_config["in_chans"],
        img_size=model_config["img_size"],
        optimizer=training_config["optimizer"],
        lr=training_config["lr"],
        optimizer_kwargs=training_config["optimizer_kwargs"],
        device=runtime["device"],
        **model_config["kwargs"],
    )
    orchestrator.fit(
        train_loader,
        epochs=training_config["epochs"],
        val_loader=val_loader,
        grad_clip_norm=training_config["grad_clip_norm"],
    )
    output = config["output"]
    checkpoint = Path(
        output["checkpoint"] or Path(output["dir"]) / f"{model_config['name']}_{dataset_name}.pt"
    )
    orchestrator.configuration.update(dataset=dataset_name, class_names=class_names)
    orchestrator.save(checkpoint)
    return {"task": "train", "checkpoint": str(checkpoint), **orchestrator.summary()}


def _run_arena(config: dict[str, Any]) -> dict[str, Any]:
    options = dict(config["arena"])
    models = options.pop("models")
    options["device"] = config["runtime"]["device"]
    options["seed"] = config["runtime"]["seed"]
    if config["runtime"]["smoke_test"]:
        options.update(epochs=1, max_train_batches=1, max_eval_batches=1)
    summary = compare_models(model_names=models, **options)
    return {
        "task": "arena",
        "output_dir": summary["output_dir"],
        "models": models,
        "table": summary["table"],
    }


def _run_analysis(config: dict[str, Any]) -> dict[str, Any]:
    options = config["analysis"]
    orchestrator = ViTOrchestrator.from_checkpoint(
        options["checkpoint"],
        device=config["runtime"]["device"],
    )
    _, _, test_loader = get_classification_dataloaders(
        dataset_name=options["dataset"],
        data_dir=options["data_dir"],
        batch_size=options["batch_size"],
        eval_batch_size=options["batch_size"],
        num_workers=options["num_workers"],
        val_split=0.0,
        img_size=int(orchestrator.configuration["img_size"]),
    )
    class_names = list(getattr(test_loader.dataset, "classes", [])) or None
    analysis = orchestrator.analyze(
        test_loader,
        class_names=class_names,
        num_bins=options["num_bins"],
    )
    output_dir = Path(options["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(analysis["confusion_matrix"], output_dir / "confusion_matrix.pt")
    serializable = {"report": analysis["report"], "calibration": analysis["calibration"]}
    (output_dir / "report.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return {"task": "analyze", "output_dir": str(output_dir), **analysis["report"]}


def run_pipeline(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Run a validated pipeline from a YAML path or an in-memory mapping."""

    resolved = (
        load_pipeline_config(config)
        if isinstance(config, (str, Path))
        else validate_pipeline_config(config)
    )
    _seed_everything(int(resolved["runtime"]["seed"]))
    if resolved["task"] == "train":
        return _run_train(resolved)
    if resolved["task"] == "arena":
        return _run_arena(resolved)
    return _run_analysis(resolved)
