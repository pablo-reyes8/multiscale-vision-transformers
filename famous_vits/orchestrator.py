"""Training and inference facade shared by every registered ViT."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .factory import create_model, model_info
from .inference import (
    calibration_stats,
    classification_report,
    collect_predictions,
    confusion_matrix,
    evaluate_loader,
    grad_cam,
    occlusion_sensitivity,
)
from .inference.metrics import unpack_batch
from .optimizers import create_optimizer


class ViTOrchestrator:
    """One high-level interface for training, prediction and model analysis."""

    def __init__(
        self,
        architecture: str,
        *,
        num_classes: int = 100,
        in_chans: int = 3,
        img_size: int | tuple[int, int] = 32,
        optimizer: str | Optimizer = "adamw",
        lr: float = 5e-4,
        optimizer_kwargs: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        device: str | torch.device | None = None,
        **model_kwargs: Any,
    ):
        self.info = model_info(architecture)
        self.num_classes = num_classes
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = create_model(
            architecture,
            num_classes=num_classes,
            in_chans=in_chans,
            img_size=img_size,
            **model_kwargs,
        ).to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = (
            optimizer
            if isinstance(optimizer, Optimizer)
            else create_optimizer(
                self.model.parameters(),
                optimizer,
                lr=lr,
                **(optimizer_kwargs or {}),
            )
        )
        self.configuration = {
            "architecture": self.model.architecture_name,
            "num_classes": num_classes,
            "in_chans": in_chans,
            "img_size": img_size,
            "optimizer": optimizer if isinstance(optimizer, str) else type(optimizer).__name__,
            "lr": lr,
            "optimizer_kwargs": optimizer_kwargs or {},
            "model_kwargs": model_kwargs,
        }
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def _run_epoch(
        self,
        loader: Iterable[Any],
        *,
        training: bool,
        grad_clip_norm: float | None = None,
    ) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        samples = 0
        batches = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images, targets = unpack_batch(batch)
                images = images.to(self.device, non_blocking=True)
                targets = targets.long().to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                output = self.model(images)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                loss = self.criterion(logits, targets)
                if training:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    self.optimizer.step()
                total_loss += float(loss.detach())
                correct += int(logits.argmax(dim=1).eq(targets).sum())
                samples += targets.numel()
                batches += 1
        if batches == 0:
            raise ValueError("The DataLoader is empty.")
        return {"loss": total_loss / batches, "accuracy": correct / samples}

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        epochs: int,
        val_loader: Iterable[Any] | None = None,
        scheduler: Any | None = None,
        grad_clip_norm: float | None = 1.0,
        callback: Callable[[int, dict[str, float]], None] | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        for epoch in range(1, epochs + 1):
            train = self._run_epoch(train_loader, training=True, grad_clip_norm=grad_clip_norm)
            self.history["train_loss"].append(train["loss"])
            self.history["train_accuracy"].append(train["accuracy"])
            logs = {"train_loss": train["loss"], "train_accuracy": train["accuracy"]}
            if val_loader is not None:
                validation = self._run_epoch(val_loader, training=False)
                self.history["val_loss"].append(validation["loss"])
                self.history["val_accuracy"].append(validation["accuracy"])
                logs.update(
                    val_loss=validation["loss"],
                    val_accuracy=validation["accuracy"],
                )
            if scheduler is not None:
                scheduler.step()
            if callback is not None:
                callback(epoch, logs)
            if verbose:
                rendered = " - ".join(f"{key}: {value:.4f}" for key, value in logs.items())
                print(f"Epoch {epoch}/{epochs} - {rendered}")
        return self.history

    def evaluate(
        self, loader: Iterable[Any], *, topk: Sequence[int] = (1, 3, 5)
    ) -> dict[str, float]:
        return evaluate_loader(
            self.model,
            loader,
            device=self.device,
            criterion=self.criterion,
            topk=topk,
        )

    @torch.no_grad()
    def predict(self, inputs: Tensor, *, probabilities: bool = True) -> Tensor:
        self.model.eval()
        output = self.model(inputs.to(self.device, non_blocking=True))
        logits = output[0] if isinstance(output, (tuple, list)) else output
        return logits.softmax(dim=1) if probabilities else logits

    @torch.no_grad()
    def predict_topk(self, inputs: Tensor, *, k: int = 5) -> tuple[Tensor, Tensor]:
        probabilities = self.predict(inputs)
        return probabilities.topk(min(k, probabilities.shape[1]), dim=1)

    @torch.no_grad()
    def extract_features(self, inputs: Tensor) -> Tensor | tuple[Any, ...]:
        """Return pre-classifier features, including models without forward_features."""

        self.model.eval()
        prepared = inputs.to(self.device, non_blocking=True)
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(prepared)
            if isinstance(features, tuple) and features and isinstance(features[0], Tensor):
                return features[0]
            return features

        captured: list[Tensor] = []

        def hook(_module, args):
            captured.append(args[0])

        handle = self.model.head.register_forward_pre_hook(hook)
        try:
            self.model(prepared)
        finally:
            handle.remove()
        return captured[0]

    def analyze(
        self,
        loader: Iterable[Any],
        *,
        class_names: Sequence[str] | None = None,
        num_bins: int = 15,
    ) -> dict[str, Any]:
        logits, targets = collect_predictions(self.model, loader, device=self.device)
        return {
            "report": classification_report(logits, targets, class_names=class_names),
            "confusion_matrix": confusion_matrix(logits, targets, num_classes=self.num_classes),
            "calibration": calibration_stats(logits, targets, num_bins=num_bins),
            "logits": logits,
            "targets": targets,
        }

    def grad_cam(self, inputs: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        return grad_cam(self.model, inputs.to(self.device), **kwargs)

    def occlusion_sensitivity(self, inputs: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        return occlusion_sensitivity(self.model, inputs.to(self.device), **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            "architecture": self.model.architecture_name,
            "family": self.model.family,
            "num_classes": self.num_classes,
            "input_channels": self.model.input_channels,
            "input_size": self.model.input_size,
            "device": str(self.device),
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "optimizer": type(self.optimizer).__name__,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "configuration": self.configuration,
                "history": self.history,
                "summary": self.summary(),
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        load_optimizer: bool = False,
        **overrides: Any,
    ) -> ViTOrchestrator:
        checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
        if "configuration" not in checkpoint:
            raise ValueError(
                "Checkpoint has no configuration; use create_model(..., checkpoint_path=...)."
            )
        saved = dict(checkpoint["configuration"])
        model_kwargs = dict(saved.get("model_kwargs", {}))
        constructor_keys = {
            "architecture",
            "num_classes",
            "in_chans",
            "img_size",
            "optimizer",
            "lr",
            "optimizer_kwargs",
        }
        constructor = {key: saved[key] for key in constructor_keys if key in saved}
        constructor.update(overrides)
        constructor["device"] = device
        orchestrator = cls(**constructor, **model_kwargs)
        orchestrator.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            orchestrator.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        orchestrator.history = checkpoint.get("history", orchestrator.history)
        orchestrator.configuration.update(saved)
        return orchestrator

    def plot_history(self, **kwargs: Any):
        from .inference.visualization import plot_history

        return plot_history(self.history, **kwargs)

    def plot_predictions(self, images: Tensor, *, targets: Tensor | None = None, **kwargs: Any):
        from .inference.visualization import plot_predictions

        probabilities = self.predict(images).cpu()
        return plot_predictions(images.cpu(), probabilities, targets=targets, **kwargs)

    def plot_confusion_matrix(self, loader: Iterable[Any], **kwargs: Any):
        from .inference.visualization import plot_confusion_matrix

        logits, targets = collect_predictions(self.model, loader, device=self.device)
        matrix = confusion_matrix(logits, targets, num_classes=self.num_classes)
        return plot_confusion_matrix(matrix, **kwargs)

    def plot_calibration(self, loader: Iterable[Any], *, num_bins: int = 15, **kwargs: Any):
        from .inference.visualization import plot_calibration

        logits, targets = collect_predictions(self.model, loader, device=self.device)
        return plot_calibration(calibration_stats(logits, targets, num_bins=num_bins), **kwargs)
