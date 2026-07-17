"""Plotting helpers kept optional so core model creation stays lightweight."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _finish_figure(figure, save_path: str | Path | None, show: bool):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, bbox_inches="tight", dpi=160)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return figure


def plot_history(
    history: dict[str, list[float]],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "val"):
        loss = history.get(f"{split}_loss", [])
        metric = history.get(f"{split}_accuracy", [])
        if loss:
            axes[0].plot(range(1, len(loss) + 1), loss, label=split)
        if metric:
            axes[1].plot(range(1, len(metric) + 1), metric, label=split)
    axes[0].set(xlabel="epoch", ylabel="loss")
    axes[1].set(xlabel="epoch", ylabel="accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    return _finish_figure(figure, save_path, show)


def plot_confusion_matrix(
    matrix: Tensor,
    *,
    class_names: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    size = matrix.shape[0]
    figure_size = min(18, max(6, size * 0.22))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix.cpu(), cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046)
    if class_names is not None and size <= 50:
        axis.set_xticks(range(size), class_names, rotation=90, fontsize=7)
        axis.set_yticks(range(size), class_names, fontsize=7)
    axis.set(xlabel="predicted", ylabel="true")
    figure.tight_layout()
    return _finish_figure(figure, save_path, show)


def plot_calibration(
    stats: dict[str, Any],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    bins = stats["bins"]
    confidence = [item["confidence"] for item in bins]
    accuracy = [item["accuracy"] for item in bins]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    axis.plot(confidence, accuracy, marker="o", label=f"model (ECE={stats['ece']:.3f})")
    axis.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return _finish_figure(figure, save_path, show)


def plot_predictions(
    images: Tensor,
    probabilities: Tensor,
    *,
    targets: Tensor | None = None,
    class_names: Sequence[str] | None = None,
    max_images: int = 12,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    import math

    import matplotlib.pyplot as plt

    count = min(max_images, images.shape[0])
    columns = min(4, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), squeeze=False)
    predictions = probabilities.argmax(dim=1)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        image = images[index].detach().cpu().float()
        if mean is not None and std is not None:
            image = image * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
        if image.shape[0] == 1:
            axis.imshow(image[0].clamp(0, 1), cmap="gray")
        else:
            axis.imshow(image[:3].permute(1, 2, 0).clamp(0, 1))
        predicted = int(predictions[index])
        name = class_names[predicted] if class_names else str(predicted)
        confidence = float(probabilities[index, predicted])
        expected = "" if targets is None else f" | true={int(targets[index])}"
        axis.set_title(f"{name} ({confidence:.1%}){expected}", fontsize=9)
    figure.tight_layout()
    return _finish_figure(figure, save_path, show)
