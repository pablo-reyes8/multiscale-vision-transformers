"""Architecture-agnostic classification evaluation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn


def unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if isinstance(batch, dict):
        image_key = "images" if "images" in batch else "image"
        target_key = "targets" if "targets" in batch else "target"
        return batch[image_key], batch[target_key]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Each batch must be (images, targets) or an equivalent dictionary.")


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    model.eval().to(device)
    logits_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for batch in loader:
        images, targets = unpack_batch(batch)
        output = model(images.to(device, non_blocking=True))
        logits = output[0] if isinstance(output, (tuple, list)) else output
        logits_parts.append(logits.detach().cpu())
        target_parts.append(targets.detach().long().cpu())
    if not logits_parts:
        raise ValueError("The DataLoader is empty.")
    return torch.cat(logits_parts), torch.cat(target_parts)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    device: str | torch.device = "cpu",
    criterion: nn.Module | None = None,
    topk: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    criterion = criterion or nn.CrossEntropyLoss()
    logits, targets = collect_predictions(model, loader, device=device)
    loss = float(criterion(logits, targets))
    max_classes = logits.shape[1]
    metrics = {"loss": loss, "samples": float(targets.numel())}
    for requested_k in topk:
        k = min(int(requested_k), max_classes)
        correct = logits.topk(k, dim=1).indices.eq(targets[:, None]).any(dim=1)
        metrics[f"top{requested_k}"] = float(correct.float().mean())
    return metrics


def confusion_matrix(
    logits: Tensor,
    targets: Tensor,
    *,
    num_classes: int | None = None,
    normalize: str | None = None,
) -> Tensor:
    predictions = logits.argmax(dim=1) if logits.ndim == 2 else logits.long()
    targets = targets.long()
    num_classes = num_classes or int(torch.cat([predictions, targets]).max()) + 1
    indices = targets * num_classes + predictions
    matrix = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    if normalize is None:
        return matrix
    matrix = matrix.float()
    if normalize == "true":
        return matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1)
    if normalize == "pred":
        return matrix / matrix.sum(dim=0, keepdim=True).clamp_min(1)
    if normalize == "all":
        return matrix / matrix.sum().clamp_min(1)
    raise ValueError("normalize must be one of None, 'true', 'pred', or 'all'.")


def calibration_stats(logits: Tensor, targets: Tensor, *, num_bins: int = 15) -> dict[str, Any]:
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets).float()
    boundaries = torch.linspace(0, 1, num_bins + 1)
    bins: list[dict[str, float]] = []
    ece = torch.tensor(0.0)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = confidence.ge(lower) & (
            confidence.le(upper) if index == num_bins - 1 else confidence.lt(upper)
        )
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else 0.0
        mean_confidence = float(confidence[mask].mean()) if count else 0.0
        fraction = mask.float().mean()
        ece += fraction * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": float(count),
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).float()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean())
    return {"ece": float(ece), "brier": brier, "bins": bins}


def classification_report(
    logits: Tensor,
    targets: Tensor,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    num_classes = logits.shape[1]
    matrix = confusion_matrix(logits, targets, num_classes=num_classes).float()
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    per_class = {
        names[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(num_classes)
    }
    return {
        "accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
    }
