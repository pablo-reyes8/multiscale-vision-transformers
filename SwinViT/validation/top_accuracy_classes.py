import numpy as np
import torch

from training.autocast import autocast_ctx


@torch.no_grad()
def collect_predictions(
    model,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True):
    """
    Returns:
        all_logits: [N, num_classes]
        all_targets: [N]
        all_preds: [N]
    """
    model.eval()
    model.to(device)

    all_logits = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_ctx(
            device=device,
            enabled=use_amp,
            dtype=autocast_dtype,
            cache_enabled=True):
            logits = model(images)

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = all_logits.argmax(dim=1)

    return all_logits, all_targets, all_preds


def per_class_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int):
    """
    Returns:
        acc_per_class: np.array [num_classes]
        counts: np.array [num_classes]
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    acc_per_class = np.zeros(num_classes, dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int64)

    for y_pred, y_true in zip(preds, targets):
        y_true = int(y_true)
        counts[y_true] += 1
        if y_pred == y_true:
            acc_per_class[y_true] += 1

    mask = counts > 0
    acc_per_class[mask] = acc_per_class[mask] / counts[mask]
    return acc_per_class, counts


def print_best_and_worst_classes(
    acc_per_class: np.ndarray,
    counts: np.ndarray,
    class_names: list[str],
    k: int = 10):
    idx_sorted = np.argsort(acc_per_class)

    print("\nHardest classes:")
    for i in idx_sorted[:k]:
        print(f"{class_names[i]:>20s} | acc={acc_per_class[i]*100:5.2f}% | n={counts[i]}")

    print("\nEasiest classes:")
    for i in idx_sorted[-k:][::-1]:
        print(f"{class_names[i]:>20s} | acc={acc_per_class[i]*100:5.2f}% | n={counts[i]}")
