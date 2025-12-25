import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def confusion_matrix(model, loader, num_classes=100, device="cuda"):
    model.eval().to(device)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[int(t), int(p)] += 1
    return cm.cpu().numpy()

def plot_confusion_matrix(cm, class_names=None, normalize=True, max_classes=30):
    """
    Para CIFAR-100, plotear 100x100 es ilegible. Esto plotea una submatriz de las clases
    con más ejemplos o más error (simplemente toma las primeras max_classes).
    """
    cm = cm.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sums

    cm_small = cm[:max_classes, :max_classes]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_small, aspect="auto")
    ax.set_title(f"Confusion matrix (first {max_classes} classes)")
    ax.set_xlabel("Pred"); ax.set_ylabel("True")

    if class_names is not None:
        ax.set_xticks(range(max_classes))
        ax.set_yticks(range(max_classes))
        ax.set_xticklabels(class_names[:max_classes], rotation=90, fontsize=7)
        ax.set_yticklabels(class_names[:max_classes], fontsize=7)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def top_confusions(cm, k=20):
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    return pairs[:k]
