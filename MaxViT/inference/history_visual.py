import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, smooth: int = 0):
    """
    history: dict con keys tipo train_loss, val_loss, train_top1, val_top1, etc.
    smooth: ventana simple (moving average) si quieres suavizar.
    """
    def _smooth(x, w):
        x = np.asarray(x, dtype=float)
        if w <= 1: return x
        k = np.ones(w) / w
        return np.convolve(x, k, mode="same")

    epochs = np.arange(1, max(len(v) for v in history.values()) + 1)

    fig, ax = plt.subplots()
    if "train_loss" in history: ax.plot(epochs, _smooth(history["train_loss"], smooth), label="train_loss")
    if "val_loss"   in history: ax.plot(epochs, _smooth(history["val_loss"], smooth),   label="val_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CE loss")
    ax.legend()
    plt.show()

    for k in ["top1", "top3", "top5"]:
        tr, va = f"train_{k}", f"val_{k}"
        if tr in history and va in history:
            fig, ax = plt.subplots()
            ax.plot(epochs, _smooth(history[tr], smooth), label=tr)
            ax.plot(epochs, _smooth(history[va], smooth), label=va)
            ax.set_title(f"Accuracy {k}")
            ax.set_xlabel("Epoch"); ax.set_ylabel("%")
            ax.legend()
            plt.show()

def plot_gaps(history):
    epochs = np.arange(1, max(len(v) for v in history.values()) + 1)
    fig, ax = plt.subplots()
    if "train_loss" in history and "val_loss" in history:
        gap = np.asarray(history["val_loss"]) - np.asarray(history["train_loss"])
        ax.plot(epochs, gap, label="gap_loss (val-train)")
    if "train_top1" in history and "val_top1" in history:
        gap = np.asarray(history["train_top1"]) - np.asarray(history["val_top1"])
        ax.plot(epochs, gap, label="gap_top1 (train-val)")
    ax.set_title("Generalization gaps")
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.show()


@torch.no_grad()
def show_predictions_grid(model, loader, class_names=None, n=16, device="cuda"):
    model.eval().to(device)

    xs, ys = next(iter(loader))
    xs, ys = xs[:n].to(device), ys[:n].to(device)

    logits = model(xs)
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)

    xs_cpu = xs.detach().cpu().permute(0, 2, 3, 1).numpy()  # [B,H,W,C]
    ys_cpu = ys.detach().cpu().numpy()
    pred_cpu = pred.detach().cpu().numpy()
    conf_cpu = conf.detach().cpu().numpy()

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows))
    axes = np.asarray(axes).reshape(-1)

    for i in range(rows*cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        img = xs_cpu[i]
        # si tus imgs están normalizadas, desnormaliza aquí si quieres
        ax.imshow(np.clip(img, 0, 1))
        y = ys_cpu[i]; p = pred_cpu[i]
        y_name = class_names[y] if class_names is not None else str(y)
        p_name = class_names[p] if class_names is not None else str(p)
        ax.set_title(f"GT: {y_name}\nPred: {p_name} ({conf_cpu[i]*100:.1f}%)", fontsize=10)

    plt.tight_layout()
    plt.show()