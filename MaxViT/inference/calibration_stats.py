import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def calibration_stats(model, loader, n_bins=15, device="cuda"):
    model.eval().to(device)

    confs = []
    corrects = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = model(x).softmax(dim=1)
        conf, pred = probs.max(dim=1)
        confs.append(conf.detach().cpu().numpy())
        corrects.append((pred == y).detach().cpu().numpy().astype(np.float32))

    confs = np.concatenate(confs)
    corrects = np.concatenate(corrects)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confs, bins) - 1
    acc = np.zeros(n_bins); avg_conf = np.zeros(n_bins); counts = np.zeros(n_bins)

    for b in range(n_bins):
        m = bin_ids == b
        counts[b] = m.sum()
        if counts[b] > 0:
            acc[b] = corrects[m].mean()
            avg_conf[b] = confs[m].mean()

    # Expected Calibration Error (ECE)
    ece = 0.0
    total = counts.sum() + 1e-12
    for b in range(n_bins):
        if counts[b] > 0:
            ece += (counts[b] / total) * abs(acc[b] - avg_conf[b])

    return {"bins": bins, "acc": acc, "avg_conf": avg_conf, "counts": counts, "ece": float(ece)}

def plot_reliability(calib):
    acc = calib["acc"]; avg_conf = calib["avg_conf"]; counts = calib["counts"]
    n_bins = len(acc)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(avg_conf, acc, marker="o")
    ax.set_title(f"Reliability diagram (ECE={calib['ece']:.4f})")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(np.arange(n_bins), counts)
    ax.set_title("Counts per bin")
    ax.set_xlabel("Bin"); ax.set_ylabel("Count")
    plt.show()