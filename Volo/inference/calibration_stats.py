import torch
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
    acc = np.zeros(n_bins)
    avg_conf = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for b in range(n_bins):
        m = bin_ids == b
        counts[b] = m.sum()
        if counts[b] > 0:
            acc[b] = corrects[m].mean()
            avg_conf[b] = confs[m].mean()

    ece = 0.0
    total = counts.sum() + 1e-12
    for b in range(n_bins):
        if counts[b] > 0:
            ece += (counts[b] / total) * abs(acc[b] - avg_conf[b])

    return {
        "bins": bins,
        "acc": acc,
        "avg_conf": avg_conf,
        "counts": counts,
        "ece": float(ece),
    }


def plot_reliability(calib, save_prefix=None, show=True):
    acc = calib["acc"]
    avg_conf = calib["avg_conf"]
    counts = calib["counts"]
    n_bins = len(acc)

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.plot(avg_conf, acc, marker="o")
    ax1.set_title(f"Reliability diagram (ECE={calib['ece']:.4f})")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    if save_prefix is not None:
        fig1.savefig(f"{save_prefix}_reliability.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.bar(np.arange(n_bins), counts)
    ax2.set_title("Counts per bin")
    ax2.set_xlabel("Bin")
    ax2.set_ylabel("Count")

    if save_prefix is not None:
        fig2.savefig(f"{save_prefix}_counts.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig2)

    return fig1, fig2
