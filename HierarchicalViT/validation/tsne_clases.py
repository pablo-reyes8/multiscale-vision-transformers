from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
from training.autocast import *


@torch.no_grad()
def collect_features(
    model,
    dataloader,
    device="cuda",
    max_samples=2000,
    autocast_dtype="fp16",
    use_amp=True,):

    """
    Recolecta embeddings finales y labels.
    Usa model.forward_features(x): [B, D].
    """
    model.eval()
    model.to(device)

    feats = []
    labels = []

    n_collected = 0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        with autocast_ctx(
            device=device,
            enabled=use_amp,
            dtype=autocast_dtype,
            cache_enabled=True):

            z = model.forward_features(images)

        feats.append(z.cpu())
        labels.append(targets.cpu())
        n_collected += images.size(0)

        if n_collected >= max_samples:
            break

    feats = torch.cat(feats, dim=0)[:max_samples]
    labels = torch.cat(labels, dim=0)[:max_samples]
    return feats.numpy(), labels.numpy()


def tsne_plot(
    features: np.ndarray,
    labels: np.ndarray,
    class_names,
    num_points=2000,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    subset_classes=None):

    """
    Hace t-SNE 2D y lo grafica.
    - subset_classes: lista de índices de clases a mostrar (para no meter las 100).
    """
    N = min(num_points, features.shape[0])
    X = features[:N]
    y = labels[:N]

    if subset_classes is not None:
        mask = np.isin(y, np.array(subset_classes))
        X = X[mask]
        y = y[mask]

    print(f"t-SNE sobre {X.shape[0]} puntos...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=random_state)

    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        cmap="tab20",
        s=6,
        alpha=0.7)

    plt.xticks([])
    plt.yticks([])

    if subset_classes is not None:
        uniq = np.unique(y)
        handles = []
        texts = []
        for c in uniq:
            handles.append(plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(c))))
            texts.append(class_names[int(c)])
        plt.legend(handles, texts, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.title("t-SNE of HierarchicalViT embeddings", fontsize=12)
    plt.tight_layout()
    plt.show()