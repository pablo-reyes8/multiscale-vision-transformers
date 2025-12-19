import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from training.autocast import autocast_ctx


@torch.no_grad()
def collect_features(
    model,
    dataloader,
    device: str = "cuda",
    max_samples: int = 2000,
    autocast_dtype: str = "fp16",
    use_amp: bool = True):
    """
    Collect final embeddings using model.forward_features(x): [B, D].
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
    class_names: list[str],
    num_points: int = 2000,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    subset_classes: list[int] | None = None):
    """
    Run t-SNE on embeddings and plot in 2D.
    """
    N = min(num_points, features.shape[0])
    X = features[:N]
    y = labels[:N]

    if subset_classes is not None:
        mask = np.isin(y, np.array(subset_classes))
        X = X[mask]
        y = y[mask]

    print(f"t-SNE on {X.shape[0]} samples...")

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

    plt.title("t-SNE of SwinViT embeddings", fontsize=12)
    plt.tight_layout()
    plt.show()
