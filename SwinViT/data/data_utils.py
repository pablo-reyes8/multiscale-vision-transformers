import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import datasets

def unnormalize(images: torch.Tensor,
                mean=CIFAR100_MEAN,
                std=CIFAR100_STD):
    """
    Des-normaliza un batch de imágenes.
    images: tensor [B, C, H, W] normalizado.
    """
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def show_batch(images: torch.Tensor,
               labels: torch.Tensor,
               class_names=None,
               n: int = 8):
    """
    Muestra las primeras n imágenes de un batch con sus labels.

    Args:
        images: tensor [B, C, H, W] (normalizado).
        labels: tensor [B].
        class_names: lista de nombres de clases (len = 100).
        n: cuántas imágenes mostrar (en una fila).
    """
    images = images[:n].cpu()
    labels = labels[:n].cpu()
    images_unnorm = unnormalize(images)

    grid = make_grid(images_unnorm, nrow=n, padding=2)
    npimg = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(2 * n, 2.5))
    plt.imshow(npimg)
    plt.axis("off")

    if class_names is not None:
        title = " | ".join(class_names[int(lbl)] for lbl in labels)
        plt.title(title, fontsize=10)
    plt.show()