import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def unnormalize(images: torch.Tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD):
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def show_batch(images: torch.Tensor, labels: torch.Tensor, class_names=None, n: int = 8):
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
