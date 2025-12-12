import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from training.autocast import *

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def unnormalize(images: torch.Tensor,
                mean=CIFAR100_MEAN,
                std=CIFAR100_STD) -> torch.Tensor:
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def show_predictions_grid(
    model,
    dataloader,
    class_names,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    n: int = 8,
    only_misclassified: bool = False,
):
    """
    Muestra un grid de imágenes con:
    - predicción
    - label verdadera
    - marca ✔ / ✘ si está bien o mal.

    Args:
        n: cuántas imágenes mostrar.
        only_misclassified: si True, intenta mostrar solo ejemplos mal clasificados.
    """
    model.eval()
    model.to(device)

    images_list = []
    labels_true = []
    labels_pred = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            with autocast_ctx(
                device=device,
                enabled=use_amp,
                dtype=autocast_dtype,
                cache_enabled=True):

                logits = model(images)
                preds = logits.argmax(dim=1)

            if only_misclassified:
                mask = preds != targets
            else:
                mask = torch.ones_like(targets, dtype=torch.bool)

            # Filtramos
            sel_images = images[mask]
            sel_targets = targets[mask]
            sel_preds = preds[mask]

            for img, y_true, y_pred in zip(sel_images, sel_targets, sel_preds):
                images_list.append(img.cpu())
                labels_true.append(int(y_true))
                labels_pred.append(int(y_pred))
                if len(images_list) >= n:
                    break

            if len(images_list) >= n:
                break

    if len(images_list) == 0:
        print("No hay ejemplos que cumplan el criterio.")
        return

    imgs = torch.stack(images_list, dim=0)
    imgs = unnormalize(imgs)

    grid = make_grid(imgs, nrow=n, padding=2)
    npimg = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(2 * n, 2.5))
    plt.imshow(npimg)
    plt.axis("off")

    titles = []
    for y_t, y_p in zip(labels_true, labels_pred):
        ok = (y_t == y_p)
        mark = "✔" if ok else "✘"
        titles.append(f"{mark} {class_names[y_p]} (true: {class_names[y_t]})")

    plt.title(" | ".join(titles), fontsize=8)
    plt.show()