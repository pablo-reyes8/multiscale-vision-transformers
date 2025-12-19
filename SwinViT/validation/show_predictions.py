import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

from training.autocast import autocast_ctx
from validation.utils import unnormalize


def show_predictions_grid(
    model,
    dataloader,
    class_names,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    n: int = 8,
    only_misclassified: bool = False):
    """
    Show a grid of predictions and ground-truth labels.

    Args:
        n: number of images to display.
        only_misclassified: show only mistakes when True.
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
        print("No samples matched the selection criteria.")
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
        mark = "OK" if ok else "X"
        titles.append(f"{mark} {class_names[y_p]} (true: {class_names[y_t]})")

    plt.title(" | ".join(titles), fontsize=8)
    plt.show()
