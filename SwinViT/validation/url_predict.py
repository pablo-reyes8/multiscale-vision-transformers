from io import BytesIO

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from torchvision import transforms

from training.autocast import autocast_ctx
from validation.utils import CIFAR100_MEAN, CIFAR100_STD, unnormalize


cifar100_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img


@torch.no_grad()
def predict_from_url(
    model,
    url: str,
    class_names,
    device: str = "cuda",
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    transform=cifar100_test_transform,
    topk: int = 5):
    """
    Download an image, preprocess, and run a top-k prediction.
    """
    model.eval()
    model.to(device)

    img_pil = load_image_from_url(url)
    x = transform(img_pil)
    x = x.unsqueeze(0).to(device)

    with autocast_ctx(
        device=device,
        enabled=use_amp,
        dtype=autocast_dtype,
        cache_enabled=True):
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    topk_vals, topk_idxs = torch.topk(probs, k=topk, dim=1)
    topk_vals = topk_vals[0].cpu().numpy()
    topk_idxs = topk_idxs[0].cpu().numpy()

    print(f"Predictions for: {url}\n")
    for p, idx in zip(topk_vals, topk_idxs):
        print(f"{class_names[int(idx)]:>15s}: {p*100:5.2f}%")

    return topk_idxs, topk_vals, img_pil, x


def tensor_to_img(x_tensor: torch.Tensor):
    assert x_tensor.dim() == 3, f"Expected [C,H,W], got {x_tensor.shape}"
    x = x_tensor.clamp(0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()
    return x


def show_url_prediction_pair(
    img_pil,
    x_tensor_norm,
    topk_idxs,
    topk_vals,
    class_names):
    """
    Show original and preprocessed image, plus top-k predictions.
    """
    if x_tensor_norm.dim() == 3:
        x_tensor_norm = x_tensor_norm.unsqueeze(0)

    x_vis_batch = unnormalize(x_tensor_norm)
    x_vis = x_vis_batch[0]
    x_vis_np = tensor_to_img(x_vis)

    title_lines = [f"{class_names[int(i)]}: {float(p)*100:4.1f}%"
        for i, p in zip(topk_idxs, topk_vals)]

    title_text = "\n".join(title_lines)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.title("Original", fontsize=10)

    plt.subplot(1, 2, 2)
    plt.imshow(x_vis_np)
    plt.axis("off")
    plt.title(title_text, fontsize=8)

    plt.tight_layout()
    plt.show()
