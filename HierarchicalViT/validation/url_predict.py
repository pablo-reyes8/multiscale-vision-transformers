import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch
from training.autocast import *
import matplotlib.pyplot as plt


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

cifar100_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

def load_image_from_url(url: str) -> Image.Image:
    """
    Descarga una imagen desde un URL y la devuelve como PIL.Image.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img


@torch.no_grad()
def predict_from_url(
    model,
    url: str,
    class_names,
    device="cuda",
    autocast_dtype="fp16",
    use_amp=True,
    transform=cifar100_test_transform,
    topk=5,
):
    """
    Descarga una imagen desde la URL, la preprocesa como CIFAR-100,
    hace forward y devuelve top-k predicciones.

    Returns:
        topk_idxs: np.array [k] con índices de clase
        topk_vals: np.array [k] con probabilidades
        img_pil: imagen original (PIL)
        x: tensor [1, C, H, W] normalizado que se le pasó al modelo
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

    print(f"Predicciones para: {url}\n")
    for p, idx in zip(topk_vals, topk_idxs):
        print(f"{class_names[int(idx)]:>15s}: {p*100:5.2f}%")

    return topk_idxs, topk_vals, img_pil, x


def unnormalize(images: torch.Tensor,
                mean= (0.5071, 0.4867, 0.4408),
                std = (0.2675, 0.2565, 0.2761)):
    """
    Des-normaliza un batch de imágenes.
    images: [B, C, H, W] o [C, H, W] (en ese caso se añade batch dim).
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)

    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def tensor_to_img(x_tensor: torch.Tensor):
    """
    Convierte un tensor [C, H, W] en imagen numpy (0-1) para plt.imshow.
    """
    assert x_tensor.dim() == 3, f"Se esperaba [C,H,W], llegó {x_tensor.shape}"
    x = x_tensor.clamp(0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    return x


def show_url_prediction_pair(
    img_pil,
    x_tensor_norm,
    topk_idxs,
    topk_vals,
    class_names):
    """
    Muestra:
      - izquierda: imagen original (PIL)
      - derecha: imagen preprocesada (32x32) que vio el modelo (des-normalizada)
      - título derecho: top-k predicciones con probabilidades.
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