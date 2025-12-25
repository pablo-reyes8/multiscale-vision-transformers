import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def occlusion_sensitivity(
    model,
    x,                   # [1,C,H,W]
    target_class=None,   # si None usa la clase predicha
    patch=6,
    stride=3,
    baseline=0.0,
    device="cuda",):
    model.eval().to(device)
    x = x.to(device)

    logits = model(x)
    probs = logits.softmax(dim=1)
    if target_class is None:
        target_class = int(probs.argmax(dim=1).item())

    base_score = float(probs[0, target_class].item())

    _, C, H, W = x.shape
    out_h = 1 + (H - patch) // stride
    out_w = 1 + (W - patch) // stride
    heat = np.zeros((out_h, out_w), dtype=np.float32)

    for i, y0 in enumerate(range(0, H - patch + 1, stride)):
        for j, x0 in enumerate(range(0, W - patch + 1, stride)):
            x_occ = x.clone()
            x_occ[:, :, y0:y0+patch, x0:x0+patch] = baseline
            p = model(x_occ).softmax(dim=1)[0, target_class].item()
            heat[i, j] = base_score - p  # caída de prob

    return base_score, target_class, heat

def plot_occlusion_heatmap(img_chw, heat, patch=6, stride=3):
    img = img_chw.permute(1,2,0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(np.clip(img, 0, 1))
    # upsample heatmap a tamaño imagen (nearest)
    hH, hW = heat.shape
    up = np.kron(heat, np.ones((stride, stride)))
    ax.imshow(up, alpha=0.5)
    ax.set_title("Occlusion sensitivity (higher = more important)")
    ax.axis("off")
    plt.show()