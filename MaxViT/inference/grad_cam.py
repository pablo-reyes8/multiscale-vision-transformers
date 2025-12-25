import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found in model.")
    return last

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.grads = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # [B,C,H,W]

    def _backward_hook(self, module, grad_inp, grad_out):
        self.grads = grad_out[0]  # [B,C,H,W]

    def close(self):
        self.h1.remove(); self.h2.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        A = self.activations  # [B,C,H,W]
        G = self.grads        # [B,C,H,W]
        w = G.mean(dim=(2,3), keepdim=True)   # [B,C,1,1]
        cam = (w * A).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-12)
        return cam.detach(), class_idx

def overlay_cam(img_chw, cam_1hw, alpha=0.5):
    img = img_chw.permute(1,2,0).cpu().numpy()
    cam = cam_1hw[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(np.clip(img, 0, 1))
    ax.imshow(cam, alpha=alpha)
    ax.axis("off")
    plt.show()