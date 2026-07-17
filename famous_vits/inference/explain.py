"""Grad-CAM and occlusion helpers that work across registered architectures."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("The model does not contain a Conv2d layer for Grad-CAM.")
    return layers[-1]


def grad_cam(
    model: nn.Module,
    inputs: Tensor,
    *,
    class_idx: int | Tensor | None = None,
    target_layer: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    model.eval()
    layer = target_layer or find_last_conv2d(model)
    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def forward_hook(_module, _args, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        indices = (
            logits.argmax(dim=1)
            if class_idx is None
            else torch.as_tensor(class_idx, device=logits.device)
        )
        if indices.ndim == 0:
            indices = indices.expand(logits.shape[0])
        logits.gather(1, indices[:, None]).sum().backward()
        weights = gradients[-1].mean(dim=(-2, -1), keepdim=True)
        cam = (weights * activations[-1]).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        minimum = cam.amin(dim=(-2, -1), keepdim=True)
        maximum = cam.amax(dim=(-2, -1), keepdim=True)
        cam = (cam - minimum) / (maximum - minimum).clamp_min(1e-8)
        return cam.detach(), indices.detach()
    finally:
        forward_handle.remove()
        backward_handle.remove()


@torch.no_grad()
def occlusion_sensitivity(
    model: nn.Module,
    inputs: Tensor,
    *,
    class_idx: int | Tensor | None = None,
    patch_size: int = 4,
    stride: int = 2,
    fill: float = 0.0,
) -> tuple[Tensor, Tensor]:
    model.eval()
    baseline_logits = model(inputs)
    if isinstance(baseline_logits, (tuple, list)):
        baseline_logits = baseline_logits[0]
    baseline_probabilities = baseline_logits.softmax(dim=1)
    indices = (
        baseline_logits.argmax(dim=1)
        if class_idx is None
        else torch.as_tensor(class_idx, device=inputs.device)
    )
    if indices.ndim == 0:
        indices = indices.expand(inputs.shape[0])
    baseline = baseline_probabilities.gather(1, indices[:, None]).squeeze(1)

    height, width = inputs.shape[-2:]
    rows = list(range(0, max(1, height - patch_size + 1), stride))
    cols = list(range(0, max(1, width - patch_size + 1), stride))
    heatmap = inputs.new_zeros((inputs.shape[0], len(rows), len(cols)))
    for row_index, top in enumerate(rows):
        for col_index, left in enumerate(cols):
            occluded = inputs.clone()
            occluded[..., top : top + patch_size, left : left + patch_size] = fill
            logits = model(occluded)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            score = logits.softmax(dim=1).gather(1, indices[:, None]).squeeze(1)
            heatmap[:, row_index, col_index] = baseline - score
    heatmap = F.interpolate(
        heatmap[:, None], size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(1)
    return heatmap, indices
