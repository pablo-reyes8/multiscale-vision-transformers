import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import random


def _one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets, num_classes=num_classes).float()


def soft_target_cross_entropy(logits: torch.Tensor, targets_soft: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(targets_soft * logp).sum(dim=1).mean()


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    prob: float = 1.0,):
    """
    Returns:
      images_aug: [B,3,H,W]
      targets_soft: [B,K]
    """
    if prob <= 0.0 or (mixup_alpha <= 0.0 and cutmix_alpha <= 0.0):
        return images, _one_hot(targets, num_classes)

    if random.random() > prob:
        return images, _one_hot(targets, num_classes)

    use_cutmix = (cutmix_alpha > 0.0) and (mixup_alpha <= 0.0 or random.random() < 0.5)
    B, _, H, W = images.shape
    perm = torch.randperm(B, device=images.device)

    y1 = _one_hot(targets, num_classes)
    y2 = _one_hot(targets[perm], num_classes)

    if use_cutmix:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        cut_w = int(W * math.sqrt(1.0 - lam))
        cut_h = int(H * math.sqrt(1.0 - lam))
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)

        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1b = max(cy - cut_h // 2, 0)
        y2b = min(cy + cut_h // 2, H)

        images_aug = images.clone()
        images_aug[:, :, y1b:y2b, x1:x2] = images[perm, :, y1b:y2b, x1:x2]

        # adjust lambda based on actual area swapped
        area = (x2 - x1) * (y2b - y1b)
        lam = 1.0 - area / float(W * H)
    else:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        images_aug = images * lam + images[perm] * (1.0 - lam)

    targets_soft = y1 * lam + y2 * (1.0 - lam)
    return images_aug, targets_soft