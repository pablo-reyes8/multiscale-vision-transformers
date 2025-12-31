
from typing import Dict
import torch
import torch.nn.functional as F


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3, 5)) -> Dict[int, float]:
    """
    targets can be:
      - int64 class indices [B]
      - soft targets [B, num_classes] (we'll argmax for accuracy reporting)
    """
    if targets.ndim == 2:
        targets = targets.argmax(dim=1)

    max_k = max(ks)
    B = targets.size(0)
    _, pred = torch.topk(logits, k=max_k, dim=1)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    out = {}
    for k in ks:
        out[k] = 100.0 * correct[:, :k].any(dim=1).float().sum().item() / B
    return out