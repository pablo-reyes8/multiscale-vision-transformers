
import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3, 5)):
    with torch.no_grad():
        max_k = max(ks)
        B = targets.size(0)
        _, pred = torch.topk(logits, k=max_k, dim=1)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        out = {}
        for k in ks:
            out[k] = 100.0 * correct[:, :k].any(dim=1).float().sum().item() / B
        return out