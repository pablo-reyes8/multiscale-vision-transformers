import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

def build_param_groups_no_wd(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # no decay for biases + norm scales (LN/BN/etc)
        if name.endswith(".bias") or ("norm" in name.lower()) or ("bn" in name.lower()) or ("ln" in name.lower()):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},]

class WarmupCosineLR:
    """Warmup linear for warmup_steps, then cosine to min_lr. Step-based."""
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        t = self.step_num

        for i, group in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i]
            if t <= self.warmup_steps and self.warmup_steps > 0:
                lr = base * (t / self.warmup_steps)
            else:
                tt = min(t, self.total_steps)
                denom = max(1, self.total_steps - self.warmup_steps)
                progress = (tt - self.warmup_steps) / denom
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base - self.min_lr) * cosine
            group["lr"] = lr

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, d):
        self.step_num = int(d.get("step_num", 0))