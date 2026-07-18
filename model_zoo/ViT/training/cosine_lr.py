import math

import torch.nn as nn


def build_param_groups_no_wd(model: nn.Module, weight_decay: float):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_l = name.lower()
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in name_l
            or name in {"cls_token", "pos_embed"}
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class WarmupCosineLR:
    def __init__(self, optimizer, total_steps: int, warmup_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        step = self.step_num

        for idx, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx]
            if step <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * (step / self.warmup_steps)
            else:
                capped_step = min(step, self.total_steps)
                denom = max(1, self.total_steps - self.warmup_steps)
                progress = (capped_step - self.warmup_steps) / denom
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine

            group["lr"] = lr

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, state_dict):
        self.step_num = int(state_dict.get("step_num", 0))
