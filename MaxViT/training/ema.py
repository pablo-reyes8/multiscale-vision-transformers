import copy
from contextlib import contextmanager
import torch
import torch.nn as nn



class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str | None = None):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        self.device = device

        if device is not None:
            self.ema.to(device)

        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.buffer_keys = set()
        for name, _ in model.named_buffers():
            self.buffer_keys.add(name)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        d = self.decay

        for k, v_ema in esd.items():
            if k not in msd:
                continue

            v = msd[k].detach()
            if self.device is not None:
                v = v.to(self.device)

            if k in self.buffer_keys:
                v_ema.copy_(v)
                continue

            if torch.is_floating_point(v_ema):
                v_ema.mul_(d).add_(v.to(dtype=v_ema.dtype), alpha=(1.0 - d))
            else:
                v_ema.copy_(v)

    @contextmanager
    def use_ema_weights(self, model: nn.Module):
        orig = {k: t.detach().clone() for k, t in model.state_dict().items()}
        model.load_state_dict(self.ema.state_dict(), strict=True)
        try:
            yield
        finally:
            model.load_state_dict(orig, strict=True)

    def state_dict(self):
        return {"decay": self.decay, "model": self.ema.state_dict()}

    def load_state_dict(self, d):
        self.decay = float(d.get("decay", self.decay))
        self.ema.load_state_dict(d["model"], strict=True)