import torch

from model.MaxViT import MaxViT
from model_configurations import maxvit_cifar100_tiny


def test_maxvit_forward_shape():
    torch.manual_seed(0)
    cfg = maxvit_cifar100_tiny(stem_type="A", drop_path_rate=0.0)
    model = MaxViT(cfg)

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 100)


def test_maxvit_backward():
    torch.manual_seed(0)
    cfg = maxvit_cifar100_tiny(stem_type="B", drop_path_rate=0.1)
    model = MaxViT(cfg)

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = model(x).sum()
    y.backward()
    assert x.grad is not None
