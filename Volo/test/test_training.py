import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.mixup_cutmix_loss import apply_mixup_cutmix
from training.train_one_epoch import train_one_epoch, evaluate_one_epoch


def _make_loader(num_samples=8, num_classes=10, img_size=8, batch_size=4):
    images = torch.randn(num_samples, 3, img_size, img_size)
    targets = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(images, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_apply_mixup_cutmix_noop():
    images = torch.randn(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 3])

    images_aug, targets_soft = apply_mixup_cutmix(
        images,
        targets,
        num_classes=5,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        prob=1.0,
    )

    assert torch.equal(images_aug, images)
    assert targets_soft.shape == (4, 5)
    assert torch.allclose(targets_soft.sum(dim=1), torch.ones(4))


def test_apply_mixup_cutmix_mixup_path():
    torch.manual_seed(0)
    random.seed(0)

    images = torch.randn(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 3])

    images_aug, targets_soft = apply_mixup_cutmix(
        images,
        targets,
        num_classes=5,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        prob=1.0,
    )

    one_hot = F.one_hot(targets, num_classes=5).float()
    assert images_aug.shape == images.shape
    assert targets_soft.shape == (4, 5)
    assert torch.allclose(targets_soft.sum(dim=1), torch.ones(4))
    assert not torch.allclose(targets_soft, one_hot)


def test_train_one_epoch_cpu_updates_weights():
    torch.manual_seed(0)
    loader = _make_loader(num_samples=8, num_classes=10, img_size=8, batch_size=4)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 10),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[1].weight.detach().clone()

    loss, metrics = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        scaler=None,
        autocast_dtype="fp16",
        use_amp=False,
        grad_clip_norm=None,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mix_prob=1.0,
        num_classes=10,
        channels_last=False,
        print_every=0,
    )

    after = model[1].weight.detach().clone()
    assert isinstance(loss, float)
    assert set(metrics.keys()) == {"top1", "top3", "top5"}
    assert not torch.allclose(before, after)


def test_evaluate_one_epoch_cpu():
    torch.manual_seed(0)
    loader = _make_loader(num_samples=8, num_classes=10, img_size=8, batch_size=4)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 10),
    )

    loss, metrics = evaluate_one_epoch(
        model=model,
        dataloader=loader,
        device="cpu",
        autocast_dtype="fp16",
        use_amp=False,
        label_smoothing=0.0,
        channels_last=False,
    )

    assert isinstance(loss, float)
    assert set(metrics.keys()) == {"top1", "top3", "top5"}
