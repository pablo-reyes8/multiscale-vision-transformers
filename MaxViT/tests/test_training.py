import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.mixup_cutmix import apply_mixup_cutmix, soft_target_cross_entropy
from training.metrics import accuracy_topk
from training.train_one_epoch import train_one_epoch, evaluate_one_epoch


def test_apply_mixup_cutmix_prob_zero_identity():
    images = torch.randn(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 3])

    images_aug, targets_soft = apply_mixup_cutmix(
        images,
        targets,
        num_classes=5,
        mixup_alpha=0.2,
        cutmix_alpha=0.0,
        prob=0.0,
    )

    assert torch.equal(images_aug, images)
    assert targets_soft.shape == (4, 5)
    assert torch.allclose(targets_soft.sum(dim=1), torch.ones(4))


def test_apply_mixup_cutmix_shapes_and_probabilities():
    torch.manual_seed(0)
    images = torch.randn(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 2, 3])

    images_aug, targets_soft = apply_mixup_cutmix(
        images,
        targets,
        num_classes=5,
        mixup_alpha=0.4,
        cutmix_alpha=0.4,
        prob=1.0,
    )

    assert images_aug.shape == images.shape
    assert targets_soft.shape == (4, 5)
    assert torch.allclose(targets_soft.sum(dim=1), torch.ones(4))


def test_soft_target_cross_entropy_low_for_easy_logits():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets_soft = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = soft_target_cross_entropy(logits, targets_soft)
    assert loss.item() < 1e-3


def test_accuracy_topk_with_soft_targets():
    logits = torch.tensor([[5.0, 0.0, -1.0], [0.0, 5.0, -1.0]])
    targets = torch.tensor([0, 1])
    targets_soft = torch.eye(3)[targets]

    acc_hard = accuracy_topk(logits, targets, ks=(1, 3))
    acc_soft = accuracy_topk(logits, targets_soft, ks=(1, 3))

    assert acc_hard[1] == 100.0
    assert acc_soft[1] == 100.0
    assert acc_hard[3] == 100.0
    assert acc_soft[3] == 100.0


class TinyDataset(Dataset):
    def __init__(self, num_samples=8, num_classes=10):
        torch.manual_seed(0)
        self.images = torch.randn(num_samples, 3, 8, 8)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


class TinyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def test_train_and_eval_one_epoch_cpu():
    dataset = TinyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = TinyNet(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_loss, train_metrics = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        use_amp=False,
        grad_clip_norm=None,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        num_classes=10,
        print_every=0,
    )

    eval_loss, eval_metrics = evaluate_one_epoch(
        model=model,
        dataloader=loader,
        device="cpu",
        use_amp=False,
        label_smoothing=0.0,
    )

    assert train_loss > 0.0
    assert eval_loss > 0.0
    assert set(train_metrics.keys()) == {"top1", "top3", "top5"}
    assert set(eval_metrics.keys()) == {"top1", "top3", "top5"}
