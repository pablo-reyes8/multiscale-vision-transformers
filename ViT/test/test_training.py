import torch
from torch.utils.data import DataLoader, Dataset

from model.vision_transformer import VisionTransformer
from training.one_epoch import evaluate_one_epoch, train_one_epoch


class TinyDataset(Dataset):
    def __init__(self, num_samples: int = 8, num_classes: int = 10):
        torch.manual_seed(0)
        self.images = torch.randn(num_samples, 3, 32, 32)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def test_train_and_eval_one_epoch_cpu():
    dataset = TinyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loss, train_metrics = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        use_amp=False,
        grad_clip_norm=None,
        label_smoothing=0.0,
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
