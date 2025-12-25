import torch
from torch.utils.data import DataLoader, Dataset

from data.data_utils import describe_loader
import data.load_cifrar100 as cifar


class TinyDataset(Dataset):
    def __init__(self):
        self.targets = [0, 1, 1, 2]
        self._data = [torch.zeros(3, 4, 4) for _ in self.targets]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self._data[idx], self.targets[idx]


def test_describe_loader_outputs_summary(capsys):
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
    describe_loader(loader, name="train", max_batches_for_stats=1)
    output = capsys.readouterr().out
    assert "TRAIN SUMMARY" in output
    assert "Num samples" in output
    assert "Full dataset label distribution" in output


class DummyCIFAR100(Dataset):
    def __init__(self, root, train, download, transform):
        self.train = train
        self.transform = transform
        n = 50 if train else 20
        self.targets = list(range(n))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.zeros(3, 32, 32)
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def test_get_cifar100_datasets_and_loaders(monkeypatch):
    monkeypatch.setattr(cifar.datasets, "CIFAR100", DummyCIFAR100)

    train_ds, val_ds, test_ds = cifar.get_cifar100_datasets(
        data_dir="./tmp", val_split=0.2
    )
    assert len(train_ds) == 40
    assert len(val_ds) == 10
    assert len(test_ds) == 20

    train_loader, val_loader, test_loader = cifar.get_cifar100_dataloaders(
        batch_size=8,
        data_dir="./tmp",
        num_workers=0,
        val_split=0.2,
    )

    assert val_loader is not None
    assert len(train_loader.dataset) == 40
    assert len(val_loader.dataset) == 10
    assert len(test_loader.dataset) == 20


def test_get_cifar100_dataloaders_no_val(monkeypatch):
    monkeypatch.setattr(cifar.datasets, "CIFAR100", DummyCIFAR100)

    train_loader, val_loader, test_loader = cifar.get_cifar100_dataloaders(
        batch_size=8,
        data_dir="./tmp",
        num_workers=0,
        val_split=0.0,
    )

    assert val_loader is None
    assert len(train_loader.dataset) == 50
    assert len(test_loader.dataset) == 20
