import torch
import pytest
from torch.utils.data import Subset

from data import load_data_ddp


class FakeCIFAR100(torch.utils.data.Dataset):
    def __init__(self, root, train, download, transform=None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.data = torch.zeros(20, 3, 32, 32)
        self.targets = torch.arange(20) % 100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])


def test_get_cifar100_datasets_val_split_deterministic(monkeypatch):
    monkeypatch.setattr(load_data_ddp.datasets, "CIFAR100", FakeCIFAR100)

    train_ds1, val_ds1, _ = load_data_ddp.get_cifar100_datasets(
        data_dir="/tmp/cifar100",
        val_split=0.2,
        img_size=32,
        seed=123,
        ddp_safe_download=False,
    )

    train_ds2, val_ds2, _ = load_data_ddp.get_cifar100_datasets(
        data_dir="/tmp/cifar100",
        val_split=0.2,
        img_size=32,
        seed=123,
        ddp_safe_download=False,
    )

    assert isinstance(train_ds1, Subset)
    assert isinstance(val_ds1, Subset)
    assert len(train_ds1) == 16
    assert len(val_ds1) == 4
    assert train_ds1.indices == train_ds2.indices
    assert val_ds1.indices == val_ds2.indices
    assert train_ds1.dataset is not val_ds1.dataset


def test_get_cifar100_datasets_download_flags(monkeypatch):
    calls = []

    class CaptureCIFAR100(FakeCIFAR100):
        def __init__(self, root, train, download, transform=None):
            calls.append({"train": train, "download": download})
            super().__init__(root, train, download, transform=transform)

    monkeypatch.setattr(load_data_ddp.datasets, "CIFAR100", CaptureCIFAR100)

    load_data_ddp.get_cifar100_datasets(
        data_dir="/tmp/cifar100",
        val_split=0.1,
        img_size=32,
        ddp_safe_download=False,
    )

    assert len(calls) == 3
    assert calls[0]["download"] is True
    assert calls[1]["download"] is False
    assert calls[2]["download"] is True


def test_get_cifar100_datasets_invalid_img_size(monkeypatch):
    monkeypatch.setattr(load_data_ddp.datasets, "CIFAR100", FakeCIFAR100)

    with pytest.raises(ValueError):
        load_data_ddp.get_cifar100_datasets(img_size=31)
