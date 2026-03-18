from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset


DATASET_ALIASES = {
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "svhn": "svhn",
    "oxfordpets": "oxford_pets",
    "oxford_pets": "oxford_pets",
    "oxford-iiit-pet": "oxford_pets",
    "oxfordiiitpet": "oxford_pets",
    "pets": "oxford_pets",
    "food101": "food101",
    "food-101": "food101",
    "tinyimagenet": "tiny_imagenet",
    "tiny-imagenet": "tiny_imagenet",
    "tiny_imagenet": "tiny_imagenet",
}


DATASET_INFO = {
    "cifar100": {
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "kind": "small",
        "use_hflip": True,
        "default_img_size": 32,
        "display_name": "CIFAR-100",
    },
    "svhn": {
        "num_classes": 10,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "kind": "small",
        "use_hflip": False,
        "default_img_size": 32,
        "display_name": "SVHN",
        "classes": [str(i) for i in range(10)],
    },
    "oxford_pets": {
        "num_classes": 37,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "kind": "natural",
        "use_hflip": True,
        "default_img_size": 32,
        "display_name": "Oxford-IIIT Pet",
    },
    "food101": {
        "num_classes": 101,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "kind": "natural",
        "use_hflip": True,
        "default_img_size": 32,
        "display_name": "Food-101",
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2302, 0.2265, 0.2262),
        "kind": "natural",
        "use_hflip": True,
        "default_img_size": 32,
        "display_name": "Tiny ImageNet",
    },
}


def canonicalize_dataset_name(dataset_name: str) -> str:
    if dataset_name is None:
        raise ValueError("dataset_name must not be None")

    key = dataset_name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Available: {sorted(DATASET_ALIASES.keys())}"
        )
    return DATASET_ALIASES[key]


def available_dataset_names():
    return list(DATASET_INFO.keys())


def get_dataset_info(dataset_name: str) -> dict:
    canonical = canonicalize_dataset_name(dataset_name)
    info = dict(DATASET_INFO[canonical])
    info["name"] = canonical
    return info


def _ddp_is_on():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _ddp_rank():
    return torch.distributed.get_rank() if _ddp_is_on() else 0


def _ddp_barrier():
    if _ddp_is_on():
        torch.distributed.barrier()


def _class_names_or_default(dataset, num_classes: int):
    classes = getattr(dataset, "classes", None)
    if classes:
        return list(classes)
    return [str(i) for i in range(num_classes)]


def _attach_dataset_metadata(dataset, dataset_name: str, num_classes: int, mean, std, class_names=None):
    dataset.dataset_name = dataset_name
    dataset.num_classes = num_classes
    dataset.normalization_mean = mean
    dataset.normalization_std = std
    if class_names is not None and not hasattr(dataset, "classes"):
        dataset.classes = list(class_names)
    return dataset


class HFDatasetWrapper(Dataset):
    def __init__(
        self,
        hf_dataset,
        transform=None,
        image_key: str = "image",
        label_key: str = "label",
        dataset_name: str | None = None,
        class_names: list[str] | None = None,
        mean=None,
        std=None,
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.dataset_name = dataset_name
        self.classes = list(class_names) if class_names is not None else None
        self.num_classes = len(self.classes) if self.classes is not None else None
        self.normalization_mean = mean
        self.normalization_std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample[self.image_key]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        label = int(sample[self.label_key])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_transforms(
    dataset_name: str,
    img_size: int = 32,
    ra_num_ops: int = 2,
    ra_magnitude: int = 9,
    random_erasing_p: float = 0.0,
    erasing_scale=(0.02, 0.20),
    erasing_ratio=(0.3, 3.3),
):
    from torchvision import transforms
    from torchvision.transforms import RandAugment

    info = get_dataset_info(dataset_name)
    mean = info["mean"]
    std = info["std"]

    if info["kind"] == "small":
        crop_padding = max(4, img_size // 8)
        train_ops = []
        if img_size != 32:
            train_ops.append(transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC))
        train_ops.append(transforms.RandomCrop(img_size, padding=crop_padding))
        if info["use_hflip"]:
            train_ops.append(transforms.RandomHorizontalFlip())
        train_ops.extend([
            RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if random_erasing_p > 0.0:
            train_ops.append(
                transforms.RandomErasing(
                    p=random_erasing_p,
                    scale=erasing_scale,
                    ratio=erasing_ratio,
                    value="random",
                )
            )

        eval_ops = []
        if img_size != 32:
            eval_ops.append(transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC))
        eval_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transforms.Compose(train_ops), transforms.Compose(eval_ops)

    resize_size = max(img_size + 8, int(img_size * 1.15))
    train_ops = [
        transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(img_size),
    ]
    if info["use_hflip"]:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.extend([
        RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if random_erasing_p > 0.0:
        train_ops.append(
            transforms.RandomErasing(
                p=random_erasing_p,
                scale=erasing_scale,
                ratio=erasing_ratio,
                value="random",
            )
        )

    eval_ops = [
        transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _split_train_val(train_aug, train_eval, val_split: float, seed: int):
    if val_split <= 0.0:
        return train_aug, None

    total = len(train_aug)
    num_val = int(total * val_split)
    num_train = total - num_val
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    train_indices = permutation[:num_train]
    val_indices = permutation[num_train:]

    return Subset(train_aug, train_indices), Subset(train_eval, val_indices)


def _build_torchvision_dataset(dataset_name: str, split: str, data_dir: str, transform, download: bool):
    from torchvision import datasets

    if dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=data_dir,
            train=(split == "train"),
            download=download,
            transform=transform,
        )

    if dataset_name == "svhn":
        return datasets.SVHN(
            root=data_dir,
            split="train" if split == "train" else "test",
            download=download,
            transform=transform,
        )

    if dataset_name == "oxford_pets":
        return datasets.OxfordIIITPet(
            root=data_dir,
            split="trainval" if split == "train" else "test",
            target_types="category",
            download=download,
            transform=transform,
        )

    if dataset_name == "food101":
        return datasets.Food101(
            root=data_dir,
            split="train" if split == "train" else "test",
            download=download,
            transform=transform,
        )

    raise ValueError(f"Torchvision builder not defined for dataset '{dataset_name}'")


def _build_tiny_imagenet_datasets(
    data_dir: str,
    train_transform,
    eval_transform,
    val_split: float,
    seed: int,
    hf_repo: str = "zh-plus/tiny-imagenet",
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "tiny_imagenet requires the Hugging Face 'datasets' package. "
            "Install it with 'pip install datasets'."
        ) from exc

    cache_dir = str(Path(data_dir) / "hf_cache")
    train_hf = load_dataset(hf_repo, split="train", cache_dir=cache_dir)
    valid_hf = load_dataset(hf_repo, split="valid", cache_dir=cache_dir)

    label_feature = train_hf.features.get("label")
    class_names = list(getattr(label_feature, "names", []) or [str(i) for i in range(200)])

    train_aug = HFDatasetWrapper(
        train_hf,
        transform=train_transform,
        dataset_name="tiny_imagenet",
        class_names=class_names,
        mean=DATASET_INFO["tiny_imagenet"]["mean"],
        std=DATASET_INFO["tiny_imagenet"]["std"],
    )
    train_eval = HFDatasetWrapper(
        train_hf,
        transform=eval_transform,
        dataset_name="tiny_imagenet",
        class_names=class_names,
        mean=DATASET_INFO["tiny_imagenet"]["mean"],
        std=DATASET_INFO["tiny_imagenet"]["std"],
    )
    test_dataset = HFDatasetWrapper(
        valid_hf,
        transform=eval_transform,
        dataset_name="tiny_imagenet",
        class_names=class_names,
        mean=DATASET_INFO["tiny_imagenet"]["mean"],
        std=DATASET_INFO["tiny_imagenet"]["std"],
    )

    train_dataset, val_dataset = _split_train_val(train_aug, train_eval, val_split=val_split, seed=seed)
    return train_dataset, val_dataset, test_dataset


def get_classification_datasets(
    dataset_name: str = "cifar100",
    data_dir: str = "./data",
    val_split: float = 0.0,
    ra_num_ops: int = 2,
    ra_magnitude: int = 9,
    random_erasing_p: float = 0.0,
    erasing_scale=(0.02, 0.20),
    erasing_ratio=(0.3, 3.3),
    img_size: int = 32,
    seed: int = 7,
    ddp_safe_download: bool = False,
):
    dataset_name = canonicalize_dataset_name(dataset_name)
    info = get_dataset_info(dataset_name)
    train_transform, eval_transform = build_transforms(
        dataset_name=dataset_name,
        img_size=img_size,
        ra_num_ops=ra_num_ops,
        ra_magnitude=ra_magnitude,
        random_erasing_p=random_erasing_p,
        erasing_scale=erasing_scale,
        erasing_ratio=erasing_ratio,
    )

    if dataset_name == "tiny_imagenet":
        return _build_tiny_imagenet_datasets(
            data_dir=data_dir,
            train_transform=train_transform,
            eval_transform=eval_transform,
            val_split=val_split,
            seed=seed,
        )

    torchvision_download = True
    if ddp_safe_download and _ddp_is_on():
        if _ddp_rank() == 0:
            _build_torchvision_dataset(dataset_name, "train", data_dir, transform=None, download=True)
            _build_torchvision_dataset(dataset_name, "test", data_dir, transform=None, download=True)
        _ddp_barrier()
        torchvision_download = False

    train_aug = _build_torchvision_dataset(
        dataset_name,
        split="train",
        data_dir=data_dir,
        transform=train_transform,
        download=torchvision_download,
    )
    train_eval = _build_torchvision_dataset(
        dataset_name,
        split="train",
        data_dir=data_dir,
        transform=eval_transform,
        download=False,
    )
    test_dataset = _build_torchvision_dataset(
        dataset_name,
        split="test",
        data_dir=data_dir,
        transform=eval_transform,
        download=torchvision_download,
    )

    class_names = _class_names_or_default(train_eval, info["num_classes"])
    _attach_dataset_metadata(train_aug, dataset_name, info["num_classes"], info["mean"], info["std"], class_names)
    _attach_dataset_metadata(train_eval, dataset_name, info["num_classes"], info["mean"], info["std"], class_names)
    _attach_dataset_metadata(test_dataset, dataset_name, info["num_classes"], info["mean"], info["std"], class_names)

    train_dataset, val_dataset = _split_train_val(train_aug, train_eval, val_split=val_split, seed=seed)
    return train_dataset, val_dataset, test_dataset


def get_classification_dataloaders(
    dataset_name: str = "cifar100",
    batch_size: int = 128,
    eval_batch_size: int | None = None,
    data_dir: str = "./data",
    num_workers: int = 2,
    val_split: float = 0.0,
    pin_memory: bool = True,
    ra_num_ops: int = 2,
    ra_magnitude: int = 9,
    random_erasing_p: float = 0.0,
    erasing_scale=(0.02, 0.20),
    erasing_ratio=(0.3, 3.3),
    img_size: int = 32,
    seed: int = 7,
    ddp_safe_download: bool = False,
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_ds, val_ds, test_ds = get_classification_datasets(
        dataset_name=dataset_name,
        data_dir=data_dir,
        val_split=val_split,
        ra_num_ops=ra_num_ops,
        ra_magnitude=ra_magnitude,
        random_erasing_p=random_erasing_p,
        erasing_scale=erasing_scale,
        erasing_ratio=erasing_ratio,
        img_size=img_size,
        seed=seed,
        ddp_safe_download=ddp_safe_download,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_cifar100_datasets(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["dataset_name"] = "cifar100"
    return get_classification_datasets(*args, **kwargs)


def get_cifar100_dataloaders(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["dataset_name"] = "cifar100"
    return get_classification_dataloaders(*args, **kwargs)
