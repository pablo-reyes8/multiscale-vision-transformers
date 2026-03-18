import torch
from torchvision import datasets

from data.dataset_zoo import get_dataset_info

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def load_model_state(model, checkpoint_path: str, device: str = "cpu"):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def resolve_class_names(dataloader=None, data_dir: str = "./data", dataset_name: str = "cifar100"):
    if dataloader is not None:
        dataset = dataloader.dataset
        if hasattr(dataset, "classes"):
            return list(dataset.classes)
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
            return list(dataset.dataset.classes)

    info = get_dataset_info(dataset_name)
    if "classes" in info:
        return list(info["classes"])

    if dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=data_dir, train=False, download=True)
        return list(ds.classes)

    return [str(i) for i in range(info["num_classes"])]


def resolve_normalization_stats(dataset=None, dataset_name: str = "cifar100"):
    if dataset is not None:
        if hasattr(dataset, "normalization_mean") and hasattr(dataset, "normalization_std"):
            return dataset.normalization_mean, dataset.normalization_std
        if hasattr(dataset, "dataset"):
            base = dataset.dataset
            if hasattr(base, "normalization_mean") and hasattr(base, "normalization_std"):
                return base.normalization_mean, base.normalization_std

    info = get_dataset_info(dataset_name)
    return info["mean"], info["std"]


def unnormalize(images: torch.Tensor, mean=None, std=None, dataset=None, dataset_name: str = "cifar100"):
    if mean is None or std is None:
        mean, std = resolve_normalization_stats(dataset=dataset, dataset_name=dataset_name)
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std + mean
