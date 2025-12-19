import torch
from torchvision import datasets

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def load_model_state(model, checkpoint_path: str, device: str = "cpu"):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def resolve_class_names(dataloader=None, data_dir: str = "./data"):
    if dataloader is not None:
        dataset = dataloader.dataset
        if hasattr(dataset, "classes"):
            return list(dataset.classes)
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "classes"):
            return list(dataset.dataset.classes)

    ds = datasets.CIFAR100(root=data_dir, train=False, download=True)
    return list(ds.classes)


def unnormalize(images: torch.Tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD):
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std + mean
