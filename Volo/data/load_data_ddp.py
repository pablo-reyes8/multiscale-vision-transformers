import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
import torch.distributed as dist
from torch.utils.data import Subset, DataLoader


def _ddp_is_on():
    return dist.is_available() and dist.is_initialized()

def _ddp_rank():
    return dist.get_rank() if _ddp_is_on() else 0

def _ddp_barrier():
    if _ddp_is_on():
        dist.barrier()

def get_cifar100_datasets(
    data_dir: str = "./data",
    val_split: float = 0.0,
    ra_num_ops: int = 2,
    ra_magnitude: int = 7,
    random_erasing_p: float = 0.25,
    erasing_scale=(0.02, 0.20),
    erasing_ratio=(0.3, 3.3),
    img_size: int = 32,
    seed: int = 7,
    ddp_safe_download: bool = True):
    """
    CIFAR-100 datasets con aug 'mix-friendly' y soporte DDP:
      - Descarga segura: solo rank0 descarga, luego barrier.
      - Split determinista: train/val indices iguales en todos los ranks.
      - Val usa test_transform (SIN aug estocásticos).
    """
    if img_size < 32:
        raise ValueError(f"img_size must be >= 32 for CIFAR-100. Got {img_size}.")

    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)

    crop_padding = max(4, img_size // 8)

    train_ops = []
    if img_size != 32:
        train_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    train_ops += [
        transforms.RandomCrop(img_size, padding=crop_padding),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
        transforms.RandomErasing(
            p=random_erasing_p,
            scale=erasing_scale,
            ratio=erasing_ratio,
            value="random",),]
    
    train_transform = transforms.Compose(train_ops)

    test_ops = []
    if img_size != 32:
        test_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    test_ops += [
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),]
    
    test_transform = transforms.Compose(test_ops)

    #  DDP-safe download 
    if ddp_safe_download and _ddp_is_on():
        if _ddp_rank() == 0:
            datasets.CIFAR100(root=data_dir, train=True, download=True)
            datasets.CIFAR100(root=data_dir, train=False, download=True)
        _ddp_barrier()
        download_flag = False
    else:
        download_flag = True

    # Base datasets 
    full_train_aug = datasets.CIFAR100(root=data_dir, train=True, download=download_flag, transform=train_transform)
    full_train_eval = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=test_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=download_flag, transform=test_transform)

    if val_split > 0.0:
        n_total = len(full_train_aug)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        train_dataset = Subset(full_train_aug, train_idx)
        val_dataset = Subset(full_train_eval, val_idx)   
    else:
        train_dataset = full_train_aug
        val_dataset = None

    return train_dataset, val_dataset, test_dataset