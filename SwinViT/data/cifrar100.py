import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment


def get_cifar100_datasets(data_dir: str = "./data", val_split: float = 0.0):
    """
    Descarga (si hace falta) y devuelve los datasets de CIFAR-100:
    - train_dataset (o train_rest, val_dataset si val_split > 0)
    - test_dataset

    Args:
        data_dir: carpeta donde se guardan/descargan los datos.
        val_split: proporción del train que se reserva como validación (0.0 = sin validación).

    Returns:
        Si val_split == 0.0:
            train_dataset, None, test_dataset
        Si val_split > 0.0:
            train_dataset, val_dataset, test_dataset
    """

    # Stats típicas de CIFAR-100
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      RandAugment(num_ops=2, magnitude=9),
      transforms.ToTensor(),
      transforms.Normalize(cifar100_mean, cifar100_std),])

    # Transformaciones para test/val (solo resize + normalize)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),])

    # Dataset de training
    full_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,)

    # Dataset de test
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform)

    if val_split > 0.0:
        n_total = len(full_train_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(7))

    else:
        train_dataset = full_train_dataset
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def get_cifar100_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    val_split: float = 0.0,
    pin_memory: bool = True):
    """
    Devuelve DataLoaders para CIFAR-100.

    Returns:
        train_loader, val_loader, test_loader
        (val_loader será None si val_split == 0.0)
    """
    train_ds, val_ds, test_ds = get_cifar100_datasets(
        data_dir=data_dir,
        val_split=val_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,)

    else:
        val_loader = None

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)

    return train_loader, val_loader, test_loader