from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_loaders(data_dir: str, batch_size: int, num_workers: int, seed: int, val_ratio: float = 0.1):
    data_dir = Path(data_dir)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.25),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    full_val = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_subset, _ = random_split(full_train, [train_size, val_size], generator=gen)
    _, val_subset = random_split(full_val, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
