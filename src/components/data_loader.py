# src/components/data_loader.py

import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# =========================================================
# Device helper (MPS-safe)
# =========================================================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================================================
# Augmentations
# =========================================================

def get_ssl_augmentations(image_size: int = 224) -> transforms.Compose:
    """
    Strong augmentations for contrastive learning.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_supervised_augmentations(train: bool, image_size: int = 224) -> transforms.Compose:
    """
    Moderate augmentations for supervised learning.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# =========================================================
# Dataset: SSL (returns 2 augmented views, no label)
# =========================================================

class SSLImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path):
                    self.image_paths.append(path)

        if not self.image_paths:
            raise ValueError(f"No images found in SSL dataset directory: {root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        view_1 = self.transform(image)
        view_2 = self.transform(image)

        return view_1, view_2


# =========================================================
# Dataset: Supervised (returns image + label)
# =========================================================

class SupervisedImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose):
        self.root_dir = root_dir
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = {}

        classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)

            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                if os.path.isfile(path):
                    self.samples.append((path, idx))

        if not self.samples:
            raise ValueError(f"No images found in supervised dataset directory: {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


# =========================================================
# DataLoader factories
# =========================================================

def create_ssl_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2
) -> DataLoader:
    dataset = SSLImageDataset(
        root_dir=data_dir,
        transform=get_ssl_augmentations()
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )


def create_supervised_dataloader(
    data_dir: str,
    batch_size: int,
    train: bool,
    num_workers: int = 2
) -> Tuple[DataLoader, dict]:
    dataset = SupervisedImageDataset(
        root_dir=data_dir,
        transform=get_supervised_augmentations(train=train)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False
    )

    return dataloader, dataset.class_to_idx
