import torchvision.transforms as transforms
from torchvision import datasets
import torch

from yaml_parser import TrainingConfig

import os

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0, 1)


data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

sim_clr = transforms.Compose([
            transforms.RandomResizedCrop(size=(224,224), scale=(0.2, 1.0)), # compatible size with pretrained models
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomAffine(
                degrees=15, 
                translate=(0.1, 0.1), 
                scale=(0.8, 1.2), 
                shear=10
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(mean=0.0, std=0.1)
        ])


class SelfSupervisedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.stack([self.transform(x), self.transform(x)], dim=0)


def get_dataloaders(config: TrainingConfig):
    supervised = config.model.supervised
    data_config = config.data

    if data_config.transform == None or data_config.transform.lower() == "basic":
        transform = data_transforms
    elif data_config.transform.lower() == "simclr":
        transform = sim_clr
    
    if not supervised:
        transform = SelfSupervisedTransform(transform)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_config.directory, "train_images"), transform=transform),
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_config.directory, "val_images"), transform=transform),
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    return train_loader, val_loader

