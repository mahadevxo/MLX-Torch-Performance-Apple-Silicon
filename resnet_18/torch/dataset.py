# Getting the CIFAR-10 dataset
import numpy as np
import torch.utils.data.dataloader as DataLoader

from torchvision import datasets, transforms

def get_cifar10(batch_size, root="."):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    
    train_dataset = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=True)
    
    train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    
    return train_loader, test_loader