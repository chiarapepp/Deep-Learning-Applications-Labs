import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
import numpy as np

"""
Creates DataLoaders for training, validation, and test sets for the specified dataset. 
Supported datasets: MNIST, CIFAR10, CIFAR100.

Args:
    name (str): Dataset name ("MNIST", "CIFAR10", or "CIFAR100").
    batch_size (int): Number of samples per batch.
    num_workers (int): Number of subprocesses for data loading.
    val_ratio (float, optional): Fraction of training data to use for validation. Default is 0.1.

Returns:
    train_dataloader, val_dataloader, test_dataloader (DataLoader): DataLoaders for training, validation, and test sets.
    classes (int): Number of classes in the dataset.
    input_size (int): Flattened input size of a single sample.
"""

def get_dataloaders(name, batch_size, num_workers, val_ratio=0.1):
    ds_train, ds_test, classes, input_size = None, None, 0, 0

    if name == "MNIST":
        # Convert images to tensor and normalize with MNIST mean and std (grayscale)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  
        ])
        ds_train = MNIST(root='./data', train=True, download=True, transform=transform)
        ds_test = MNIST(root='./data', train=False, download=True, transform=transform)
        classes = 10
        input_size = 28 * 28

    elif name == "CIFAR10":
        # Convert images to tensor and normalize with CIFAR-10 mean and std (RGB), added augmentations
        transform_augm_train = transforms.Compose([            
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        # No augmentation for test set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform_augm_train)
        ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
        classes = 10
        input_size = 32 * 32 * 3

    elif name == "CIFAR100":
        # Convert images to tensor and normalize with CIFAR-100 mean and std (RGB)
        transform_augm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        # No augmentation for test set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        ds_train = CIFAR100(root='./data', train=True, download=True, transform=transform_augm_train)
        ds_test = CIFAR100(root='./data', train=False, download=True, transform=transform)
        classes = 100
        input_size = 32 * 32 * 3

    else:
        raise ValueError(f"Dataset {name} not supported.")

    # Split train into train and validation based on val_ratio (instead of using a fixed size)
    val_size = int(len(ds_train)* val_ratio)
    indices = np.random.permutation(len(ds_train))
    ds_val = Subset(ds_train, indices[:val_size])
    ds_train = Subset(ds_train, indices[val_size:])
    
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, classes, input_size
