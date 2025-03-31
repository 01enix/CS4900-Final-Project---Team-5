import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

def get_test_loader(batch=64):
    """
    This function loads and preprocesses the CIFAR-100 test dataset.

    Args:
        batch (int): Batch size for testing (default 64).

    Returns:
        DataLoader for the test dataset.
    """
    # For consistency, we will use the same preprocessing as used in training
    # Define the transformation pipeline for the test dataset
    transform_pipeline = T.Compose([
        T.ToTensor()
    ])

    # Load CIFAR-100 test dataset
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_pipeline)

    # Initialize DataLoader
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    return test_loader