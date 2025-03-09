import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets

def get_train_val_loaders(batch=64, val=0.2,):
    """
    This function will load and split the CIFAR-100 dataset into random subsets w/ 20% validation and 80% training 

    Args:
        batch (int): batch size
        val (float): Percentage of dataset to be used validation (default 20%)

    Returns:
        DataLoaders for both training and validation
    """

    # Normialize the pixel values 
    transform_pipeline = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download/Initalize the dataset
    data_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_pipeline)

    # Calculate the split 
    train_size = int(1 - val) * len(data_set)            #80%
    validation_size = int(len(data_set) - train_set)     #20%

    # initialize the set with corresponing split
    val_set, train_set = random_split(data_set, [train_size, validation_size])

    # Intitiallize DataLoaders
    train_loader = DataLoader(data_set, batch_size=batch, shuffle = True)
    val_loader = DataLoader(data_set, batch_size=batch, shuffle =True)

    return train_loader, val_loader