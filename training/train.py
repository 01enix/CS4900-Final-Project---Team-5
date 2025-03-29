import argparse
import datetime
import numpy as np

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import os


def get_train_val_loaders(data_dir, batch=64, val=0.2):
    """
    Load custom dataset, preprocess, and create train/val loaders.

    Args:
        data_dir (str): Path to dataset root folder.
        batch (int): Batch size (default 64).
        val (float): Percentage of dataset for validation (default 20%).

    Returns:
        train_loader, val_loader
    """
    transform_pipeline = T.Compose([
        T.Resize((224, 224)),  # Resize images if needed
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform_pipeline)

    # Split into train/val sets
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    split = int(np.floor(val * num_samples))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch, sampler=val_sampler)

    return train_loader, val_loader

# Define timestamp to save models and logs
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#CNN model will be used for training 
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(data_dir, learning_rate=0.001, num_epochs=20):
    """
    This function trains the CNN model using a custom dataset.

    Args:
        data_dir (str): Path to the dataset root folder.
        learning_rate (float): Learning rate for training (default: 0.001).
        num_epochs (int): Number of epochs for training (default: 20).

    Returns:
        None
    """

    # Load dataset
    train_loader, val_loader = get_train_val_loaders(data_dir)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model (number of classes determined from dataset)
    num_classes = len(train_loader.dataset.classes)
    net = Net(num_classes=num_classes).to(device)

    # Create summary writer for TensorBoard logs
    writer = SummaryWriter(log_dir=f'runs/custom_dataset_{timestamp}')

    # Define optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0  

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        net.eval()
        val_loss = 0.0  
        total = 0  
        correct = 0  

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # Print and log metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        writer.add_scalar('Loss/epoch_train', avg_train_loss, epoch)
        writer.add_scalar("Loss/epoch_val", avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    # Save the trained model
    model_save = f'cnn_custom_dataset_{timestamp}.pth'
    torch.save(net.state_dict(), model_save)
    print(f"Model saved to {model_save}")

    # Close the TensorBoard writer
    writer.close()

    #allow adjusting of parameters 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root folder')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')

    args = parser.parse_args()  
    train(data_dir=args.data_dir, learning_rate=args.lr, num_epochs=args.epochs)