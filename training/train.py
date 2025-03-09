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


def get_train_val_loaders(batch=64, val=0.2,):
    """
    This function will preprocess the data; creating mini batches of 64 and splitting the CIFAR-100 dataset into random subsets with 20% validation and 80% training 

    Args:
        batch (int): batch size (default 64)
        val (float): Percentage of dataset to be used for validation (default 20%)

    Returns:
        DataLoaders for both training and validation
    """

    # Normialize the pixel values 
    transform_pipeline = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download/Initialize the dataset
    data_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_pipeline)

    # Create indices and shuffle for the SubsetRandonSampler()
    num_samples = len(data_set)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    #split data_set 80/20
    split = int(np.floor(val * num_samples))
    train_indices, val_indices = indices[split:], indices[:split]

   #initialize samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Initialize DataLoaders
    train_loader = DataLoader(data_set, batch_size=batch, sampler=train_sampler)
    val_loader = DataLoader(data_set, batch_size=batch, sampler=val_sampler)

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

def train(learning_rate=0.001,num_epochs=20):
    """
    This function trains the CNN model

    Args:
        learning_rate: sets the learning rate which adjusts how much the model will adjust its weight during each iteration of training
        num_epochs: sets the number of iterations through the dataset for the model

    Returns:
        Lists of training and validation losses per epoch.

    """
    # data loading
    train_loader, val_loader = get_train_val_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #enables the use of a GPU if avaliable 
    net=Net(num_classes=100).to(device)

    # create summary writer for tensorboard
    writer = SummaryWriter(log_dir=f'runs/cifar100_{timestamp}')

    #define optimizer and loss - DNN_basics - Slide 17
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    #training epochs loop from CNN_basics
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step
            running_loss += loss.item()

        #caluculate average training loss 
        avg_train_loss = running_loss / len(train_loader)

        #validation loop
        net.eval()          #ensure that dropout is turned off
        total = 0           #tracks the number of images processed
        correct = 0         #tracks the number of correct images predicted
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        #calculate validation accuracy
        val_accuracy = 100 * correct / total

        #at the end of each epoch, print loss (training set) and accuracy (val set for B)
        # save loss and/or accuracy after each epoch in the summary writer
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        writer.add_scalar('Loss/epoch_train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    #save the model
    model_save = f'cnn_cifar100_{timestamp}.pth' #save the model and place it in current working directory
    torch.save(net.state_dict(), model_save)     #save parameters for model 
    print(f"Model saved to {model_save}")

    #close the writer
    writer.close()

    #allow adjusting of parameters 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    args = parser.parse_args()
    train(learning_rate= args.lr, num_epochs = args.epochs)





           




