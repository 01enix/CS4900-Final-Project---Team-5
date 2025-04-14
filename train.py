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
from model import Net
from linearModel import LinearNet


# enable the switching between cifar classes
class CIFAR_class_switch(torchvision.datasets.CIFAR100):
    def __init__(self, class_type='100', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_type == '20':
            self.targets = self.coarse_labels
        else:
            self.targets = self.targets
        
def get_train_val_loaders(batch=64, val=0.2,class_type='100'):
    """
    This function will preprocess the data, creating mini-batches of 64 and splitting the CIFAR-100 dataset into random subsets with 20% validation and 80% training 

    Args:
        batch: batch size (default 64)
        val: Percentage of dataset to be used for validation (default 20%)
        class_type(str): '100' or '20', used to select which labels are used from the CIFAR dataset
    Returns:
        DataLoaders for both training and validation
    """
    #transform without normalization
    transform = T.Compose([
    T.ToTensor()
    ])

    # Download/Initialize the dataset
    data_set = CIFAR_class_switch(root='./data', train=True, download=True, transform=transform, class_type=class_type)

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


def train(model,learning_rate=0.001,num_epochs=20,class_type='100'):
    """
    This function trains the CNN model

    Args:
        model: sets which CNN model will be used in the function 
        learning_rate: sets the learning rate, which adjusts how much the model will adjust its weight during each iteration of training(default 0.001)
        num_epochs: sets the number of iterations through the dataset for the model(default 20)
        class_type(str): '100' or '20', used to select which labels are used from the CIFAR dataset(default 100)

    Returns:
        Lists of training, accuracy, and validation losses per epoch.

    """
    # data loading
    train_loader, val_loader = get_train_val_loaders(class_type=class_type)

    #enable GPU (CUDA) usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 100 if class_type == '100' else 20
    net = model(num_classes=num_classes).to(device)

    # create summary writer for tensorboard
    writer = SummaryWriter(log_dir=f'runs/cifar100{class_type}_{timestamp}')

    #define optimizer and loss - DNN_basics - Slide 17
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    #training epochs loop from CNN_basics
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0  
        train_correct = 0
        train_total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        #caluculate average training loss 
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        #validation loop
        net.eval()          #ensure that dropout is turned off
        val_loss =0.0       #used to track average validation loss
        total = 0           #tracks the number of images processed
        correct = 0         #tracks the number of correct images predicted

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        #calculate validation loss/validation accuracy 
        avg_val_loss = val_loss / len(train_loader)
        val_accuracy = 100 * correct / total

        #at the end of each epoch, print loss (training set) and accuracy (val set for B)

        # save loss and / or accuracy after each epoch in the summary writer
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss:{avg_val_loss:.4f},Validation Accuracy: {val_accuracy:.2f}%')
        writer.add_scalar('Loss/epoch_train', avg_train_loss, epoch)
        writer.add_scalar("Loss/epoch_val", avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    #save the model
    model_save = f'cnn_cifar100_{timestamp}.pth' #save the model and place it in the current working directory
    torch.save(net.state_dict(), model_save)     #save parameters for model 
    print(f"Model saved to {model_save}")

    #close the writer
    writer.close()

    #allow adjusting of parameters 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_type', type=str, default='100', choices=['100', '20'],help="Choose number of output classes: '100' for class or '20' for superclasses (default: 100)")
    parser.add_argument('--model', type=str, default='Net', help='Choose model: Net or LinearNet(default: Net)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    args = parser.parse_args()

    # turn string into model class
    if args.model.lower() == 'net':
        selected_model = Net
    elif args.model.lower() == 'linearnet':
        selected_model = LinearNet
    else:
        raise ValueError("Model must be 'Net' or 'LinearNet'")

    train(model=selected_model, learning_rate=args.lr, num_epochs=args.epochs,class_type=args.class_type)
