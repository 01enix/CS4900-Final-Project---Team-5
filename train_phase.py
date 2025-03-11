#Jaden Livingston

# Import necessary modules
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

def train_phase(model, train_loader, criterion, optimizer, writer, epoch, scheduler):
    """
    In this function we'll train one epoch at a time, by doing forward passes and backward passes, calculating the loss, and optimizing the parameters.

    Args:
        model (nn.Module): The model being trained.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): The loss function used to compute the error.
        optimizer (optim.Optimizer): The optimizer used to update the model's weights.
        writer (SummaryWriter): TensorBoard writer for logging metrics during training.
        epoch (int): The current epoch number. Used for logging.
        scheduler (lr_scheduler): The learning rate scheduler that adjusts the learning rate.

    Returns:
        None
    """
    model.train()  # Setting the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # for loop for iterating through the data
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients

        # the forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # the backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate training accuracy
    train_accuracy = 100 * correct / total


    writer.add_scalar("Training Loss", running_loss / len(train_loader), epoch)
    writer.add_scalar("Training Accuracy", train_accuracy, epoch)

    # Adjust the learning rate
    scheduler.step()

   
    print(f"Epoch [{epoch+1}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Training Accuracy: {train_accuracy:.2f}%") 
    

    tydytdytdtrdtr
