#Jaden Livingston

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def val_phase(model, val_loader, criterion, writer, epoch):
    """
    This function performs one epoch of validation, evaluating the model's performance on the validation set. 
    It computes the loss and accuracy of the model on the validation data and logs these metrics to TensorBoard.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): The loss function used to compute the error.
        writer (SummaryWriter): TensorBoard writer for logging metrics during validation.
        epoch (int): The current epoch number. Used for logging.

    Returns:
        None
    """
    model.eval()  #model is in evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate validation accuracy
    val_accuracy = 100 * correct / total

    # Log the metrics to TensorBoard
    writer.add_scalar("Validation Loss", running_loss / len(val_loader), epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

    print(f"Epoch [{epoch+1}], Validation Loss: {running_loss/len(val_loader):.4f}, "
        f"Validation Accuracy: {val_accuracy:.2f}%")
