import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model_registry import models

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
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-100 test dataset
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_pipeline)

    # Initialize DataLoader
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    # network and trained model loading
    num_classes = 100  # CIFAR-100 has 100 classes
    net = Net(num_classes=num_classes)
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in range(num_classes)}  # Assuming CIFAR-100 classes are 0 to 99
    total_pred = {classname: 0 for classname in range(num_classes)}
    
    # evaluate the model without gradients
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update per-class metrics
            for label, prediction in zip(labels, predicted):
                correct_pred[label.item()] += (label == prediction).item()
                total_pred[label.item()] += 1

    # overall accuracy
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    # accuracy for each class
    for classname in range(num_classes):
        accuracy = 100 * correct_pred[classname] / total_pred[classname]
        print(f'Accuracy for class {classname}: {accuracy:.1f} %')

    

if __name__ == '__main__':
    # specify the path to the model saved after training
    model_path = './cnn_cifar100_20250322-163000.pth'  # Change to your actual path

    # run the test function with this model path
    test(batch=64, model_path=model_path)

    return test_loader