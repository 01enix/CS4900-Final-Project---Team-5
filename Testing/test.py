import argparse
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
    # Define the transformation pipeline for the test dataset
    transform_pipeline = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-100 test dataset
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_pipeline)

    # Initialize DataLoader
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    return test_loader

def test(model, model_path, batch_size=64):
    """
    In this function we use the test data on the CNN model on the CIFAR-100 dataset.

    Args:
        model: the CNN model to be tested
        model_path: path to the trained model
        batch_size: batch size for testing (default 64)
    """

    test_loader = get_test_loader(batch=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model(num_classes=100).to(device)

    # network and trained model loading
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in range(100)}  
    total_pred = {classname: 0 for classname in range(100)}
    
    # evaluate the model without gradients
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # update the accuracy per class
            for label, prediction in zip(labels, predicted):
                correct_pred[label.item()] += (label == prediction).item()
                total_pred[label.item()] += 1

    # overall accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

    # accuracy per class
    for classname in range(100):
        class_accuracy = 100 * correct_pred[classname] / total_pred[classname]
        print(f'Accuracy for class {classname}: {class_accuracy:.1f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Net', help='Choose model: Net or LinearNet (default: Net)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing (default: 64)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    args = parser.parse_args()

    # get the model based on the argument
    model_name = args.model.lower()
    if model_name not in models:
        raise ValueError(f"Model '{args.model}' not found. Available models: {list(models.keys())}")
    
    selected_model = models[model_name]

    # test the model
    test(model=selected_model, model_path=args.model_path, batch_size=args.batch_size)