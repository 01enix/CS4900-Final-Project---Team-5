import argparse
import os
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms as tv_transforms  
from model_registry import models
from sklearn.metrics import precision_score, recall_score, f1_score

def get_test_loader(batch=64):
    """
    This function loads and preprocesses the CIFAR-100 test dataset.

    Args:
        batch (int): Batch size for testing (default 64).

    Returns:
        DataLoader for the test dataset.
    """
    transform_pipeline = T.Compose([
        T.ToTensor()
    ])
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_pipeline)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)
    return test_loader

def test(model, model_path, batch_size=64):
    """
    Test the CNN model on the CIFAR-100 dataset.

    Args:
        model: the CNN model to be tested.
        model_path: path to the trained model.
        batch_size: batch size for testing (default 64).
    """
    test_loader = get_test_loader(batch=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model(num_classes=100).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in range(100)}
    total_pred = {classname: 0 for classname in range(100)}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                correct_pred[label.item()] += (label == prediction).item()
                total_pred[label.item()] += 1

    accuracy = 100 * correct / total
    print(f'Accuracy on the 10000 test images: {accuracy:.2f}%')

    #Comput per-class accuracy 
    for classname in range(100):
        class_accuracy = 100 * correct_pred[classname] / total_pred[classname]
        print(f'Accuracy for class {classname}: {class_accuracy:.1f}%')
    #Compute mean accuracy
    class_accuracies = [100 * correct_pred[c] / total_pred[c] for c in range(100)]
    mean_accuracy = sum(class_accuracies) / len(class_accuracies)
    print(f'Mean Accuracy across all classes: {mean_accuracy:.2f}%')

    #Compute Precision, Recall, F1-score 
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    precision = precision_score(y_true, y_pred, average=None, labels=range(100))
    recall = recall_score(y_true, y_pred, average=None, labels=range(100))
    f1 = f1_score(y_true, y_pred, average=None, labels=range(100))

    print("\nPer-class Precision, Recall, and F1-score:")
    for classname in range(100):
        print(f'Class {classname:2d} - Precision: {precision[classname]:.2f}, Recall: {recall[classname]:.2f}, F1-score: {f1[classname]:.2f}')


def predict_images(model, model_path, image_directory):
    """
    Run inference on a set of images in a directory using the specified model.

    Args:
        model: the CNN model to use for prediction.
        model_path: path to the trained model.
        image_directory: directory containing images to predict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model(num_classes=100).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Define the preprocessing pipeline matching your training
    transform_pipeline = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for image_name in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform_pipeline(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = net(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            print(f"Image: {image_name}, Predicted Class: {predicted_class}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Net', help='Choose model: Net, LinearNet, etc. (default: Net)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing (default: 64)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_directory', type=str, default=None, help='Directory with images for prediction')
    args = parser.parse_args()

    model_name = args.model.lower()
    if model_name not in models:
        raise ValueError(f"Model '{args.model}' not found. Available models: {list(models.keys())}")
    selected_model = models[model_name]

    # If an image directory is provided, run image predictions; otherwise, run CIFAR-100 tests.
    if args.image_directory:
        predict_images(selected_model, args.model_path, args.image_directory)
    else:
        test(selected_model, args.model_path, batch_size=args.batch_size)
