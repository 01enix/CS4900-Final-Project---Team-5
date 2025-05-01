import argparse
import torch
import torchvision
import torchvision.transform as T
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from model import Net
from linearModel import LinearNet

def get_test_loader(batch_size, class_type):
    
     """
    Loads and returns the CIFAR-100 test set DataLoader, class names, and dataset object
    for either fine (100 classes) or coarse (20 superclasses) classification.

    Args:
        batch_size (int): Number of samples per batch.
        class_type (str): '100' for fine labels or '20' for coarse labels.

    Returns:
        tuple: (DataLoader, class_names list, dataset object)
    """
    
    transform = T.Compose([
    T.ToTensor()
    ])

    if class_type == '100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        class_names = dataset.classes
    elif class_type == '20':
        class CIFAR_class_switch(torchvision.datasets.CIFAR100):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.targets = self.coarse_labels
        dataset = CIFAR_class_switch(root='./data', train=False, download=True, transform=transform)
        class_names = [
            'aquatic mammals', 'fish', 'flowers', 'food containers',
            'fruit and vegetables', 'household electrical device', 'household furniture', 'insects',
            'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes',
            'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
            'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
        ]
    else:
        raise ValueError("Invalid class_type")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False), class_names, dataset

def test(model_cls, model_path, batch_size=64, class_type='100'):

    """
    Evaluates a trained model on the CIFAR-100 test set and prints classification metrics.
    Handles both 100-class fine and 20-class coarse evaluations based on class_type.

    Args:
        model_cls (nn.Module): Model class (e.g., Net or LinearNet).
        model_path (str): Path to the trained model's state_dict.
        batch_size (int): Number of samples per batch.
        class_type (str): '100' for fine labels or '20' for coarse labels.

    Returns:
        None
    """ 
    
    test_loader, class_names, dataset = get_test_loader(batch_size, class_type)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = 100 * sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    print(f"Overall Accuracy: {accuracy:.2f}%")

    if class_type == '100':
        precision = precision_score(y_true, y_pred, average=None, labels=range(100))
        recall = recall_score(y_true, y_pred, average=None, labels=range(100))
        f1 = f1_score(y_true, y_pred, average=None, labels=range(100))
        print("\nPer-class Precision, Recall, F1:")
        for i in range(100):
            print(f"Class {i:02d} | P: {precision[i]:.2f}, R: {recall[i]:.2f}, F1: {f1[i]:.2f}")

        print("\nMacro Averages (100 classes):")
        print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
        print(f"F1 Score:  {f1_score(y_true, y_pred, average='macro'):.4f}")

    # Coarse label evaluation
    if class_type == '20':
        y_true_coarse = y_true
        y_pred_coarse = y_pred
    elif class_type == '100':
        y_true_coarse = [dataset.coarse_labels[i] for i in range(len(y_true))]
        y_pred_coarse = [dataset.coarse_labels[i] for i in y_pred]
    else:
        raise ValueError("Invalid class_type for evaluation")

    print("\nSuperclass Precision, Recall, F1:")
    precision_c = precision_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    recall_c = recall_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    f1_c = f1_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    class_names_coarse = [
        'aquatic mammals', 'fish', 'flowers', 'food containers',
        'fruit and vegetables', 'household electrical device', 'household furniture', 'insects',
        'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes',
        'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
        'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
    ]
    for cid in range(20):
        cname = class_names_coarse[cid]
        print(f"{cname:25} | P: {precision_c[cid]:.2f}, R: {recall_c[cid]:.2f}, F1: {f1_c[cid]:.2f}")

    print("\nSuperclass Macro Averages:")
    print(f"Precision: {precision_score(y_true_coarse, y_pred_coarse, average='macro'):.4f}")
    print(f"Recall:    {recall_score(y_true_coarse, y_pred_coarse, average='macro'):.4f}")
    print(f"F1 Score:  {f1_score(y_true_coarse, y_pred_coarse, average='macro'):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ground_truth', type=str, choices=['fine', 'coarse'], default='fine')
    args = parser.parse_args()

    model_class = LinearNet if 'linear' in args.model_path.lower() else Net
    class_type = '100' if args.ground_truth == 'fine' else '20'

    test(model_class, args.model_path, args.batch_size, class_type=class_type)
