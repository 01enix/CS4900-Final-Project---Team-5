import argparse
import os
import pickle
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from model import Net
from linearModel import LinearNet
from torchvision.datasets import CIFAR100  

# Global list of coarse label names
COARSE_LABEL_NAMES = [
    'aquatic mammals', 'fish', 'flowers', 'food containers',
    'fruit and vegetables', 'household electrical device', 'household furniture', 'insects',
    'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes',
    'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
    'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
]

class CIFAR100Manual(CIFAR100):
    """
    CIFAR-100 dataset wrapper that allows switching between fine (100-class)
    and coarse (20 super-class) labels using the `class_type` argument.
    """
    def __init__(self, class_type='100', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_type = class_type

        # Load coarse labels manually
        data_file = self.train_list[0][0] if self.train else self.test_list[0][0]
        path = os.path.join(self.root, self.base_folder, data_file)
        with open(path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.coarse_labels = entry['coarse_labels']

        if class_type == '20':
            self.targets = self.coarse_labels


def get_test_loader(batch_size, class_type):
    """
    Returns the CIFAR-100 test DataLoader, class names, and dataset object.

    Args:
        batch_size (int): Batch size for DataLoader.
        class_type (str): '100' for fine labels or '20' for coarse labels.

    Returns:
        tuple: (DataLoader, class_names list, dataset object)
    """
    transform = T.Compose([T.ToTensor()])
    dataset = CIFAR100Manual(root='./data', train=False, download=True, transform=transform, class_type=class_type)
    class_names = dataset.classes if class_type == '100' else COARSE_LABEL_NAMES
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), class_names, dataset


def test(model, model_path, batch_size=64, class_type='100'):
    """
    Evaluates a trained model on the CIFAR-100 test set using fine or coarse labels.

    Args:
        model (nn.Module): Model( Net or LinearNet).
        model_path (str): Path to the trained model's .pth file.
        batch_size (int): Batch size for test data.
        class_type (str): '100' for fine labels or '20' for coarse labels.

    Returns:
        None
    """
    test_loader, class_names, dataset = get_test_loader(batch_size, class_type)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    # Overall accuracy
    accuracy = 100 * sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    # Per-class metrics if trained on fine labels
    if class_type == '100':
        correct_per_class = torch.zeros(num_classes)
        total_per_class = torch.zeros(num_classes)
        for i in range(len(y_true_tensor)):
            label = y_true_tensor[i]
            pred = y_pred_tensor[i]
            total_per_class[label] += 1
            if label == pred:
                correct_per_class[label] += 1

        acc_per_class = (correct_per_class / total_per_class.clamp(min=1)) * 100
        print("\nPer-class Accuracy:")
        for i in range(num_classes):
            print(f"Class {i:02d}: {acc_per_class[i]:.2f}%")

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"\nMacro Precision: {precision:.4f}")
        print(f"Macro Recall:    {recall:.4f}")
        print(f"Macro F1 Score:  {f1:.4f}")

    # Always compute superclass metrics
    if class_type == '20':
        y_true_coarse = y_true
        y_pred_coarse = y_pred
    else:
        y_true_coarse = dataset.coarse_labels
        y_pred_coarse = [dataset.coarse_labels[i] for i in y_pred]

    print("\nSuperclass Precision, Recall, F1:")
    precision_c = precision_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    recall_c = recall_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    f1_c = f1_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    for cid in range(20):
        cname = COARSE_LABEL_NAMES[cid]
        print(f"{cname:25} | P: {precision_c[cid]:.2f}, R: {recall_c[cid]:.2f}, F1: {f1_c[cid]:.2f}")
    

    print("\nSuperclass Macro Averages:")
    print(f"Precision: {precision_score(y_true_coarse, y_pred_coarse, average='macro'):.4f}")
    print(f"Recall:    {recall_score(y_true_coarse, y_pred_coarse, average='macro'):.4f}")
    print(f"F1 Score:  {f1_score(y_true_coarse, y_pred_coarse, average='macro'):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved .pth model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for test DataLoader')
    parser.add_argument('--ground_truth', type=str, choices=['fine', 'coarse'], default='fine', help='Type of label to evaluate on')
    parser.add_argument('--model', type=str, default='Net', help='Choose model: Net or LinearNet(default: Net)')
    args = parser.parse_args()
    
    # turn string into model class
    if args.model.lower() == 'net':
        selected_model = Net
    elif args.model.lower() == 'linearnet':
        selected_model = LinearNet
    else:
        raise ValueError("Model must be 'Net' or 'LinearNet'")
    
    class_type = '100' if args.ground_truth == 'fine' else '20'

    test(selected_model, args.model_path, args.batch_size, class_type)
