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
from sklearn.metrics import precision_score, recall_score, f1_score
import pprint as pp
from model import Net
from linearModel import LinearNet


#Coarse and Fine label mapping
mapping_C_F = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper', 'banana', 'peach', 'potato'],
    'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

fine_labels = [label for labels in mapping_C_F.values() for label in labels]

def new_dicts():
    """
    Constructs and returns mappings between fine and coarse labels.

    This function builds several dictionaries that map:
    - fine label names to IDs and vice versa,
    - coarse label names to IDs and vice versa,
    - fine labels to their corresponding coarse labels,
    - fine label IDs to coarse label IDs,
    - coarse label IDs to a list of fine label IDs they contain.

    Returns:
        dict: A dictionary containing the following mappings:
            - 'fine_id' (dict): Maps fine label names to integer IDs.
            - 'id_fine' (dict): Maps fine label IDs to their label names.
            - 'coarse_id' (dict): Maps coarse label names to integer IDs.
            - 'id_coarse' (dict): Maps coarse label IDs to their label names.
            - 'fine_coarse' (dict): Maps each fine label name to its coarse label name.
            - 'fine_id_coarse_id' (dict): Maps fine label IDs to their corresponding coarse label IDs.
            - 'coarse_id_fine_id' (dict): Maps coarse label IDs to a list of fine label IDs.
    """
    # Build the necessary mappings between fine and coarse labels
    fine_id = dict()
    id_fine = dict()
    for id, label in enumerate(fine_labels):
        fine_id[label] = id
        id_fine[id] = label

    

    coarse_id = dict()
    id_coarse = dict()
    fine_coarse = dict()
    fine_id_coarse_id = dict()
    coarse_id_fine_id = dict()

    for id, (coarse, fines) in enumerate(mapping_C_F.items()):
        coarse_id[coarse] = id
        id_coarse[id] = coarse
        fine_labels_ids = []
        for fine in fines:
            fine_coarse[fine] = coarse
            fine_label_id = fine_id[fine]
            fine_id_coarse_id[fine_label_id] = id
            fine_labels_ids.append(fine_label_id)
        coarse_id_fine_id[id] = fine_labels_ids

    # dicts = ['fine_id', 'id_fine', 'coarse_id', 'id_coarse', 'fine_coarse', 'fine_id_coarse_id', 'coarse_id_fine_id']
    # for dic in dicts:
    #     dic_value = locals()[dic]
    #     print(dic + ' = ')
    #     pp.pprint(dic_value)

    return {
        'fine_id': fine_id,
        'id_fine': id_fine,
        'coarse_id': coarse_id,
        'id_coarse': id_coarse,
        'fine_coarse': fine_coarse,
        'fine_id_coarse_id': fine_id_coarse_id,
        'coarse_id_fine_id': coarse_id_fine_id
    }

def get_test_loader(batch=64, class_type='100'):
    """
    This function loads and preprocesses the CIFAR-100 test dataset.

    Args:
        batch (int): Batch size for testing (default 64).
        class_type (str): Class type - '100' for fine labels, '20' for coarse labels (default '100').

    Returns:
        A tuple (DataLoader, class_names) for test set evaluation.
    """

    # Define the preprocessing pipeline matching your training
    transform_pipeline = tv_transforms.Compose([tv_transforms.ToTensor(), tv_transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])])

    # Load CIFAR-100 dataset
    cifar_data = torchvision.datasets.CIFAR100(root='./data', train=False)

    # Set class names depending on the class type
    if class_type == '100':
        # Use CIFAR-100 fine labels
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_pipeline)
        class_names = test_set.classes  # Fine class names
    elif class_type == '20':
        # Use coarse labels
        test_set = CIFAR_class_switch(
            root='./data', train=False, download=True, transform=transform_pipeline, class_type='20'
        )  # Assuming CIFAR_class_switch is used for coarse labels
        class_names = test_set.coarse_label_names  # Coarse class names
    else:
        raise ValueError(f"Unsupported class_type: {class_type}. Choose '100' or '20'.")

    # DataLoader for test set
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)
    
    return test_loader, class_names  # Now returning both test_loader and class_names


def test(model, model_path, batch_size=64, class_type='100'):
    """
    Test the CNN model on the CIFAR-100 dataset.

    Args:
        model: the CNN model to be tested.
        model_path: path to the trained model.
        batch_size: batch size for testing (default 64).
        class_type: '100' for fine labels or '20' for coarse labels (default '100').
    """
    test_loader, class_names = get_test_loader(batch=batch_size, class_type=class_type)
    num_classes = len(class_names) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model(num_classes=num_classes).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in range(num_classes)}
    total_pred = {classname: 0 for classname in range(num_classes)}

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
    for classname in range(num_classes):
        class_accuracy = 100 * correct_pred[classname] / total_pred[classname]
        print(f'Accuracy for class {classname}: {class_accuracy:.1f}%')

    #Compute mean accuracy
    class_accuracies = [100 * correct_pred[c] / total_pred[c] for c in range(num_classes) if total_pred[c] > 0]
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

    #metric calculation
    precision = precision_score(y_true, y_pred, average=None, labels=range(100))
    recall = recall_score(y_true, y_pred, average=None, labels=range(100))
    f1 = f1_score(y_true, y_pred, average=None, labels=range(100))

    print("\nPer-class Precision, Recall, and F1-score:")
    for classname in range(100):
        print(f'Class {classname:2d} - Precision: {precision[classname]:.2f}, Recall: {recall[classname]:.2f}, F1-score: {f1[classname]:.2f}')

    #macro metric calculation
    m_precision = precision_score(y_true, y_pred, average = 'macro')
    m_recall = recall_score(y_true, y_pred, average = 'macro')
    m_f1 = f1_score(y_true, y_pred, average = 'macro')

    print(f"Macro Precision: {m_precision:.4f}")
    print(f"Macro Recall: {m_recall:.4f}")
    print(f"Macro F1-score: {m_f1:.4f}")

     #Superclass metric map tings
    label_maps = new_dicts()
    fine_id_coarse_id = label_maps['fine_id_coarse_id']
    id_coarse = label_maps['id_coarse']

    #Map fine predictions to superclasses
    y_true_coarse = [fine_id_coarse_id[int(y)] for y in y_true]
    y_pred_coarse = [fine_id_coarse_id[int(p)] for p in y_pred]


    precision_c = precision_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    recall_c = recall_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))
    f1_c = f1_score(y_true_coarse, y_pred_coarse, average=None, labels=range(20))

    print("\nSuperclass Precision, Recall, F1:")
    for cid in range(20):
        cname = id_coarse[cid]
        print(f"{cname:25} | Precision: {precision_c[cid]:.2f} | Recall: {recall_c[cid]:.2f} | F1: {f1_c[cid]:.2f}")

    # Macro over superclasses
    macro_p = precision_score(y_true_coarse, y_pred_coarse, average='macro')
    macro_r = recall_score(y_true_coarse, y_pred_coarse, average='macro')
    macro_f1 = f1_score(y_true_coarse, y_pred_coarse, average='macro')

    print("\nSuperclass Macro-Averages:")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F1 Score:  {macro_f1:.4f}")

def model_loader(model_name, model_path, ground_truth):
    num_classes = 100 if ground_truth == 'fine' else 20

    if model_name.lower() == 'net':
        model = Net(num_classes)
    elif model_name.lower() == 'linearnet':
        model = LinearNet(num_classes)
    else:
        raise ValueError("Unsupported model type. Choose 'Net' or 'LinearNet'")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Adjust for GPU or CPU
    model.eval()
    return model

def predict_images(model, model_path, image_directory, class_type='100'):
    """
    Run inference on a set of images in a directory using the specified model.

    Args:
        model: the CNN model to use for prediction.
        model_path: path to the trained model.
        image_directory: directory containing images to predict.
        class_type: '100' for fine labels, '20' for superclasses.
    """

    num_classes = 100 if class_type == '100' else 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model(num_classes=100).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()


    print(f"\nPredicting on images in '{image_directory}' using class type {class_type} ({num_classes} classes)...\n")
        
    for image_name in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform_pipeline(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = net(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            class_label = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
            print(f"Image: {image_name} - Predicted Class: {predicted_class} ({class_label})")
        except Exception as e:
            print(f"[ERROR] Could not process image '{image_name}': {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Net', help='Choose model: Net, LinearNet, etc. (default: Net)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing (default: 64)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_directory', type=str, default=None, help='Directory with images for prediction')
    parser.add_argument('--ground_truth', type=str, default='fine', choices=['fine', 'coarse'], help='Specify label type the model was trained on: fine (100 classes) or coarse (20 superclasses)')
    args = parser.parse_args()

    model = model_loader(args.model, args.model_path, args.ground_truth)

    # If an image directory is provided, run image predictions; otherwise, run CIFAR-100 tests.
    if args.image_directory:
        predict_images(model, args.model_path, args.image_directory, class_type='100' if args.ground_truth == 'fine' else '20')
    else:
        test(Net, args.model_path, batch_size=args.batch_size, class_type='100' if args.ground_truth == 'fine' else '20')