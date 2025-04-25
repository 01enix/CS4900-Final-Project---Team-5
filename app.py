import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import argparse
from model import Net
from torchvision.datasets import CIFAR100
import os

# Subclass to expose coarse label names
class CIFAR100WithCoarse(CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coarse_label_names = [
            'aquatic mammals', 'fish', 'flowers', 'food containers',
            'fruit and vegetables', 'household electrical device', 'household furniture', 'insects',
            'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes',
            'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
            'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2'
        ]
#CLI args for modelpath selection
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth file)')
args = parser.parse_args()
if not os.path.exists(args.model_path):
    print(f"[ERROR] Model file not found at: {args.model_path}")
    sys.exit(1)
    
#device selection
model_path = args.model_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model
model = Net(num_classes=100).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Dataset info
cifar = CIFAR100WithCoarse(root='./data', train=False)
fine_labels = cifar.classes
coarse_labels = cifar.coarse_label_names

# Build fine-to-coarse mapping manually using the official CIFAR-100 meta file
with open('./data/cifar-100-python/train', 'rb') as f:
    train_data = pickle.load(f, encoding='bytes')
    fine_labels_list = train_data[b'fine_labels']
    coarse_labels_list = train_data[b'coarse_labels']

#image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Prediction function
def predict_image(path):
    """
    Classifies an image using the loaded CNN model.
    Displays the predicted fine label and corresponding super-class label in the GUI.

    Args:
        path (str): File path to the image to be classified.
    """   
    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class_id = torch.argmax(probs, dim=1).item()

    pred_class_name = fine_labels[pred_class_id]
    super_class_name = fine_to_coarse_dict[pred_class_id]

    result_text = f"{super_class_name} \n {pred_class_name}"
    result_label.config(text=result_text)
    

#image loader
def open_image():
    """
    Opens a file dialog to select an image file.
    Loads and displays the selected image in the GUI.
    Triggers classification of the image.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path).resize((400,300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        predict_image(file_path)
        
# GUI startup
def main():
    """
    Initializes the GUI application window.
    Sets up all interactive components and starts the main event loop.
    """
    global result_label, image_label

    root = tk.Tk()
    root.state('zoomed')
    root.title("Image Classifier")

    open_button = tk.Button(root, text="Open Image", command=open_image)
    open_button.pack()

    image_label = tk.Label(root)
    image_label.pack()

    result_label = tk.Label(root)
    result_label.pack()

    root.mainloop()

if __name__ == '__main__':
    main()
