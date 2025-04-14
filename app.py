import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import argparse
from model import Net              
from test import new_dicts           
import os


#CLI args for modelpath selection
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth file)')
args = parser.parse_args()
if not os.path.exists(args.model_path):
    print(f"[ERROR] Model file not found at: {args.model_path}")
    sys.exit(1)

model_path = args.model_path
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

#load model
model = Net(num_classes=100).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#labels
label_maps = new_dicts()
id_to_fine = label_maps['id_fine']
fine_to_coarse = label_maps['fine_coarse']

transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_image(path):
    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class_id = torch.argmax(probs, dim=1).item()

    pred_class_name = id_to_fine[pred_class_id]
    super_class_name = fine_to_coarse[pred_class_name]

    result_label.config(
        text=f"Prediction: {pred_class_name.replace('_', ' ').title()}\nSuper-Class: {super_class_name.title()}"
    )

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path).resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        predict_image(file_path)

root = tk.Tk()
root.title("Image Classifier")

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root)
result_label.pack()

root.mainloop()
