 import os
 from PIL import Image
 import torch
 from torchvision import transforms
 
 # Load the model definition and weights
 net = Net(num_classes=100)  
 net.load_state_dict(torch.load('model.pth'))  
 net.eval()  
 
 # Define preprocessing pipeline (matching training preprocessing)
 transform_pipeline = transforms.Compose([
     transforms.Resize((224, 224)),  
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
 ])
 
 # Specify the directory containing the images
 image_directory = "path_to_image_directory"  
 
 # Loop through each image in the directory
 for image_name in os.listdir(image_directory):
     image_path = os.path.join(image_directory, image_name)
     try:
         # Open and preprocess the image
         image = Image.open(image_path).convert("RGB")
         input_tensor = transform_pipeline(image).unsqueeze(0)  # Add batch dimension for the model
 
         # Run the image through the model
         with torch.no_grad():  
             output = net(input_tensor)
 
         # Process model output 
         predicted_class = output.argmax(dim=1).item()
         print(f"Image: {image_name}, Predicted Class: {predicted_class}")
     
     except Exception as e:
         print(f"Error processing {image_name}: {e}")
