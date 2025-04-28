# --CIFAR-100 Image Classification and Prediction App--
This project provides a full solution for classifying images using a Convolutional Neural Network (CNN) or a Linear Neural Network on the CIFAR-100 dataset.
It includes both:

A command-line interface operation for training, testing, and evaluation of the selected model,

And a Tkinter-based GUI for interactive image classification.

# --Requirements--
### *Python 3.7+ https://www.python.org/downloads/

Install packages:

![image](https://github.com/user-attachments/assets/6d8dd32a-bb51-474e-ac5e-5e8f0395b40e)

### NOTE: CIFAR-100 dataset will automatically be downloaded into the working directory if not already installed upon executing train.py 

### *CIFAR-100 dataset:(https://www.cs.toronto.edu/~kriz/cifar.html)
  -Download and place the extracted files (meta, train, test) inside a ./data folder within the project file

(Optional (for monitoring training):

### *TensorBoard

# --Execution Instructions--
## 1. Prepare CIFAR-100 dataset
   
   -Make sure the CIFAR-100 dataset files (meta, train, test) are inside (project path)/data/cifar-100-python/

   ### NOTE: If the CIFAR-100 dataset is not found it will be downloaded upon execution of train.py

## 2. Generate 32 x 32. png images for GUI implementation using the following code

  https://gist.github.com/ahanagemini/aad2fc08913fe20a0ba9b137b3a1944b

   ### NOTE: Using any other images may result in improper GUI output.
   
## 3. Open Command Line Interface/Bash
   
   ### NOTE: This program was designed to run on Windows but will work with other OS but LI commands may vary.

   -change the working directory to the path of the project file
   
   ![image](https://github.com/user-attachments/assets/718ef8d5-c952-4020-83dc-432f82c60b0e)

   
## 4. Training a Model
   
  -Train a CNN (Net) or LinearNet on fine or coarse labels
  
   
   ![image](https://github.com/user-attachments/assets/26f8a866-3dfc-4db6-9fc9-5e99631bdb0d)

  Arguments:
  
    --model: Net or LinearNet
    
    --epochs: Number of training epochs (default 20)
    
    --lr: Learning rate (default 0.001)
    
    --class_type: 100 (fine labels) or 20 (coarse labels)
    
  ### NOTE: The trained model is saved automatically within the project folder in the format:
  
    <model_name>_<dataset>_<timestamp>.pth
    
  ### NOTE: Training logs are saved inside the "runs/" folder as within the project folder:
  
    view the logs using the CLI TensorBoard command (optional)
    
  ![image](https://github.com/user-attachments/assets/08ac29f8-86a6-471e-bbbe-1e3b45616259)

## 5. Testing and Evaluating a Model
   
  -Evaluate a saved model on CIFAR-100 test set:
  
  ![image](https://github.com/user-attachments/assets/e161e665-a2ae-4b9e-bf80-f85db218855c)


   Arguments:
   
    --model: Net or LinearNet
    
    --model_path: Path to the saved .pth model
    
    --batch_size: Testing batch size (default 64)
    
    --class_type: 100 (fine labels) or 20 (coarse labels)
   
  Outputs:
  
    -Overall test accuracy
    
    -Per-class precision, recall, F1-scores
    
    -Mean accuracy over all classes
    
    -Superclass metrics if using class_type 20
    
## 6. GUI-Based Single Image Prediction
   
  -Launch the GUI and predict classes for a single image:
  
  ![image](https://github.com/user-attachments/assets/d3f9c8d3-1f32-40db-858c-507e553576fa)

  -Load an image (.jpg, .jpeg, .png) by clicking "Open Image".
  
  ![image](https://github.com/user-attachments/assets/d16d9122-d691-40ec-9627-4b7919c792a5)

  The GUI displays:
  
  Image
  Predicted coarse (superclass) label
  Predicted fine (100-class) label
  
  ![image](https://github.com/user-attachments/assets/8f3fd7e9-eeb2-4857-b83d-3b70d14cdae6)

# --Directory Structure--

![image](https://github.com/user-attachments/assets/45102f00-a5de-4244-b7fa-2c6ea19b3c36)

# --Important Notes--
GPU (CUDA) will be used automatically if available.

Make sure to match the architecture used for training and testing.

Models trained with class_type=20 will predict superclasses; otherwise, they predict fine classes.

The GUI expects input images resized to 32x32 internally to match CIFAR-100 dimensions.



