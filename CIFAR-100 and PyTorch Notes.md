#CIFAR-100 Dataset Overview
##CIFAR-100 Dataset:
-Consists of 100 different classes (like “apple,” “cat,” “truck”) with each class having 600 images (500 for training and 100 for testing).
-The images are 32x32 pixels in RGB color.
-The dataset is split into a training set and a test set.
##Downloading the Dataset:
-You can load CIFAR-100 directly through PyTorch using the 'torchvision.datasets.CIFAR100' function. This will automatically download the dataset if you don’t have it already stored locally.
-Transformations: Images are typically converted into tensors using 'ToTensor()'. A normalization transformation (Normalize()) is also used to scale the pixel values, which can -help improve the training process.

#PyTorch Overview
##PyTorch Basics:
-PyTorch provides the tools necessary to create deep learning models. It includes modules for creating layers (torch.nn), performing mathematical operations (torch), and optimizing the model (torch.optim).
-You’ll also use 'torch.utils.data.DataLoader' to load your training and test data in batches.
##Key PyTorch Components:
-Tensors: Multi-dimensional arrays (like NumPy arrays) that are used in computations.
-Neural Networks: Models are created by stacking layers (convolutional, fully connected, etc.).
--Optimizer: Used to update the model's weights. Common optimizers include 'Adam' and 'SGD' (Stochastic Gradient Descent).
Loss Function: Measures the difference between the predicted and actual outputs. For classification, the common loss function is 'CrossEntropyLoss()'.
