import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, num_classes):
        input_size = 3072  # 32x32x3 for CIFAR-100
        super().__init__()
        
        # Adding more layers and increasing neuron counts
        self.fc1 = nn.Linear(input_size, 1024)  # Increased neurons
        self.fc2 = nn.Linear(1024, 512)         # New hidden layer with more neurons
        self.fc3 = nn.Linear(512, 256)          # Existing hidden layer
        self.fc4 = nn.Linear(256, num_classes)  # Output layer

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dropout(F.relu(self.fc1(x)))  # First hidden layer with ReLU
        x = self.dropout(F.relu(self.fc2(x)))  # Second hidden layer with ReLU
        x = F.relu(self.fc3(x))  # Third hidden layer with ReLU
        x = self.fc4(x)  # Output layer
        return x
