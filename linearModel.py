import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, num_classes):
        input_size = 3072  # 32x32x3 for CIFAR-100
        super().__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 1024) 
        self.fc2 = nn.Linear(1024, 768)        
        self.fc3 = nn.Linear(768, 512)         
        self.fc4 = nn.Linear(512, 384)
        self.fc5 = nn.Linear(384, 256)
        self.fc6 = nn.Linear(256, num_classes)  # Output layer
        
        # Dropout layer (applied after fc2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after second layer
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  # Output layer
        return x
