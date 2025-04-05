import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(self, num_classes):
        input_size = 3072  # 32x32x3 for CIFAR-100
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
