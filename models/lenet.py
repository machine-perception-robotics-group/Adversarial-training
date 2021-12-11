import torch.nn as nn
import torch.nn.functional as F

class Lenet5(nn.Module):
    def __init__(self, num_classes=10):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.maxpool(self.relu(self.conv1(x)))
        h = self.maxpool(self.relu(self.conv2(h)))
        
        h = torch.flatten(h, start_dim=1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h