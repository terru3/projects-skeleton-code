# NOTE: These model parameters are not finalized
import torch
import torch.nn as nn

class StartingNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=5)
        self.pool4 = nn.MaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1176, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.pool1(self.relu(self.conv1(x))))
        x = self.bn2(self.pool2(self.relu(self.conv2(x))))
        x = self.bn3(self.pool3(self.relu(self.conv3(x))))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x


''' Tests for the output dimension after convolutional layers to pass into FC layers

 test = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size = 5),
        nn.MaxPool2d(2, 2), 
        nn.Conv2d(6, 12, kernel_size = 5), 
        nn.MaxPool2d(2, 2), 
        nn.Conv2d(12, 24, kernel_size = 5),
        nn.MaxPool2d(2, 2), 
        nn.Conv2d(24, 24, kernel_size = 5),
        nn.MaxPool2d(1),
        nn.Flatten()
    )

    input = torch.randn(16, 3, 120, 120)
    # note image is now 120x120
    output = test(input)
    print(output.size())
'''