# NOTE: These model parameters are not finalized——they're virtually random atm
import torch
import torch.nn as nn

class StartingNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(12, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(48, 120, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(94080, 14400)
        self.fc2 = nn.Linear(14400, 1024)
        self.fc3 = nn.Linear(1024, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


''' Tests for the output dimension after convolutional layers to pass into FC layers

test = nn.Sequential(
        nn.Conv2d(3, 12, kernel_size=5, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(12, 48, kernel_size=3, padding=1),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 120, kernel_size=3, padding=1),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten()
    )

input = torch.randn(16, 3, 224, 224)
output = test(input)
print(output.size())
'''

model = StartingNetwork()
print(model)