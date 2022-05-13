import torch
import torch.nn as nn
import torchvision.models as models

class StartingNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model_resnet = models.resnet18(pretrained=True)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, 5)

    def forward(self, x):
        with torch.no_grad():
            features = self.model_resnet(x)
        prediction = self.fc(features)
        return prediction