import torch
from torch import nn
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, num_classes, extractor_trainable=True):
        super(ResNetModel, self).__init__()
        resnet = models.resnet34(pretrained=True)
        
        if not extractor_trainable:
            for param in resnet.parameters():
                param.requires_grad = False
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x