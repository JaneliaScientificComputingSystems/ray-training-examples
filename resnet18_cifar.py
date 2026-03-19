#!/usr/bin/env python3
"""
ResNet-18 adapted for CIFAR-10 (32x32 input).
Shared by training and inference scripts.
"""
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
