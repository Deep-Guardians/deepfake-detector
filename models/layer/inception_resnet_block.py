import torch
import torch.nn as nn
from models.layer.basic_layers import Conv2dBnRelu
from models.layer.scaling_layer import ScalingLayer

class InceptionResNetA(nn.Module):
    def __init__(self, scale=True):
        super(InceptionResNetA, self).__init__()
        self.scale = scale
        self.branch1 = Conv2dBnRelu(192, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            Conv2dBnRelu(192, 32, kernel_size=1),
            Conv2dBnRelu(32, 32, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            Conv2dBnRelu(192, 32, kernel_size=1),
            Conv2dBnRelu(32, 48, kernel_size=3, padding=1),
            Conv2dBnRelu(48, 64, kernel_size=3, padding=1)
        )
        self.conv = nn.Conv2d(128, 192, kernel_size=1)
        if self.scale:
            self.scaling = ScalingLayer(scale=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        concat = torch.cat([branch1, branch2, branch3], dim=1)
        conv = self.conv(concat)
        if self.scale:
            conv = self.scaling(conv)
        return self.relu(x + conv)

class InceptionResNetB(nn.Module):
    def __init__(self, scale=True):
        super(InceptionResNetB, self).__init__()
        self.scale = scale
        self.branch1 = Conv2dBnRelu(192, 128, kernel_size=1)
        self.branch2 = nn.Sequential(
            Conv2dBnRelu(192, 128, kernel_size=1),
            Conv2dBnRelu(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            Conv2dBnRelu(160, 192, kernel_size=(7, 1), padding=(3, 0))
        )
        self.conv = nn.Conv2d(320, 192, kernel_size=1)
        if self.scale:
            self.scaling = ScalingLayer(scale=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        concat = torch.cat([branch1, branch2], dim=1)
        conv = self.conv(concat)
        if self.scale:
            conv = self.scaling(conv)
        return self.relu(x + conv)

class InceptionResNetC(nn.Module):
    def __init__(self, scale=True):
        super(InceptionResNetC, self).__init__()
        self.scale = scale
        self.branch1 = Conv2dBnRelu(192, 192, kernel_size=1)
        self.branch2 = nn.Sequential(
            Conv2dBnRelu(192, 192, kernel_size=1),
            Conv2dBnRelu(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            Conv2dBnRelu(224, 256, kernel_size=(3, 1), padding=(1, 0))
        )
        self.conv = nn.Conv2d(448, 192, kernel_size=1)
        if self.scale:
            self.scaling = ScalingLayer(scale=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        concat = torch.cat([branch1, branch2], dim=1)
        conv = self.conv(concat)
        if self.scale:
            conv = self.scaling(conv)
        return self.relu(x + conv)
