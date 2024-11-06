import torch
import torch.nn as nn
from models.layer.basic_layers import Conv2dBnRelu

class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2 = Conv2dBnRelu(192, 192, kernel_size=3, stride=2)
        self.branch3 = nn.Sequential(
            Conv2dBnRelu(192, 192, kernel_size=1),
            Conv2dBnRelu(192, 224, kernel_size=3, padding=1),
            Conv2dBnRelu(224, 256, kernel_size=3, stride=2)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat([branch1, branch2, branch3], dim=1)


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2 = Conv2dBnRelu(192, 192, kernel_size=3, stride=2)
        self.branch3 = nn.Sequential(
            Conv2dBnRelu(192, 192, kernel_size=1),
            Conv2dBnRelu(192, 224, kernel_size=3, padding=1),
            Conv2dBnRelu(224, 256, kernel_size=3, stride=2)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat([branch1, branch2, branch3], dim=1)
