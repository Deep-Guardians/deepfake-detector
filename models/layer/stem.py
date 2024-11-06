import torch.nn as nn
from models.layer.basic_layers import Conv2dBnRelu

class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            Conv2dBnRelu(3, 32, kernel_size=3, stride=2, padding=0),  # 299x299 -> 149x149
            Conv2dBnRelu(32, 32, kernel_size=3, stride=1, padding=0), # 149x149 -> 147x147
            Conv2dBnRelu(32, 64, kernel_size=3, stride=1, padding=1), # 147x147 -> 147x147
            nn.MaxPool2d(3, stride=2),                                # 147x147 -> 73x73
            Conv2dBnRelu(64, 80, kernel_size=1, stride=1, padding=0),
            Conv2dBnRelu(80, 192, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(3, stride=2)                                 # 73x73 -> 35x35
        )

    def forward(self, x):
        return self.stem(x)
