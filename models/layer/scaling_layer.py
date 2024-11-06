import torch.nn as nn

class ScalingLayer(nn.Module):
    def __init__(self, scale=0.1):
        super(ScalingLayer, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
