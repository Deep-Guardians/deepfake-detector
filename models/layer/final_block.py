import torch
import torch.nn as nn

class FinalBlock(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.2):
        super(FinalBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
