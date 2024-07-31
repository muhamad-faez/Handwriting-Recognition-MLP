# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import torch
import torch.nn as nn
import torch.nn.functional as F

class HandWritingMLPModelV18(nn.Module):
    def __init__(self):
        super(HandWritingMLPModelV18, self).__init__()
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(128*128, 2048)
        # Removed batch normalization layers as SELU provides self-normalization
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 62)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.flatten(x)
        x = F.selu(self.fc0(x))
        x = self.dropout(x)
        x = F.selu(self.fc1(x))
        x = self.dropout(x)
        x = F.selu(self.fc2(x))
        x = self.dropout(x)
        x = F.selu(self.fc3(x))
        x = self.dropout(x)
        x = F.selu(self.fc4(x))
        x = self.dropout(x)
        x = F.selu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)  
        return x
