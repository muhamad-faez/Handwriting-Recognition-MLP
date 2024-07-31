# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import torch
import torch.nn as nn
import torch.nn.functional as F

class HandWritingMLPModelV2(nn.Module):
    def __init__(self):
        super(HandWritingMLPModelV2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*64, 1024)
        self.bn1 = nn.BatchNorm1d(1024)  # Batch normalization for the first layer
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)   # Batch normalization for the second layer
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)   # Batch normalization for the third layer
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)   # Batch normalization for the fourth layer
        self.fc5 = nn.Linear(128, 62)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
