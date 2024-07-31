# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import torch
import torch.nn as nn
import torch.nn.functional as F

class HandWritingMLPModelV11(nn.Module):
    def __init__(self):
        super(HandWritingMLPModelV11, self).__init__()
        self.flatten = nn.Flatten()
        # Adjusting layers for smoother transition
        self.fc1 = nn.Linear(128*128, 1024) # Input layer
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 768) # Smoothing layer
        self.bn2 = nn.BatchNorm1d(768)
        self.fc3 = nn.Linear(768, 512) # Intermediate layer
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256) # Continuing the pattern
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128) # Further smooth transition
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 62) # Output layer
        self.dropout = nn.Dropout(0.25) # Keeping dropout for regularization

    def forward(self, x):
        x = self.flatten(x)
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn5(self.fc5(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc6(x) # No activation
        return x
