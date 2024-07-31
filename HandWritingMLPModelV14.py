# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import torch
import torch.nn as nn
import torch.nn.functional as F

class HandWritingMLPModelV14(nn.Module):
    def __init__(self):
        super(HandWritingMLPModelV14, self).__init__()
        self.flatten = nn.Flatten()
        # Adjusting layers to smooth out the transition from 2048 to 1024
        self.fc0 = nn.Linear(128*128, 2048)
        self.bn0 = nn.BatchNorm1d(2048)
        # Intermediate layer to smooth the transition
        self.fc0_5 = nn.Linear(2048, 1536)  # New intermediate layer
        self.bn0_5 = nn.BatchNorm1d(1536)  # BatchNorm for the new intermediate layer
        self.fc1 = nn.Linear(1536, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.fc3 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 62)
        self.dropout = nn.Dropout(0.25)  # Keeping dropout for regularization

    def forward(self, x):
        x = self.flatten(x)
        x = F.leaky_relu(self.bn0(self.fc0(x)), negative_slope=0.01)
        x = self.dropout(x)
        # Applying ReLU and dropout to the new intermediate layer
        x = F.leaky_relu(self.bn0_5(self.fc0_5(x)), negative_slope=0.01)
        x = self.dropout(x)
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
        x = self.fc6(x)  
        return x
