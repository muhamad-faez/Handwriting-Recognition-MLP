# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import torch
import torch.nn as nn
import torch.nn.functional as F

class HandWritingMLPModelV1(nn.Module):
    def __init__(self):
        super(HandWritingMLPModelV1, self).__init__()
        # Flatten the 64x64 image into a 4096-dimensional vector
        self.flatten = nn.Flatten()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(64*64, 512)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(512, 256)
        # Define the third fully connected layer that outputs the class scores
        self.fc3 = nn.Linear(256, 62)  # 62 classes

    def forward(self, x):
        # Flatten the input image
        x = self.flatten(x)
        # Apply the first fully connected layer with ReLU activation function
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer with ReLU activation function
        x = F.relu(self.fc2(x))
        # Apply the third fully connected layer to get class scores
        x = self.fc3(x)
        return x
