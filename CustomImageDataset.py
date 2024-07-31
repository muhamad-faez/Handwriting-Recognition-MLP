# CustomImageDataset.py
# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None):
        """
        Args:
            img_labels (list of tuples): List of (image_path, label) tuples.
            img_dir (string): Directory with all the images, without including 'Img/'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = img_labels  # List of (image_path, label) tuples
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        # Ensure the 'Img/' part is not redundantly included in the full path
        img_path = img_path.replace('Img/', '', 1) if img_path.startswith('Img/') else img_path
        full_img_path = os.path.join(self.img_dir, img_path)  # Correctly construct full image path
        image = Image.open(full_img_path).convert('RGB')  # Load and convert image to RGB

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, label
