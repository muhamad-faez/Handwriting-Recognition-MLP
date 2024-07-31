# DatasetProcessor.py
# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import os
import csv
from torch.utils.data import DataLoader
from CustomImageDataset import CustomImageDataset
from torchvision import transforms
from collections import defaultdict

class DatasetProcessor:
    def __init__(self, csv_file_path, img_folder_path, output_size=(64, 64)):
        self.csv_file_path = csv_file_path
        self.img_folder_path = img_folder_path
        self.output_size = output_size
        self.transform = transforms.Compose([
            transforms.Resize(output_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Create a mapping for labels
        self.label_mapping = self._create_label_mapping()
        print(f"Initialized DatasetProcessor with images resized to {output_size}.")

    def _create_label_mapping(self):
        """Creates a mapping from character labels to integers."""
        # Mapping for digits
        label_mapping = {str(i): i for i in range(10)}
        # Mapping for uppercase letters
        label_mapping.update({chr(i): i - 55 for i in range(65, 91)})
        # Mapping for lowercase letters
        label_mapping.update({chr(i): i - 61 for i in range(97, 123)})
        return label_mapping

    def load_and_split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        grouped_img_paths = self._read_csv_and_group_by_label()
        
        train_paths, val_paths, test_paths = [], [], []
        for label, paths in grouped_img_paths.items():
            split1 = int(len(paths) * train_ratio)
            split2 = split1 + int(len(paths) * val_ratio)
            paths = sorted(paths)
            train_paths.extend([(path, self.label_mapping[label]) for path in paths[:split1]])
            val_paths.extend([(path, self.label_mapping[label]) for path in paths[split1:split2]])
            test_paths.extend([(path, self.label_mapping[label]) for path in paths[split2:]])
        
        train_dataset = CustomImageDataset(train_paths, self.img_folder_path, self.transform)
        val_dataset = CustomImageDataset(val_paths, self.img_folder_path, self.transform)
        test_dataset = CustomImageDataset(test_paths, self.img_folder_path, self.transform)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        print("Created DataLoader instances for train, validation, and test sets.")
        return train_loader, val_loader, test_loader

    def _read_csv_and_group_by_label(self):
        grouped_img_paths = defaultdict(list)
        with open(self.csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for img_path, label in reader:
                grouped_img_paths[label].append(img_path)
        return grouped_img_paths
