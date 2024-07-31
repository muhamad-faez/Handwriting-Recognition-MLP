# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from HandWritingMLPModelV13 import HandWritingMLPModelV13

class ExtendedLocalisation:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = HandWritingMLPModelV13().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set model to evaluation mode
        # No need to resize in transform since we're handling patches directly
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.label_mapping = self._create_label_mapping()

    def _create_label_mapping(self):
        # Inverting the label mapping from DatasetProcessor.py
        inverted_mapping = {}
        for i in range(10):  # Digits
            inverted_mapping[i] = str(i)
        for i in range(65, 91):  # Uppercase letters
            inverted_mapping[i - 55] = chr(i)
        for i in range(97, 123):  # Lowercase letters
            inverted_mapping[i - 61] = chr(i)
        return inverted_mapping

    def preprocess_image(self, image_path):
        # Load and convert the image to grayscale and tensor
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
        return image_tensor

    def extract_patches(self, image, patch_size=128, stride=128):
        patches = []
        _, _, H, W = image.shape
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                if y + patch_size <= H and x + patch_size <= W: 
                    patch = image[:, :, y:y+patch_size, x:x+patch_size]
                    patches.append((x, y, patch))
        return patches

    def predict(self, image_tensor):
        patches = self.extract_patches(image_tensor)
        predictions = []

        for x, y, patch in patches:
            output = self.model(patch)
            pred = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(pred, dim=1)
            if confidence.item() > 0.01:  # Adjust threshold as needed
                character = self.label_mapping[predicted_class.item()]
                predictions.append((x, y, character, confidence.item()))

        return predictions

    def localize_characters(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        predictions = self.predict(image_tensor)
        return predictions


