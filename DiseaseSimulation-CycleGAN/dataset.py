from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HealthyDiseasedDataset(Dataset):
    def __init__(self, root_diseased, root_healthy, transform=None):
        self.root_diseased = root_diseased
        self.root_healthy = root_healthy
        self.transform = transform

        self.diseased_images = os.listdir(root_diseased)
        self.healthy_images = os.listdir(root_healthy)
        self.length_dataset = max(len(self.diseased_images), len(self.healthy_images)) # 1000, 1500
        self.diseased_len = len(self.diseased_images)
        self.healthy_len = len(self.healthy_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        diseased_img = self.diseased_images[index % self.diseased_len]
        healthy_img = self.healthy_images[index % self.healthy_len]

        diseased_path = os.path.join(self.root_diseased, diseased_img)
        healthy_path = os.path.join(self.root_healthy, healthy_img)

        diseased_img = np.array(Image.open(diseased_path).convert("RGB"))
        healthy_img = np.array(Image.open(healthy_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=diseased_img, image0=healthy_img)
            diseased_img = augmentations["image"]
            healthy_img = augmentations["image0"]

        return diseased_img, healthy_img