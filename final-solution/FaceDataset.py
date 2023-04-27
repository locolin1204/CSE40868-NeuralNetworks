import torch
from torchvision.io import read_image
import os
from torch.utils.data import Dataset
import pandas as pd


class FaceDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform):
        self.img_labels = pd.read_csv(label_dir, header=0)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image.to(torch.float) / 256.
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return (image, label)