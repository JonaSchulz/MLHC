import os
from PIL import Image
import torch
from torch.utils.data import Dataset


# TODO: Data Augmentation
class XrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_transform = label_transform
        self.file_list = ([(os.path.join(root_dir, "NORMAL", f), 0) for f in os.listdir(os.path.join(root_dir, "NORMAL"))] +
                          [(os.path.join(root_dir, "PNEUMONIA", f), 1) for f in os.listdir(os.path.join(root_dir, "PNEUMONIA"))])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        image = Image.open(self.file_list[item][0]).convert("RGB")
        label = self.file_list[item][1]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label
