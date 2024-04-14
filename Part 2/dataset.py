import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset


class XrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_transform=None, unchanged_transform=None, randomize_labels=False):
        self.root_dir = root_dir
        self.transform = transform
        self.label_transform = label_transform
        self.unchanged_transform = unchanged_transform
        self.labels = ([0 for _ in os.listdir(os.path.join(root_dir, "NORMAL"))] +
                  [1 for _ in os.listdir(os.path.join(root_dir, "PNEUMONIA"))])
        self.file_list = ([os.path.join(root_dir, "NORMAL", f) for f in
                           os.listdir(os.path.join(root_dir, "NORMAL"))] +
                          [os.path.join(root_dir, "PNEUMONIA", f) for f in
                           os.listdir(os.path.join(root_dir, "PNEUMONIA"))])
        if randomize_labels:
            random.shuffle(self.labels)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        image = Image.open(self.file_list[item]).convert("RGB")
        label = self.labels[item]
        if self.transform:
            image_transformed = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        if self.unchanged_transform:
            image_unchanged = self.unchanged_transform(image)
            return image_transformed, label, image_unchanged
        return image_transformed, label
