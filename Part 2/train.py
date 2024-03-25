import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt

from dataset import XrayDataset

transform = T.Compose([T.Resize((256, 256)),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = XrayDataset("archive/chest_xray/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)

