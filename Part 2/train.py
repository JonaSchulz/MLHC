import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt

from dataset import XrayDataset


device = "cpu"
transform = T.Compose([T.Resize((256, 256)),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = XrayDataset("archive/chest_xray/train", transform=transform)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters())

for i, (image, label) in enumerate(train_loader):
    image.to(device)
    label.to(device)
    out = model(image)
    pass