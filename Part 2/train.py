import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import XrayDataset


# Parameters:
data_root = "archive"
device = "cuda"
batch_size = 1
epochs = 1
test_frequency = 1
model_save_path = "model.pth"
loss_save_path = "losses.npz"

# Creating train and val data loaders:
transform = T.Compose([T.Resize((256, 256)),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_dataset = XrayDataset(os.path.join(data_root, "train"), transform=transform)
val_dataset = XrayDataset(os.path.join(data_root, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Initializing model and loss function:
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

# Initializing optimizer:
fc_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
optimizer = optim.Adam([{"params": base_params, "lr": 1e-4},
                        {"params": model.fc.parameters(), "lr": 1e-3}])


# Train for one epoch:
def train(model, dataloader, loss_fn, optimizer):
    model.train()
    loss_list = []

    for i, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        out = model(image)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    return loss_list


# Test/validate:
def test(model, dataloader, loss_fn):
    model.eval()
    loss_list = []

    with torch.no_grad():
        for i, (image, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)

            out = model(image)
            loss = loss_fn(out, label)

            loss_list.append(loss.item())

    return loss_list


# Train for desired number of epochs:
train_loss = []
val_loss = []
for epoch in range(epochs):
    loss = train(model=model, dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
    train_loss += loss
    print(f"[TRAIN] Epoch {epoch}, average train loss: {sum(loss) / len(loss)}")

    if not epoch % test_frequency:
        loss += test(model=model, dataloader=val_loader, loss_fn=loss_fn)
        val_loss += loss
        print(f"[VAL] Epoch {epoch}, average val loss: {sum(loss) / len(loss)}")

# Saving model and losses:
train_loss = np.array(train_loss)
val_loss = np.array(val_loss)
np.savez(loss_save_path, train_loss=train_loss, val_loss=val_loss)
torch.save(model.state_dict(), model_save_path)
