import os
import json
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import XrayDataset


# Parameters:
data_root = "chest_xray"
device = "cuda"
image_size = 256
center_crop_size = 224
batch_size = 16
base_lr = 1e-4
fc_lr = 1e-3
epochs = 50
val_frequency = 10
run_name = "test"

parser = ArgumentParser()
parser.add_argument("--data_root", type=str, required=False, default=data_root)
parser.add_argument("--randomize_labels", type=bool, required=False, default=False)
parser.add_argument("--run_name", type=str, required=False, default=run_name)
parser.add_argument("--epochs", type=int, required=False, default=epochs)
args = parser.parse_args()
data_root = args.data_root
randomize_labels = args.randomize_labels
print(randomize_labels)
model_save_path = f"{args.run_name}_model.pth"
loss_save_path = f"{args.run_name}_loss.npz"
epochs = args.epochs

# Create hyperparameter info file:
hyperparameters = {
    "model_save_path": model_save_path,
    "loss_save_path": loss_save_path,
    "batch_size": batch_size,
    "base_lr": base_lr,
    "fc_lr": fc_lr,
    "epochs": epochs,
    "image_size": image_size,
    "center_crop_size": center_crop_size,
}
with open(f"{model_save_path.split('.')[0]}_info.json", "w+") as fp:
    json.dump(hyperparameters, fp)


# Creating train and val data loaders:
transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(center_crop_size),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_dataset = XrayDataset(os.path.join(data_root, "val"), transform=transform, randomize_labels=randomize_labels)
val_dataset = XrayDataset(os.path.join(data_root, "val"), transform=transform)
test_dataset = XrayDataset(os.path.join(data_root, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Initializing model and loss function:
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

# Initializing optimizer:
fc_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
optimizer = optim.Adam([{"params": base_params, "lr": base_lr},
                        {"params": model.fc.parameters(), "lr": fc_lr}])


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
best_val_loss = 100
for epoch in range(epochs):
    loss = train(model=model, dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
    train_loss += loss
    print(f"[TRAIN] Epoch {epoch}, average train loss: {sum(loss) / len(loss)}")

    if not epoch % val_frequency:
        loss = test(model=model, dataloader=val_loader, loss_fn=loss_fn)
        val_loss.append(sum(loss) / len(loss))
        if sum(loss) / len(loss) < best_val_loss:
            best_val_loss = sum(loss) / len(loss)
            torch.save(model.state_dict(), f"{model_save_path.split('.')[0]}_best.pth")
        print(f"[VAL] Epoch {epoch}, average val loss: {sum(loss) / len(loss)}")

# Saving model and losses:
train_loss = np.array(train_loss)
val_loss = np.array(val_loss)
np.savez(loss_save_path, train_loss=train_loss, val_loss=val_loss)
torch.save(model.state_dict(), model_save_path)
