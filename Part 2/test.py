from tqdm import tqdm
import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, roc_auc_score

from dataset import XrayDataset


# Parameters:
data_root = "chest_xray"
device = "cuda"
image_size = 512
batch_size = 1
model_path = "model_224_2.pth"

parser = ArgumentParser()
parser.add_argument("--data_root", type=str, required=False, default=data_root)
args = parser.parse_args()
data_root = args.data_root

# Creating test data loader:
transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(512),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_dataset = XrayDataset(os.path.join(data_root, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Initializing model and loss function:
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)
loss_fn = nn.CrossEntropyLoss().to(device)


# Test:
def test(model, dataloader, loss_fn):
    model.eval()
    loss_list = []
    gt_labels = []
    logits = []

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            label = label.to(device)

            out = model(image)
            loss = loss_fn(out, label)

            loss_list.append(loss.item())
            gt_labels += [l.item() for l in label]
            logits += [l for l in out]

    return loss_list, gt_labels, logits


test_loss, gt_labels, logits = test(model=model, dataloader=test_loader, loss_fn=loss_fn)
pred_labels = [torch.argmax(l, -1).item() for l in logits]
tp, fp, tn, fn = 0, 0, 0, 0
for gt, pred in zip(gt_labels, pred_labels):
    if gt == 1 and pred == 1:
        tp += 1
    elif gt == 0 and pred == 1:
        fp += 1
    elif gt == 0 and pred == 0:
        tn += 1
    elif gt == 1 and pred == 0:
        fn += 1

print(f"Average test loss: {sum(test_loss) / len(test_loss)}")
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN:{fn}")

f1 = f1_score(np.array(gt_labels), np.array([torch.argmax(l, dim=-1).item() for l in logits]))
print(f"F1 Score: {f1}")

roc_auc = roc_auc_score(np.array(gt_labels), np.array([F.softmax(l, dim=0)[1].item() for l in logits]))
print(f"AUC ROC Score: {roc_auc}")
