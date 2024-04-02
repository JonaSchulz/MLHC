import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, roc_auc_score

from dataset import XrayDataset


# Parameters:
device = "cuda"
batch_size = 2
model_path = "model.pth"

# Creating test data loader:
transform = T.Compose([T.Resize((256, 256)),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_dataset = XrayDataset("archive/chest_xray/test", transform=transform)
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
        for i, (image, label) in enumerate(dataloader):
            if i == 10:
                break
            image = image.to(device)
            label = label.to(device)

            out = model(image)
            loss = loss_fn(out, label)

            loss_list.append(loss.item())
            gt_labels += [l.item() for l in label]
            logits += [l for l in out]

    return loss_list, gt_labels, logits


test_loss, gt_labels, logits = test(model=model, dataloader=test_loader, loss_fn=loss_fn)
print(f"Average test loss: {sum(test_loss) / len(test_loss)}")

f1 = f1_score(np.array(gt_labels), np.array([torch.argmax(l, dim=-1).item() for l in logits]))
print(f"F1 Score: {f1}")
