import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import captum
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dataset import XrayDataset


data_root = "archive"
device = "cuda"
model_path = "model.pth"
batch_size = 1

transform = T.Compose([T.Resize((64, 64)),
                       T.CenterCrop(64),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_dataset = XrayDataset("archive/chest_xray/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
ig = IntegratedGradients(model)

for i, (image, label) in enumerate(test_loader):
    image = image.to(device)
    label = label.to(device)

    out = model(image)
    out = F.softmax(out, dim=1)
    pred_score, pred_label = torch.topk(out, 1)

    attributions_ig = ig.attribute(image, target=pred_label, n_steps=200)
    _ = viz.visualize_image_attr(None, np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method="original_image", title="Original Image")

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#0000ff'),
                                                      (1, '#0000ff')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 title='Integrated Gradients')

    break


