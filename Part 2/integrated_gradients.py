import os
from tqdm import tqdm
from argparse import ArgumentParser
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


data_root = "chest_xray"
device = "cuda"
model_path = "models/model_224_3.pth"
attributions_save_path = "attributions_ig.npy"
batch_size = 1
image_size = 64
center_crop_size = 64
n_images = 10

parser = ArgumentParser()
parser.add_argument("--data_root", type=str, required=False, default=data_root)
args = parser.parse_args()
data_root = args.data_root

transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(center_crop_size),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
unchanged_transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(center_crop_size),
                       T.ToTensor()])
test_dataset = XrayDataset(os.path.join(data_root, "val"), transform=transform, unchanged_transform=unchanged_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
ig = IntegratedGradients(model)

attributions_ig = []
labels = []
original_images = []

for i, (image, label, original_image) in enumerate(tqdm(test_loader)):
    image = image.to(device)
    label = label.to(device)

    out = model(image)
    pred_label = torch.argmax(out, dim=1)

    original_images.append(np.transpose(original_image.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
    labels.append(label.item())
    attribution_ig = np.transpose(ig.attribute(image, target=label, n_steps=200).squeeze().cpu().detach().numpy(),
                                  (1, 2, 0))
    attributions_ig.append(attribution_ig)

attributions_ig = np.array(attributions_ig)
labels = np.array(labels)
np.save(attributions_save_path, attributions_ig)

healthy_indices = np.argwhere(labels == 0).flatten()
pneumonia_indices = np.argwhere(labels == 1).flatten()
visualize_indices = np.concatenate((healthy_indices[:n_images // 2], pneumonia_indices[:n_images // 2]))

for i in visualize_indices:
    fig, ax = plt.subplots(1, 2)

    _ = viz.visualize_image_attr(None, original_images[i],
                                 method="original_image", title="Original Image", plt_fig_axis=(fig, ax[0]),
                                 use_pyplot=False)

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#0000ff'),
                                                      (1, '#0000ff')], N=256)

    _ = viz.visualize_image_attr(attributions_ig[i],
                                 original_images[i],
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 title='Integrated Gradients',
                                 plt_fig_axis=(fig, ax[1]),
                                 use_pyplot=False)

    label = "pneumonia" if labels[i] else "healthy"
    plt.savefig(f"ig_{i}_{label}.png")
