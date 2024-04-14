import os
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
model_path = "model_224_rl.pth"
batch_size = 1
image_size = 256

parser = ArgumentParser()
parser.add_argument("--data_root", type=str, required=False, default=data_root)
args = parser.parse_args()
data_root = args.data_root

transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
unchanged_transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(224),
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

for i, (image, label, original_image) in enumerate(test_loader):
    image = image.to(device)
    label = label.to(device)

    out = model(image)
    pred_label = torch.argmax(out, dim=1)

    attributions_ig.append(ig.attribute(image, target=label, n_steps=200).flatten().cpu().numpy())



for i, image in enumerate(example_images_healthy):
    image = image.unsqueeze(0).to(device)
    image_unchanged = example_images_healthy_unchanged[i]

    out = model(image)
    out = F.softmax(out, dim=1)
    pred_score, pred_label = torch.topk(out, 1)

    print(pred_label)

    attributions_ig = ig.attribute(image, target=pred_label, n_steps=200)

    #ax[0][i].imshow(np.transpose(image_unchanged.detach().numpy(), (1, 2, 0)))
    #ax[1][i].imshow(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)))

    fig, ax = plt.subplots(1, 2)

    _ = viz.visualize_image_attr(None, np.transpose(image_unchanged.cpu().detach().numpy(), (1, 2, 0)),
                                 method="original_image", title="Original Image", plt_fig_axis=(fig, ax[0]), use_pyplot=False)

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
                                 title='Integrated Gradients',
                                 plt_fig_axis=(fig, ax[1]),
                                 use_pyplot=False)

    plt.savefig(f"ig_{i}_healthy.png")


for i, image in enumerate(example_images_disease):
    image = image.unsqueeze(0).to(device)
    image_unchanged = example_images_disease_unchanged[i]

    out = model(image)
    out = F.softmax(out, dim=1)
    pred_score, pred_label = torch.topk(out, 1)

    print(pred_label)

    attributions_ig = ig.attribute(image, target=pred_label, n_steps=200)

    #ax[0][i].imshow(np.transpose(image_unchanged.detach().numpy(), (1, 2, 0)))
    #ax[1][i].imshow(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)))

    fig, ax = plt.subplots(1, 2)

    _ = viz.visualize_image_attr(None, np.transpose(image_unchanged.cpu().detach().numpy(), (1, 2, 0)),
                                 method="original_image", title="Original Image", plt_fig_axis=(fig, ax[0]), use_pyplot=False)

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
                                 title='Integrated Gradients',
                                 plt_fig_axis=(fig, ax[1]),
                                 use_pyplot=False)

    plt.savefig(f"ig_{i}_disease.png")


