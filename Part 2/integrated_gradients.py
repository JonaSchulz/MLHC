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
model_path = "model.pth"
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
transform_unchanged = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(224),
                       T.ToTensor()])
test_dataset = XrayDataset(os.path.join(data_root, "val"), transform=transform)
test_dataset_unchanged = XrayDataset(os.path.join(data_root, "val"), transform=transform_unchanged)

example_images_healthy = [test_dataset[i][0] for i in range(5)]
example_images_disease = [test_dataset[-i][0] for i in range(1, 6)]
example_images_healthy_unchanged = [test_dataset_unchanged[i][0] for i in range(5)]
example_images_disease_unchanged = [test_dataset_unchanged[-i][0] for i in range(1, 6)]

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
ig = IntegratedGradients(model)

#fig, ax = plt.subplots(2, 5)


for i, image in enumerate(example_images_disease):
    image = image.unsqueeze(0).to(device)
    image_unchanged = example_images_healthy_unchanged[i]

    out = model(image)
    out = F.softmax(out, dim=1)
    pred_score, pred_label = torch.topk(out, 1)

    print(pred_label)

    attributions_ig = ig.attribute(image, target=pred_label, n_steps=200)
    attributions_ig /= torch.max(attributions_ig)

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

    plt.savefig(f"ig_{i}.png")


