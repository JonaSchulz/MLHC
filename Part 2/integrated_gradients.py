import os
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms as T
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import XrayDataset


data_root = "chest_xray"
device = "cuda"
model_path = "models/model_224_long.pth"
attributions_save_path = "attributions_ig.npy"
batch_size = 1
image_size = 256
center_crop_size = 224
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
# np.save(attributions_save_path, attributions_ig)

healthy_indices = np.argwhere(labels == 0).flatten()
pneumonia_indices = np.argwhere(labels == 1).flatten()
visualize_indices = np.concatenate((healthy_indices[:n_images // 2], pneumonia_indices[:n_images // 2]))

fig, ax = plt.subplots(10, 3, figsize=(10, 30))

for i, ind in enumerate(visualize_indices):
    label = "pneumonia" if labels[ind] else "healthy"
    # fig, ax = plt.subplots(1, 2)
    ax[i, 0].set_axis_on()

    _ = viz.visualize_image_attr(None, original_images[ind],
                                 method="original_image", title="Original Image" if i == 0 else None, plt_fig_axis=(fig, ax[i, 0]),
                                 use_pyplot=False)

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#0000ff'),
                                                      (1, '#0000ff')], N=256)

    _ = viz.visualize_image_attr(attributions_ig[ind],
                                 original_images[ind],
                                 method='heat_map',
                                 cmap=default_cmap,
                                 sign='positive',
                                 title='Integrated Gradients' if i == 0 else None,
                                 plt_fig_axis=(fig, ax[i, 1]),
                                 use_pyplot=False)
    ax[i, 0].set_ylabel(label, fontsize="x-large")

    #plt.savefig(f"ig_{i}_{label}.png")


########################
# GRADCAM:

# Parameters:
attributions_save_path = "attributions_cam.npy"

# Initializing GradCAM with last convolutional block of resnet model:
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

attributions_cam = []
labels = []
original_images = []

for i, (image, label, original_image) in enumerate(tqdm(test_loader)):
    # Preparing input tensor and feeding into GradCAM:
    grayscale_cam = cam(input_tensor=image)
    grayscale_cam = grayscale_cam[0, :]

    original_images.append(np.transpose(original_image.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
    attributions_cam.append(grayscale_cam)
    labels.append(label.item())

attributions_cam = np.array(attributions_cam)
labels = np.array(labels)
# np.save(attributions_save_path, attributions_cam)

healthy_indices = np.argwhere(labels == 0).flatten()
pneumonia_indices = np.argwhere(labels == 1).flatten()
# visualize_indices = np.concatenate((healthy_indices[:n_images // 2], pneumonia_indices[:n_images // 2]))

for i, ind in enumerate(visualize_indices):
    label = "pneumonia" if labels[ind] else "healthy"

    gradcam_image = visualization = show_cam_on_image(original_images[ind],
                                                      attributions_cam[ind], use_rgb=True)
    ax[i, 2].imshow(gradcam_image)
    ax[i, 2].set_axis_off()

    # plt.savefig(os.path.join(image_save_path, f"gradcam_{i}_{label}.png"))

ax[0, 2].set_title("Grad-CAM")
plt.savefig(f"ig_gradcam_10_images.png")
