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
model_rl_path = "models/model_224_rl.pth"
attributions_save_path = "attributions_ig.npy"
batch_size = 1
image_size = 256
center_crop_size = 224
n_images = 100

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
test_dataset = XrayDataset(os.path.join(data_root, "test"), transform=transform, unchanged_transform=unchanged_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

model_rl = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model_rl.fc = nn.Linear(512, 2)
model_rl.load_state_dict(torch.load(model_rl_path))
model_rl.to(device)
model_rl.eval()

ig = IntegratedGradients(model)

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

ig_rl = IntegratedGradients(model_rl)

target_layers = [model_rl.layer4[-1]]
cam_rl = GradCAM(model=model_rl, target_layers=target_layers)

attributions_ig = []
attributions_cam = []
attributions_ig_rl = []
attributions_cam_rl = []
labels = []
original_images = []

for i, (image, label, original_image) in enumerate(tqdm(test_loader)):
    image = image.to(device)
    label = label.to(device)

    out = model(image)
    pred_label = torch.argmax(out, dim=1)

    original_images.append(np.transpose(original_image.squeeze(0).cpu().detach().numpy(), (1, 2, 0)))
    labels.append(label.item())

    grayscale_cam = cam(input_tensor=image)
    grayscale_cam = grayscale_cam[0, :]
    attributions_cam.append(grayscale_cam)

    grayscale_cam_rl = cam_rl(input_tensor=image)
    grayscale_cam_rl = grayscale_cam_rl[0, :]
    attributions_cam_rl.append(grayscale_cam_rl)

    attribution_ig = np.transpose(ig.attribute(image, target=label, n_steps=200).squeeze().cpu().detach().numpy(),
                                  (1, 2, 0))
    attributions_ig.append(attribution_ig)

    attribution_ig_rl = np.transpose(ig_rl.attribute(image, target=label, n_steps=200).squeeze().cpu().detach().numpy(),
                                  (1, 2, 0))
    attributions_ig_rl.append(attribution_ig_rl)

attributions_ig = np.array(attributions_ig)
attributions_cam = np.array(attributions_cam)
attributions_ig_rl = np.array(attributions_ig_rl)
attributions_cam_rl = np.array(attributions_cam_rl)
labels = np.array(labels)
# np.save(attributions_save_path, attributions_ig)

healthy_indices = np.argwhere(labels == 0).flatten()[:n_images // 2]
pneumonia_indices = np.argwhere(labels == 1).flatten()[:n_images // 2]

for i in range(len(healthy_indices) + len(pneumonia_indices)):
    ind = healthy_indices[i // 2] if i % 10 < 5 else pneumonia_indices[i // 2]
    if not i % 10:
        if i != 0:
            ax[0, 2].set_title("Grad-CAM")
            ax[0, 4].set_title("Grad-CAM (randomized)")
            plt.savefig(f"ig_gradcam_{i // 10}.png")
        fig, ax = plt.subplots(10, 5, figsize=(15, 30))

    label = "pneumonia" if labels[ind] else "healthy"
    # fig, ax = plt.subplots(1, 2)
    ax[i % 10, 0].set_axis_on()

    _ = viz.visualize_image_attr(None, original_images[ind],
                                 method="original_image", title="Original Image" if not i % 10 else None,
                                 plt_fig_axis=(fig, ax[i % 10, 0]),
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
                                 title='Integrated Gradients' if not i % 10 else None,
                                 plt_fig_axis=(fig, ax[i % 10, 1]),
                                 use_pyplot=False)

    _ = viz.visualize_image_attr(attributions_ig_rl[ind],
                                 original_images[ind],
                                 method='heat_map',
                                 cmap=default_cmap,
                                 sign='positive',
                                 title='Integ. Grad. (randomized)' if not i % 10 else None,
                                 plt_fig_axis=(fig, ax[i % 10, 3]),
                                 use_pyplot=False)
    ax[i % 10, 0].set_ylabel(label, fontsize="x-large")

    gradcam_image = show_cam_on_image(original_images[ind], attributions_cam[ind], use_rgb=True)
    gradcam_image_rl = show_cam_on_image(original_images[ind], attributions_cam_rl[ind], use_rgb=True)
    ax[i % 10, 2].imshow(gradcam_image)
    ax[i % 10, 2].set_axis_off()
    ax[i % 10, 4].imshow(gradcam_image_rl)
    ax[i % 10, 4].set_axis_off()

    #plt.savefig(f"ig_{i}_{label}.png")
