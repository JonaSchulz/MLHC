import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import XrayDataset


# Parameters:
data_root = "chest_xray"
split = "val"
device = "cuda"
image_size = 256
center_crop_size = 224
batch_size = 1
model_path = "models/model_224_rl.pth"
attributions_save_path = "attributions_cam.npy"
n_images = 10

# Creating test data loader:
transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(center_crop_size),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
unchanged_transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(center_crop_size),
                       T.ToTensor()])
test_dataset = XrayDataset(os.path.join(data_root, "val"), transform=transform, unchanged_transform=unchanged_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initializing model:
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

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
np.save(attributions_save_path, attributions_cam)

healthy_indices = np.argwhere(labels == 0).flatten()
pneumonia_indices = np.argwhere(labels == 1).flatten()
visualize_indices = np.concatenate((healthy_indices[:n_images // 2], pneumonia_indices[:n_images // 2]))

for i in visualize_indices:
    gradcam_image = visualization = show_cam_on_image(original_images[i],
                                                      attributions_cam[i], use_rgb=True)
    plt.imshow(gradcam_image)

    label = "pneumonia" if labels[i] else "healthy"
    plt.savefig(f"gradcam_{i}_{label}.png")
