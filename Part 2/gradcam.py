import os
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
batch_size = 1
model_path = "model_224.pth"

# Creating test data loader:
transform = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_unchanged = T.Compose([T.Resize((image_size, image_size)),
                       T.CenterCrop(224),
                       T.ToTensor()])

test_dataset = XrayDataset(os.path.join(data_root, split), transform=transform)
test_dataset_unchanged = XrayDataset(os.path.join(data_root, split), transform=transform_unchanged)

example_images_healthy = [test_dataset[i][0] for i in range(5)]
example_images_disease = [test_dataset[-i][0] for i in range(1, 6)]
example_images_healthy_unchanged = [test_dataset_unchanged[i][0] for i in range(5)]
example_images_disease_unchanged = [test_dataset_unchanged[-i][0] for i in range(1, 6)]

# Initializing model:
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights="ResNet34_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load(model_path))
model.to(device)

# Initializing GradCAM with last convolutional block of resnet model:
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

for i, img in enumerate(example_images_healthy):
    # Preparing input tensor and feeding into GradCAM:
    grayscale_cam = cam(input_tensor=img.unsqueeze(0))

    grayscale_cam = grayscale_cam[0, :]
    original_image = example_images_healthy_unchanged[i].permute(1, 2, 0).numpy()
    gradcam_image = visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    model_outputs = cam.outputs
    print(f"Model Outputs: {torch.argmax(model_outputs)}")

    plt.imshow(gradcam_image)
    plt.savefig(f"gradcam_{i}_healthy.png")

for i, img in enumerate(example_images_disease):
    # Preparing input tensor and feeding into GradCAM:
    grayscale_cam = cam(input_tensor=img.unsqueeze(0))

    grayscale_cam = grayscale_cam[0, :]
    original_image = example_images_disease_unchanged[i].permute(1, 2, 0).numpy()
    gradcam_image = visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    model_outputs = cam.outputs
    print(f"Model Outputs: {torch.argmax(model_outputs)}")

    plt.imshow(gradcam_image)
    plt.savefig(f"gradcam_{i}_disease.png")