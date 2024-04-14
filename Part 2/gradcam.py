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
model_path = "model_224_rl.pth"

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

for i, (image, label, original_image) in enumerate(tqdm(test_loader)):
    # Preparing input tensor and feeding into GradCAM:
    grayscale_cam = cam(input_tensor=image)

    grayscale_cam = grayscale_cam[0, :]
    gradcam_image = visualization = show_cam_on_image(original_image.squeeze(0).permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)

    model_outputs = cam.outputs
    print(f"Model Outputs: {torch.argmax(model_outputs)}")

    plt.imshow(gradcam_image)
    plt.savefig(f"gradcam_{i}_healthy.png")
