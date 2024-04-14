import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

attribution_file = "attributions_cam.npy"
attribution_file_rl = "attributions_cam.npy"

attributions = np.load(attribution_file)
attributions_rl = np.load(attribution_file_rl)

average_ssim = 0
for attribution, attribution_rl in zip(attributions, attributions_rl):
    attribution /= np.max(np.abs(attribution))
    attribution_rl /= np.max(np.abs(attribution_rl))
    mssim = ssim(attribution, attribution_rl, win_size=5, data_range=1)
    average_ssim += mssim / len(attribution)

print(average_ssim)

