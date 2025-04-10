import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load images
original = cv2.imread("try11.bmp", cv2.IMREAD_GRAYSCALE)
super_res = cv2.imread("sharpened_output.jpg", cv2.IMREAD_GRAYSCALE)

# Compute PSNR
psnr_value = psnr(original, super_res, data_range=255)
print(f"PSNR: {psnr_value:.2f} dB")

# Compute SSIM
ssim_value = ssim(original, super_res, data_range=255)
print(f"SSIM: {ssim_value:.4f}")
