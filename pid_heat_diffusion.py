import numpy as np
import cv2
import matplotlib.pyplot as plt

def heat_diffusion(image, alpha, iterations=1):
    """Apply heat diffusion (isotropic smoothing) to an image adaptively."""
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    for _ in range(iterations):
        laplacian = cv2.filter2D(image, -1, kernel)
        image = image + np.multiply(alpha, laplacian)  # Element-wise multiplication
        image = np.clip(image, 0, 255)  # Ensure pixel values are within [0, 255]
    
    return image.astype(np.uint8)

def compute_edge_strength(image):
    """Compute edge strength using gradient magnitude."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    edge_strength = cv2.normalize(edge_strength, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
    return edge_strength

def anisotropic_diffusion(image, iterations=20, kappa=15, gamma=0.05):
    """Apply Perona-Malik anisotropic diffusion with improved edge preservation."""
    image = image.astype(np.float32)
    for _ in range(iterations):
        nablaN = np.roll(image, -1, axis=0) - image
        nablaS = np.roll(image, 1, axis=0) - image
        nablaE = np.roll(image, -1, axis=1) - image
        nablaW = np.roll(image, 1, axis=1) - image
        diff_coef = np.exp(-(np.abs(nablaN) / kappa) ** 2)
        image += gamma * (diff_coef * nablaN + diff_coef * nablaS + diff_coef * nablaE + diff_coef * nablaW)
    return np.clip(image, 0, 255).astype(np.uint8)

def pid_controlled_diffusion(image, alpha_0, K_p, K_i, K_d, iterations=10):
    """Apply PID-controlled heat diffusion to an image with adaptive alpha scaling."""
    alpha = alpha_0 * np.ones_like(image, dtype=np.float32)
    integral_error = np.zeros_like(image, dtype=np.float32)
    prev_error = np.zeros_like(image, dtype=np.float32)
    
    for _ in range(iterations):
        edge_strength = compute_edge_strength(image)
        desired_edge_strength = 100  # Target edge strength
        error = desired_edge_strength - edge_strength
        
        integral_error += error
        derivative_error = error - prev_error
        prev_error = error
        
        # Compute adaptive alpha using PID with edge-based scaling
        alpha = alpha_0 + K_p * error + K_i * integral_error + K_d * derivative_error
        alpha = np.clip(alpha, 0, np.maximum(1, 5 - edge_strength / 50))  # Edge-aware diffusion limit
        
        image = heat_diffusion(image, alpha)
    
    return image

def unsharp_mask(image, strength=2.0):
    """Apply unsharp masking to enhance edges."""
    blurred = cv2.GaussianBlur(image, (3, 3), 1)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# Load the image
image = cv2.imread("try12.bmp", cv2.IMREAD_GRAYSCALE)

# Normalize image to [0, 255]
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# Step 1: Upscale the image by 8x using bicubic interpolation
upscaled_image = cv2.resize(image, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)

# Step 2: Apply Anisotropic Diffusion (or Bilateral Filtering as an alternative)
use_bilateral = True  # Set to False to use anisotropic diffusion instead
if use_bilateral:
    denoised_image = cv2.bilateralFilter(upscaled_image, d=9, sigmaColor=75, sigmaSpace=75)
else:
    denoised_image = anisotropic_diffusion(upscaled_image, iterations=20, kappa=15, gamma=0.05)

# Set PID parameters
alpha_0 = 0.1  # Baseline diffusion coefficient
K_p = 0.00001  # Proportional gain
K_i = 0.0001   # Integral gain
K_d = 0.005    # Derivative gain

# Step 3: Apply PID-controlled heat diffusion
output_image = pid_controlled_diffusion(denoised_image, alpha_0, K_p, K_i, K_d, iterations=10)

# Step 4: Apply Unsharp Masking for final edge enhancement
sharpened_image = unsharp_mask(output_image, strength=2.0)

# Save and display results
cv2.imwrite("sharpened_output.jpg", sharpened_image)

plt.figure(figsize=(15, 6))
plt.subplot(1, 5, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 5, 2)
plt.title("Upscaled Image (8x Bicubic)")
plt.imshow(upscaled_image, cmap="gray")

plt.subplot(1, 5, 3)
plt.title("Denoised Image")
plt.imshow(denoised_image, cmap="gray")

plt.subplot(1, 5, 4)
plt.title("PID + Heat Diffusion")
plt.imshow(output_image, cmap="gray")

plt.subplot(1, 5, 5)
plt.title("Final Sharpened Image")
plt.imshow(sharpened_image, cmap="gray")

plt.show()
