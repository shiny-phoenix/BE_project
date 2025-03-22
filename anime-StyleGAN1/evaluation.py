# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# !pip install lpips

# +
import argparse
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import scipy.linalg
from PIL import Image
from tqdm import tqdm
import os

# Define InceptionV3 Model for FID
class InceptionV3FID(nn.Module):
    def __init__(self):
        super(InceptionV3FID, self).__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        self.model = inception
        self.model.fc = nn.Identity()  # Remove the classification layer

    def forward(self, x):
        return self.model(x)  # Extract features

# Image Preprocessing Function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize for InceptionV3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Extract Features from Images
def get_activations(image_paths, model, device):
    model.eval()
    model.to(device)
    activations = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting Features"):
            img = preprocess_image(img_path).to(device)  # Convert image to tensor
            act = model(img)
            activations.append(act.cpu().numpy())  # Move to CPU before converting to NumPy

    activations = np.concatenate(activations, axis=0)
    return activations

# Compute FID Score
def calculate_fid(real_acts, gen_acts):
    mu_real, sigma_real = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    mu_gen, sigma_gen = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)

    diff = np.sum((mu_real - mu_gen) ** 2)
    
    try:
        covmean, _ = scipy.linalg.sqrtm(sigma_real @ sigma_gen, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception as e:
        print(f"Error in sqrtm calculation: {e}")
        covmean = np.zeros_like(sigma_real)  # Fallback to zeros
    
    fid = diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

# Compute FID
def compute_fid(real_images_dir, generated_images_dir, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = InceptionV3FID().to(device)

    real_images = [os.path.join(real_images_dir, img) for img in os.listdir(real_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    gen_images = [os.path.join(generated_images_dir, img) for img in os.listdir(generated_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    if not real_images or not gen_images:
        print("Error: Make sure both real and generated images folders contain valid images.")
        return None

    real_acts = get_activations(real_images, model, device)
    gen_acts = get_activations(gen_images, model, device)

    fid_score = calculate_fid(real_acts, gen_acts)
    
    print(f"FID Score: {fid_score:.2f}")
    return fid_score

# Main Function with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID Score between real and generated images.")
    parser.add_argument("--real", type=str, required=True, help="Path to folder containing real images.")
    parser.add_argument("--generated", type=str, required=True, help="Path to folder containing generated images.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu).")

    args = parser.parse_args()

    compute_fid(args.real, args.generated, args.device)

