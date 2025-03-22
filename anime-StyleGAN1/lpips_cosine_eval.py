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

# +
import torch
import lpips
import argparse
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
import torch.nn.functional as F

# Initialize LPIPS model
loss_fn = lpips.LPIPS(net='vgg')

# Initialize ResNet model for feature extraction
model = resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = transform(Image.open(image_path)).unsqueeze(0)
    return img

def compute_lpips(image1, image2):
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    return loss_fn(img1, img2).item()

def get_features(image_path):
    img = preprocess_image(image_path)
    return model(img).detach()

def compute_cosine_similarity(img1, img2):
    feat1 = get_features(img1)
    feat2 = get_features(img2)
    return F.cosine_similarity(feat1, feat2).item()

def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS and Cosine Similarity")
    parser.add_argument("merged", type=str, help="Path to merged image")
    parser.add_argument("input_a", type=str, help="Path to input image A")
    parser.add_argument("input_b", type=str, help="Path to input image B")
    args = parser.parse_args()
    
    # Compute LPIPS Similarity
    lpips_a = compute_lpips(args.merged, args.input_a)
    lpips_b = compute_lpips(args.merged, args.input_b)
    
    # Compute Cosine Similarity
    cosine_a = compute_cosine_similarity(args.merged, args.input_a)
    cosine_b = compute_cosine_similarity(args.merged, args.input_b)
    
    print(f"LPIPS Similarity (Merged vs A): {lpips_a:.4f}")
    print(f"LPIPS Similarity (Merged vs B): {lpips_b:.4f}")
    print(f"Cosine Similarity (Merged vs A): {cosine_a:.4f}")
    print(f"Cosine Similarity (Merged vs B): {cosine_b:.4f}")

if __name__ == "__main__":
    main()

