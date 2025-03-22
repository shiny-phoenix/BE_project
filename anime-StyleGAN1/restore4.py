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
import argparse
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from restore3 import StyleGANWithEncoder, preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to trained StyleGAN weights")
parser.add_argument("--image1", type=str, required=True, help="Path to first input image")
parser.add_argument("--image2", type=str, required=True, help="Path to second input image")
parser.add_argument("--output_image", type=str, default="merged_output.png")
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--merge_ratio", type=float, default=0.5, help="Weighting factor for merging features (0 to 1)")

def main(args):
    # Create model
    img_shape = (args.image_size, args.image_size, 3)
    model = StyleGANWithEncoder(img_shape=img_shape, latent_dim=args.latent_dim)
    model.load_weights(args.model_path)
    
    # Load and preprocess images
    img1 = preprocess_image(args.image1, (args.image_size, args.image_size))
    img2 = preprocess_image(args.image2, (args.image_size, args.image_size))
    
    img1 = np.expand_dims(img1, axis=0)  # Add batch dimension
    img2 = np.expand_dims(img2, axis=0)

    # Encode images to get latent vectors
    z1 = model.encode_images(img1)
    z2 = model.encode_images(img2)

    # Reconstruct images from latent vectors
    rec1 = model.g([tf.ones((1, 1)), tf.repeat(tf.expand_dims(z1, axis=1), model.n_style_block, axis=1), tf.random.normal([1, args.image_size, args.image_size])])
    rec2 = model.g([tf.ones((1, 1)), tf.repeat(tf.expand_dims(z2, axis=1), model.n_style_block, axis=1), tf.random.normal([1, args.image_size, args.image_size])])
    
    # Merge latent vectors (weighted sum)
    merged_z = args.merge_ratio * z1 + (1 - args.merge_ratio) * z2
    
    # Generate merged image
    batch_size = tf.shape(merged_z)[0]
    ones = tf.ones((batch_size, 1))
    z_repeated = tf.repeat(tf.expand_dims(merged_z, axis=1), model.n_style_block, axis=1)
    noise = tf.random.normal([batch_size, args.image_size, args.image_size])
    
    merged_image = model.g([ones, z_repeated, noise])

    # Convert images to display format
    img1 = ((img1[0] + 1) * 127.5).astype(np.uint8)
    img2 = ((img2[0] + 1) * 127.5).astype(np.uint8)
    rec1 = ((rec1.numpy()[0] + 1) * 127.5).astype(np.uint8)
    rec2 = ((rec2.numpy()[0] + 1) * 127.5).astype(np.uint8)
    merged_image = ((merged_image.numpy()[0] + 1) * 127.5).astype(np.uint8)

    # Display and save results
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    axes[0].imshow(img1)
    axes[0].set_title("Input Image 1")
    axes[0].axis("off")
    
    axes[1].imshow(rec1)
    axes[1].set_title("Reconstructed 1")
    axes[1].axis("off")
    
    axes[2].imshow(merged_image)
    axes[2].set_title("Merged Output")
    axes[2].axis("off")
    
    axes[3].imshow(rec2)
    axes[3].set_title("Reconstructed 2")
    axes[3].axis("off")
    
    axes[4].imshow(img2)
    axes[4].set_title("Input Image 2")
    axes[4].axis("off")
    
    plt.tight_layout()
    plt.savefig(args.output_image)
    print(f"Merged image saved at: {args.output_image}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

