# +
import argparse
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from restore3 import StyleGANWithEncoder, preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to trained StyleGAN weights")
parser.add_argument("--image1", type=str, required=True, help="Path to first input image")
parser.add_argument("--image2", type=str, required=True, help="Path to second input image")
parser.add_argument("--output_image", type=str)
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

    # Merge latent vectors (weighted sum)
    merged_z = args.merge_ratio * z1 + (1 - args.merge_ratio) * z2

    # Generate merged image
    batch_size = tf.shape(merged_z)[0]
    ones = tf.ones((batch_size, 1))
    z_repeated = tf.repeat(tf.expand_dims(merged_z, axis=1), model.n_style_block, axis=1)
    noise = tf.random.normal([batch_size, args.image_size, args.image_size])
    
    merged_image = model.g([ones, z_repeated, noise])

    # Convert and save output image
    merged_image = ((merged_image.numpy()[0] + 1) * 127.5).astype(np.uint8)
    Image.fromarray(merged_image).save(args.output_image)

    print(f"Merged image saved at: {args.output_image}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


# +
# import argparse
# import numpy as np
# import os
# from PIL import Image
# import tensorflow as tf

# +
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, required=True, help="Path to trained StyleGAN weights")
# parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
# parser.add_argument("--output_dir", type=str, default="encoded_results")
# parser.add_argument("--image_size", type=int, default=64)
# parser.add_argument("--latent_dim", type=int, default=128)
# parser.add_argument("--batch_size", type=int, default=16)

# +
# def main(args):
#     # Create model
#     img_shape = (args.image_size, args.image_size, 3)
#     model = StyleGANWithEncoder(img_shape=img_shape, latent_dim=args.latent_dim)
#     model.load_weights(args.model_path)
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Load and preprocess images
#     image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
#     for i in range(0, len(image_files), args.batch_size):
#         batch_files = image_files[i:i + args.batch_size]
#         batch_images = []
        
#         for img_file in batch_files:
#             img_path = os.path.join(args.image_dir, img_file)
#             img = preprocess_image(img_path, (args.image_size, args.image_size))
#             batch_images.append(img)
            
#         batch_images = np.array(batch_images)
        
#         # Encode images to get latent vectors
#         z_vectors = model.encode_images(batch_images)
        
#         # Generate reconstructions
#         reconstructed = model.reconstruct_images(batch_images)
        
#         # Save results
#         for j, (img_file, z, rec) in enumerate(zip(batch_files, z_vectors, reconstructed)):
#             # Save latent vector
#             z_path = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}_z.npy")
#             np.save(z_path, z.numpy())
            
#             # Save reconstruction
#             rec_img = ((rec.numpy() + 1) * 127.5).astype(np.uint8)
#             rec_path = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}_reconstruction.png")
#             Image.fromarray(rec_img).save(rec_path)
            
#         print(f"Processed {i + len(batch_files)}/{len(image_files)} images")

# +
# if __name__ == "__main__":
#     args = parser.parse_args()
#     main(args)
