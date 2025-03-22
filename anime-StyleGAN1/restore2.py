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
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from stylegan import StyleGAN

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--image1", type=str, help="Path to first input image")
parser.add_argument("--image2", type=str, help="Path to second input image")
parser.add_argument("-o", "--output_path", type=str, default="demo/stylemix.png")
parser.add_argument("-n", "--n_z1", type=int, default=1, help="Number of z1 style blocks")
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("-s", "--image_size", dest="image_size", default=64, type=int)

args = parser.parse_args()
IMG_SHAPE = (args.image_size, args.image_size, 3)

def load_and_preprocess(image_path):
    image = Image.open(image_path).convert("RGB").resize((args.image_size, args.image_size))
    image = np.array(image) / 127.5 - 1  # Normalize to [-1, 1]
    return np.expand_dims(image, axis=0)

def latent_inversion(generator, image, latent_dim, steps=500, learning_rate=0.01):
    
    print("\n\n***********************************16*************************************\n\n")
    
    z = tf.Variable(tf.random.normal([1, latent_dim], dtype=tf.float32), trainable=True)  # Fixed shape issue
    print("shape of z : ", z.shape, "\nz:", z)
    optimizer = Adam(learning_rate)
    print("\n\n***********************************17*************************************\n\n")
    
    for step in range(steps):
        print("\n\n***********************************19*************************************\n\n")
        
        with tf.GradientTape() as tape:
            print("\n\n***********************************21*************************************\n\n")
            
            generated = generator([
                tf.ones((1, 1)),
                z,  
                tf.zeros((1, IMG_SHAPE[0], IMG_SHAPE[1]))
            ], training=False)
            print("\n\n***********************************22*************************************\n\n")
            
            loss = tf.reduce_mean(tf.abs(generated - image))
        print("\n\n***********************************20*************************************\n\n")
        
        grads = tape.gradient(loss, [z])
        optimizer.apply_gradients(zip(grads, [z]))
    print("\n\n***********************************18*************************************\n\n")
    
    return z.numpy().reshape(1, latent_dim)  # Ensure correct shape

def generate(generator, n_style_block, image1, image2):
    print("\n\n***********************************7*************************************\n\n")
    
    z1 = latent_inversion(generator, image1, args.latent_dim) #<- error
    print("\n\n***********************************8*************************************\n\n")
    
    z2 = latent_inversion(generator, image2, args.latent_dim)
    print("\n\n***********************************9*************************************\n\n")
    
    print("z1 shape:", z1.shape)
    print("z2 shape:", z2.shape)
    
    print("\n\n***********************************10*************************************\n\n")
    
    noise = np.random.normal(0, 1, [1, IMG_SHAPE[0], IMG_SHAPE[1]])
    print("\n\n***********************************11*************************************\n\n")
    
    n_z1 = args.n_z1
    print("\n\n***********************************12*************************************\n\n")
    
    # Fixed latent code concatenation
    inputs = [
        np.ones((1, 1)),
        np.concatenate((np.tile(z1, (1, n_z1)), np.tile(z2, (1, n_style_block - n_z1))), axis=1),
        noise
    ]
    print("\n\n***********************************13*************************************\n\n")
    
    print("Generated input shape:", inputs[1].shape)
    
    print("\n\n***********************************14*************************************\n\n")
    
    imgs = generator.predict(inputs)
    imgs = (imgs.squeeze(0) + 1) / 2
    
    print("\n\n***********************************15*************************************\n\n")
    
    plt.imshow(imgs)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path)

if __name__ == "__main__":
    print(args)
    print("\n\n***********************************1*************************************\n\n")
    gan = StyleGAN(img_shape=IMG_SHAPE, latent_dim=args.latent_dim)
    print("\n\n***********************************2*************************************\n\n")
    gan.load_weights(args.model_path)
    print("\n\n***********************************3*************************************\n\n")
    
    image1 = load_and_preprocess(args.image1)
    print("\n\n***********************************4*************************************\n\n")
    
    image2 = load_and_preprocess(args.image2)
    print("\n\n***********************************5*************************************\n\n")
    
    
    generate(gan.g, gan.n_style_block, image1, image2)
    print("\n\n***********************************6*************************************\n\n")
    


# +
# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from PIL import Image
# from stylegan import StyleGAN

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str)
# parser.add_argument("--image1", type=str, help="Path to first input image")
# parser.add_argument("--image2", type=str, help="Path to second input image")
# parser.add_argument("-o", "--output_path", type=str, default="demo/stylemix.png")
# parser.add_argument("-n", "--n_z1", type=int, default=1, help="number of z1 style blocks")
# parser.add_argument("--latent_dim", type=int, default=128)
# parser.add_argument("-s", "--image_size", dest="image_size", default=64, type=int)

# args = parser.parse_args()
# IMG_SHAPE = (args.image_size, args.image_size, 3)

# def load_and_preprocess(image_path):
#     image = Image.open(image_path).convert("RGB").resize((args.image_size, args.image_size))
#     image = np.array(image) / 127.5 - 1  # Normalize to [-1, 1]
#     return np.expand_dims(image, axis=0)

# def latent_inversion(generator, image, latent_dim, steps=500, learning_rate=0.01):
#     z = tf.Variable(tf.random.normal([1, 1, latent_dim]), trainable=True)
# #     z = tf.Variable(tf.random.normal([1, latent_dim]), trainable=True)

#     optimizer = Adam(learning_rate)

#     for step in range(steps):
#         with tf.GradientTape() as tape:
#             generated = generator([tf.ones((1, 1)), z, tf.zeros((1, IMG_SHAPE[0], IMG_SHAPE[1]))], training=False)
# #             generated = generator([tf.ones((1, 1)), tf.expand_dims(z, axis=1), tf.zeros((1, IMG_SHAPE[0], IMG_SHAPE[1]))], training=False)
#             loss = tf.reduce_mean(tf.abs(generated - image))
#         grads = tape.gradient(loss, [z])
#         optimizer.apply_gradients(zip(grads, [z]))

#     return z.numpy()

# def generate(generator, n_style_block, image1, image2):
#     z1 = latent_inversion(generator, image1, args.latent_dim)
#     z2 = latent_inversion(generator, image2, args.latent_dim)
#     print("z1 shape:", z1.shape)
#     print("z2 shape:", z2.shape)
    
#     noise = np.random.normal(0, 1, [len(z1), IMG_SHAPE[0], IMG_SHAPE[1]])
    
#     n_z1 = args.n_z1
# #     inputs = [
# #         np.ones((len(z1), 1)),
# #         np.concatenate((z1.repeat(n_z1, axis=1), np.repeat(z2, n_style_block - n_z1, axis=1)), axis=1),
# #         noise,
# #     ]
    

#     inputs = [
#         np.ones((len(z1), 1)),
#         np.concatenate((np.tile(z1, (1, n_z1, 1)), np.tile(z2, (1, n_style_block - n_z1, 1))), axis=1),
#         noise
#     ]
    
#     print("Generated input shape:", inputs[1].shape)

#     imgs = generator.predict(inputs)
    
#     imgs = (imgs.squeeze(0) + 1) / 2
    
#     plt.imshow(imgs)
#     plt.axis("off")
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
#     plt.savefig(args.output_path)

# if __name__ == "__main__":
#     print(args)
#     gan = StyleGAN(img_shape=IMG_SHAPE, latent_dim=args.latent_dim)
#     gan.load_weights(args.model_path)
    
#     image1 = load_and_preprocess(args.image1)
#     image2 = load_and_preprocess(args.image2)
    
#     generate(gan.g, gan.n_style_block, image1, image2)


# +
# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from stylegan import StyleGAN
# from encoder import LatentInverter  # Importing a latent space inversion module
# from PIL import Image

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str)
# parser.add_argument("--image1", type=str, help="Path to first input image")
# parser.add_argument("--image2", type=str, help="Path to second input image")
# parser.add_argument("-o", "--output_path", type=str, default="demo/stylemix.png")
# parser.add_argument("-n", "--n_z1", type=int, default=1, help="number of z1 style blocks")
# parser.add_argument("--latent_dim", type=int, default=128)
# parser.add_argument("-s", "--image_size", dest="image_size", default=64, type=int)

# args = parser.parse_args()
# IMG_SHAPE = (args.image_size, args.image_size, 3)

# def load_and_preprocess(image_path):
#     image = Image.open(image_path).convert("RGB").resize((args.image_size, args.image_size))
#     image = np.array(image) / 127.5 - 1  # Normalize to [-1, 1]
#     return np.expand_dims(image, axis=0)

# def generate(generator, n_style_block, image1, image2):
#     inverter = LatentInverter(model_path=args.model_path)  # Initialize inverter
    
#     # Convert images to latent codes
#     z1 = inverter.invert(image1)
#     z2 = inverter.invert(image2)
    
#     noise = np.random.normal(0, 1, [len(z1), IMG_SHAPE[0], IMG_SHAPE[1]])
    
#     n_z1 = args.n_z1
#     inputs = [
#         np.ones((len(z1), 1)),
#         np.concatenate((z1.repeat(n_z1, axis=1), np.repeat(z2, n_style_block - n_z1, axis=1)), axis=1),
#         noise,
#     ]
#     imgs = generator.predict(inputs)
    
#     imgs = (imgs.squeeze(0) + 1) / 2
    
#     plt.imshow(imgs)
#     plt.axis("off")
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
#     plt.savefig(args.output_path)

# if __name__ == "__main__":
#     print(args)
#     gan = StyleGAN(img_shape=IMG_SHAPE, latent_dim=args.latent_dim)
#     gan.load_weights(args.model_path)
    
#     image1 = load_and_preprocess(args.image1)
#     image2 = load_and_preprocess(args.image2)
    
#     generate(gan.g, gan.n_style_block, image1, image2)

