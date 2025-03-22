import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import InstanceNormalization
from stylegan import StyleGAN  # Ensure this points to the correct file where StyleGAN is defined


class StyleGANEncoder(keras.Model):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder architecture
        self.encoder = keras.Sequential([
            # Initial conv layer
            keras.layers.Conv2D(64, 3, strides=2, padding='same'),
            InstanceNormalization(),
            keras.layers.LeakyReLU(0.2),
            
            # Downsampling blocks
            keras.layers.Conv2D(128, 3, strides=2, padding='same'),
            InstanceNormalization(),
            keras.layers.LeakyReLU(0.2),
            
            keras.layers.Conv2D(256, 3, strides=2, padding='same'),
            InstanceNormalization(),
            keras.layers.LeakyReLU(0.2),
            
            keras.layers.Conv2D(512, 3, strides=2, padding='same'),
            InstanceNormalization(),
            keras.layers.LeakyReLU(0.2),
            
            # Flatten and project to latent space
            keras.layers.Flatten(),
            keras.layers.Dense(latent_dim * 2),  # Output mean and log variance
        ])
        
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
        return mean, logvar
        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
        
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

def preprocess_image(image_path, target_size=(64, 64)):
    """Load and preprocess image for encoder"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]
    return img

class StyleGANWithEncoder(StyleGAN):
    def __init__(self, img_shape, latent_dim, **kwargs):
        super().__init__(img_shape, latent_dim, **kwargs)
        self.encoder = StyleGANEncoder(latent_dim)
        
    def encode_images(self, images):
        """Encode images to latent vectors"""
        z, mean, _ = self.encoder(images)
        return z
        
    def reconstruct_images(self, images):
        """Reconstruct images through encode-decode process"""
        z = self.encode_images(images)
        # Prepare inputs for generator
        batch_size = tf.shape(z)[0]
        ones = tf.ones((batch_size, 1))
        z_repeated = tf.repeat(tf.expand_dims(z, axis=1), self.n_style_block, axis=1)
        noise = tf.random.normal([batch_size, self.img_shape[0], self.img_shape[1]])
        
        reconstructed = self.g([ones, z_repeated, noise])
        return reconstructed
        
    def compile_encoder(self, learning_rate=0.0001):
        """Compile encoder with reconstruction loss"""
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        
    @tf.function
    def train_encoder_step(self, real_images):
        """Single training step for encoder"""
        with tf.GradientTape() as tape:
            # Encode images
            z, mean, logvar = self.encoder(real_images)
            
            # Generate reconstructed images
            batch_size = tf.shape(z)[0]
            ones = tf.ones((batch_size, 1))
            z_repeated = tf.repeat(tf.expand_dims(z, axis=1), self.n_style_block, axis=1)
            noise = tf.random.normal([batch_size, self.img_shape[0], self.img_shape[1]])
            
            reconstructed = self.g([ones, z_repeated, noise])
            
            # Compute reconstruction loss
            recon_loss = tf.reduce_mean(tf.abs(real_images - reconstructed))
            
            # Add KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
            
            # Total loss
            total_loss = recon_loss + 0.1 * kl_loss
            
        # Update encoder
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        
        return total_loss, recon_loss, kl_loss
