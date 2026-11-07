import tensorflow as tf
from tensorflow import keras
from keras import layers
from config import NOISE_DIM

def build_generator(noise_dim=NOISE_DIM):
    model = tf.keras.Sequential([
        layers.Dense(256 * 8 * 8, activation='relu', input_shape=(noise_dim,)),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(1, (4, 4), strides=2, padding='same', activation='sigmoid'),  # Use sigmoid activation for grayscale output
    ])
    return model