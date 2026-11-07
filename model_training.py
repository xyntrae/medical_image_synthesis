import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import losses
from config import NOISE_DIM
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.data_loader import load_data

GENERATOR_WEIGHTS = 'generator_w.weights.h5'
DISCRIMINATOR_WEIGHTS = 'discriminator_w.weights.h5'
SYNTHETIC_BANNER = "SYNTHETIC - GAN samples, research artifact, not real scans, no clinical validity"


class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=False)

    def compile(self, generator_optimizer, discriminator_optimizer):
        super(GAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}


class PeriodicCheckpoint(keras.callbacks.Callback):
    """Save generator and discriminator weights every `every` epochs (optional, off by default)."""

    def __init__(self, output_dir, every):
        super().__init__()
        self.dir = os.path.join(output_dir, 'checkpoints')
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0:
            os.makedirs(self.dir, exist_ok=True)
            tag = f'epoch_{epoch + 1:04d}'
            self.model.generator.save_weights(os.path.join(self.dir, f'{tag}_{GENERATOR_WEIGHTS}'))
            self.model.discriminator.save_weights(os.path.join(self.dir, f'{tag}_{DISCRIMINATOR_WEIGHTS}'))


class SampleGrid(keras.callbacks.Callback):
    """Save an 8x8 grid of synthetic samples every `every` epochs (optional, off by default).

    The fixed noise comes from a private NumPy generator, so enabling this callback does not
    consume TensorFlow or global NumPy random state and does not change training numerics.
    """

    def __init__(self, output_dir, every, seed=0):
        super().__init__()
        self.dir = os.path.join(output_dir, 'samples')
        self.every = every
        self.noise = np.random.default_rng(seed).standard_normal((64, NOISE_DIM)).astype('float32')

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return
        from PIL import Image, ImageDraw
        images = self.model.generator(self.noise, training=False).numpy()[..., 0]
        images = (np.clip(images, 0, 1) * 255).round().astype(np.uint8)
        h, w = images.shape[1:]
        gap, banner = 2, 16
        grid = Image.new('L', (8 * w + 9 * gap, banner + 8 * h + 9 * gap), 0)
        for k in range(64):
            r, c = divmod(k, 8)
            grid.paste(Image.fromarray(images[k], mode='L'), (gap + c * (w + gap), banner + gap + r * (h + gap)))
        draw = ImageDraw.Draw(grid)
        draw.rectangle([0, 0, grid.width, banner - 1], fill=255)
        draw.text((6, 3), f'{SYNTHETIC_BANNER} (epoch {epoch + 1})', fill=0)
        os.makedirs(self.dir, exist_ok=True)
        grid.save(os.path.join(self.dir, f'synthetic_grid_epoch_{epoch + 1:04d}.png'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the unconditional GAN on data/brain_tumor_images.npy. "
                    "Defaults reproduce the original configuration.")
    parser.add_argument('--epochs', type=int, default=50, help='training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for Python, NumPy and TensorFlow (default: none, as originally)')
    parser.add_argument('--output-dir', default='.',
                        help='folder for the weight files (default: current working directory)')
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='save generator and discriminator weights every N epochs (default: 0, off)')
    parser.add_argument('--sample-every', type=int, default=0,
                        help='save an 8x8 synthetic sample grid every N epochs (default: 0, off)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for name in ('epochs', 'batch_size'):
        if getattr(args, name) < 1:
            raise SystemExit(f'--{name.replace("_", "-")} must be >= 1')
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)

    # Load and preprocess data
    brain_tumor_images, _ = load_data()  # Only use brain tumor images for training

    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile GAN
    gan = GAN(generator, discriminator)
    generator_optimizer = optimizers.Adam(1e-4)
    discriminator_optimizer = optimizers.Adam(1e-4)
    gan.compile(generator_optimizer, discriminator_optimizer)

    callbacks = []
    if args.checkpoint_every > 0:
        callbacks.append(PeriodicCheckpoint(args.output_dir, args.checkpoint_every))
    if args.sample_every > 0:
        callbacks.append(SampleGrid(args.output_dir, args.sample_every,
                                    seed=args.seed if args.seed is not None else 0))
    fit_kwargs = {'callbacks': callbacks} if callbacks else {}

    # Train GAN
    gan.fit(brain_tumor_images, epochs=args.epochs, batch_size=args.batch_size, **fit_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    generator.save_weights(os.path.join(args.output_dir, GENERATOR_WEIGHTS))
    discriminator.save_weights(os.path.join(args.output_dir, DISCRIMINATOR_WEIGHTS))


if __name__ == "__main__":
    main()
