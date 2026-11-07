"""Architecture checks: output shapes, ranges and parameter counts (unchanged from baseline-v0)."""
import numpy as np
import tensorflow as tf

from config import NOISE_DIM
from models.discriminator import build_discriminator
from models.generator import build_generator


def test_noise_dim_constant():
    assert NOISE_DIM == 100
    assert build_generator().input_shape == (None, 100)


def test_generator_output_shape_and_range():
    g = build_generator()
    out = g(tf.random.normal([4, NOISE_DIM], seed=1), training=False).numpy()
    assert out.shape == (4, 128, 128, 1)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_discriminator_output_shape_and_range():
    d = build_discriminator()
    out = d(np.random.default_rng(0).random((3, 128, 128, 1), dtype=np.float32), training=False).numpy()
    assert out.shape == (3, 1)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_parameter_counts():
    assert build_generator().count_params() == 2_343_649
    assert build_discriminator().count_params() == 722_369
