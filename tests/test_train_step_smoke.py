"""Training smoke tests on random data (pipeline checks only, not model quality)."""
import os

import numpy as np
import pytest
import tensorflow as tf
from keras import optimizers

import model_training
from model_training import GAN
from models.discriminator import build_discriminator
from models.generator import build_generator


def _gan(cls=GAN):
    gan = cls(build_generator(), build_discriminator())
    gan.compile(optimizers.Adam(1e-4), optimizers.Adam(1e-4))
    return gan


class RecordingGAN(GAN):
    """Records what fit() passes to train_step, then runs the unchanged baseline step."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.traced = []       # (python type, static shape, dtype) at trace time
        self.batch_sizes = []  # dynamic batch size of every step

    def train_step(self, data):
        self.traced.append((type(data), tuple(data.shape), data.dtype))
        tf.py_function(lambda n: self.batch_sizes.append(int(n)), [tf.shape(data)[0]], [])
        return super().train_step(data)


def test_two_train_steps_give_finite_losses():
    tf.keras.utils.set_random_seed(0)
    gan = _gan()
    x = np.random.default_rng(0).random((8, 128, 128, 1), dtype=np.float32)
    for _ in range(2):
        logs = gan.train_step(tf.constant(x[:4]))
        assert np.isfinite(float(logs["gen_loss"])) and np.isfinite(float(logs["disc_loss"]))


def test_fit_passes_plain_float32_batch_tensor():
    """Answers the open question: under fit() with a NumPy array, train_step receives a single
    float32 tensor of shape (batch, 128, 128, 1), not a tuple."""
    tf.keras.utils.set_random_seed(0)
    gan = _gan(RecordingGAN)
    x = np.random.default_rng(1).random((10, 128, 128, 1), dtype=np.float32)
    hist = gan.fit(x, epochs=1, batch_size=4, verbose=0, shuffle=False)
    assert gan.traced, "train_step was never traced"
    for typ, shape, dtype in gan.traced:
        assert issubclass(typ, tf.Tensor)
        assert shape[1:] == (128, 128, 1) and len(shape) == 4
        assert dtype == tf.float32
    assert gan.batch_sizes == [4, 4, 2]
    assert all(np.isfinite(v[0]) for v in hist.history.values())


def test_cli_defaults_reproduce_original_configuration():
    a = model_training.parse_args([])
    assert (a.epochs, a.batch_size, a.seed, a.output_dir, a.checkpoint_every, a.sample_every) == \
        (50, 64, None, ".", 0, 0)
    assert model_training.GENERATOR_WEIGHTS == "generator_w.weights.h5"


def _run_main(tmp_path, monkeypatch, extra):
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    rng = np.random.default_rng(2)
    np.save(work / "data" / "brain_tumor_images.npy", rng.random((6, 128, 128, 1), dtype=np.float32))
    np.save(work / "data" / "non_brain_tumor_images.npy", rng.random((2, 128, 128, 1), dtype=np.float32))
    monkeypatch.chdir(work)
    out = tmp_path / "out"
    model_training.main(["--epochs", "2", "--batch-size", "4", "--seed", "3", "--output-dir", str(out)] + extra)
    return out


def test_main_saves_both_weight_files(tmp_path, monkeypatch):
    out = _run_main(tmp_path, monkeypatch, [])
    assert sorted(p.name for p in out.iterdir()) == ["discriminator_w.weights.h5", "generator_w.weights.h5"]


def test_optional_callbacks_write_checkpoints_and_synthetic_grids(tmp_path, monkeypatch):
    out = _run_main(tmp_path, monkeypatch, ["--checkpoint-every", "1", "--sample-every", "2"])
    ck = sorted(p.name for p in (out / "checkpoints").iterdir())
    assert ck == ["epoch_0001_discriminator_w.weights.h5", "epoch_0001_generator_w.weights.h5",
                  "epoch_0002_discriminator_w.weights.h5", "epoch_0002_generator_w.weights.h5"]
    grids = sorted((out / "samples").iterdir())
    assert [p.name for p in grids] == ["synthetic_grid_epoch_0002.png"]
    from PIL import Image
    im = Image.open(grids[0])
    assert im.mode == "L" and im.size == (8 * 128 + 9 * 2, 16 + 8 * 128 + 9 * 2)


def test_callbacks_do_not_change_training_numerics(tmp_path, monkeypatch):
    a = _run_main(tmp_path / "a", monkeypatch, [])
    b = _run_main(tmp_path / "b", monkeypatch, ["--checkpoint-every", "1", "--sample-every", "1"])
    ga, gb = build_generator(), build_generator()
    ga.load_weights(str(a / "generator_w.weights.h5"))
    gb.load_weights(str(b / "generator_w.weights.h5"))
    for wa, wb in zip(ga.get_weights(), gb.get_weights()):
        assert np.array_equal(wa, wb)


def test_invalid_epochs_rejected():
    with pytest.raises(SystemExit):
        model_training.main(["--epochs", "0"])
