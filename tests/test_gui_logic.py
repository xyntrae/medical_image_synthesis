"""GUI logic tests. Tk-dependent tests skip cleanly when no display is available."""
import numpy as np
import pytest

import gui_app


def test_to_uint8_image_matches_original_formula():
    rng = np.random.default_rng(0)
    gen = rng.random((1, 128, 128, 1), dtype=np.float32)
    expected = (gen.reshape(128, 128) * 255).astype(np.uint8)  # formula used at baseline-v0
    out = gui_app.to_uint8_image(gen)
    assert out.dtype == np.uint8 and out.shape == (128, 128)
    assert np.array_equal(out, expected)


def test_to_uint8_image_rejects_wrong_size():
    with pytest.raises(ValueError):
        gui_app.to_uint8_image(np.zeros((1, 64, 64, 1), np.float32))


def test_load_generator_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        gui_app.load_generator(str(tmp_path / "missing.weights.h5"))


def test_label_text_says_synthetic():
    assert "SYNTHETIC" in gui_app.SYNTHETIC_LABEL and "Not a real scan" in gui_app.SYNTHETIC_LABEL


@pytest.fixture(scope="module")
def _tk_session():
    # One Tk interpreter per module: creating several Tk() roots in one process can fail
    # intermittently on Windows during Tcl initialisation.
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def tk_root(_tk_session):
    import tkinter as tk
    top = tk.Toplevel(_tk_session)
    top.withdraw()
    yield top
    top.destroy()


def test_missing_weights_shows_message_and_disables_button(tk_root, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(gui_app.messagebox, "showerror", lambda *a, **k: shown.append(a))
    app = gui_app.ImageGeneratorApp(tk_root, weights_path=str(tmp_path / "missing.weights.h5"))
    assert len(shown) == 1 and "not found" in shown[0][1]
    assert str(app.generate_button["state"]) == "disabled"
    assert "SYNTHETIC" in app.synthetic_label["text"]
    app.generate_images()  # must be a no-op, not a crash
    assert app.tk_img is None


class _FakeGenerator:
    def __call__(self, noise, training=False):
        import tensorflow as tf
        assert tuple(noise.shape) == (1, 100) and training is False
        return tf.fill([1, 128, 128, 1], 0.5)


def test_generate_keeps_persistent_image_and_does_not_block(tk_root, monkeypatch):
    monkeypatch.setattr(gui_app.messagebox, "showerror", lambda *a, **k: pytest.fail("unexpected error"))
    app = gui_app.ImageGeneratorApp(tk_root, generator_loader=lambda path: _FakeGenerator())
    assert str(app.generate_button["state"]) == "normal"
    app.generate_images()  # returns (no nested mainloop)
    first = app.tk_img
    assert first is not None and first.width() == 128 and first.height() == 128
    app.generate_images()
    assert app.tk_img is not first
    assert len(app.canvas.find_all()) == 1
