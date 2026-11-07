"""Preprocessing tests on the synthetic fixture (procedural images, not real data)."""
import hashlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import REPO

import data_preprocessing
from utils.image_utils import load_images_from_folder


def _baseline_script(tmp_path):
    """Extract data_preprocessing.py as it was at tag baseline-v0."""
    proc = subprocess.run(["git", "-C", str(REPO), "show", "baseline-v0:data_preprocessing.py"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("git tag baseline-v0 not available")
    path = tmp_path / "baseline_data_preprocessing.py"
    path.write_text(proc.stdout, encoding="utf-8")
    return path


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_new_script_matches_baseline_v0(synthetic_fixture, tmp_path):
    base_cwd = tmp_path / "base"
    shutil.copytree(synthetic_fixture, base_cwd / "data")
    script = _baseline_script(tmp_path)
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO)}
    p = subprocess.run([sys.executable, "-B", str(script)], cwd=base_cwd, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    new_out = tmp_path / "new"
    data_preprocessing.main(["--brain-tumor-dir", str(synthetic_fixture / "brain_tumor"),
                             "--non-brain-tumor-dir", str(synthetic_fixture / "non_brain_tumor"),
                             "--output-dir", str(new_out)])
    for name in ("brain_tumor_images.npy", "non_brain_tumor_images.npy"):
        a, b = np.load(base_cwd / "data" / name), np.load(new_out / name)
        assert a.dtype == b.dtype == np.float32
        assert np.array_equal(a, b)
        assert _sha(base_cwd / "data" / name) == _sha(new_out / name)


def test_default_paths_unchanged():
    args = data_preprocessing.parse_args([])
    assert args.brain_tumor_dir.replace("\\", "/") == "data/brain_tumor"
    assert args.non_brain_tumor_dir.replace("\\", "/") == "data/non_brain_tumor"
    assert args.output_dir == "data"


def test_subdirectories_ignored_cleanly(tmp_path, capsys):
    d = tmp_path / "imgs"
    (d / "nested").mkdir(parents=True)
    Image.new("L", (40, 30), 100).save(d / "a.png")
    Image.new("L", (128, 128), 200).save(d / "nested" / "b.png")
    arr = load_images_from_folder(str(d))
    out = capsys.readouterr().out
    assert arr.shape == (1, 128, 128, 1)
    assert "Error loading" not in out


def test_unreadable_files_reported(synthetic_fixture, capsys):
    arr = load_images_from_folder(str(synthetic_fixture / "brain_tumor"))
    out = capsys.readouterr().out
    assert "Error loading image" in out and "corrupt_0000.png" in out
    files = list((synthetic_fixture / "brain_tumor").iterdir())
    assert arr.shape == (len(files) - 1, 128, 128, 1)  # only the corrupt file is skipped
