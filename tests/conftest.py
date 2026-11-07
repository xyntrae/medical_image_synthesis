"""Shared test setup: repository root on sys.path, quiet TensorFlow, synthetic fixture."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # tests run on CPU


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def synthetic_fixture(tmp_path_factory):
    """Small synthetic procedural fixture (not real data)."""
    fixgen = load_module("make_synthetic_fixture_t", REPO / "tools" / "make_synthetic_fixture.py")
    out = tmp_path_factory.mktemp("fixture") / "synthetic_raw"
    fixgen.generate(out, per_class=20, seed=1234)
    return out
