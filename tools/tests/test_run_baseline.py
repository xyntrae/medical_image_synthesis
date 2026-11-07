"""TensorFlow-free tests for tools/run_baseline.py (parsing, validation, links)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
RUNNER = TOOLS / "run_baseline.py"

spec = importlib.util.spec_from_file_location("run_baseline", RUNNER)
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

KERAS_TTY = (
    "Epoch 1/2\r\n"
    "1/5 [=====>........................] - ETA: 10s - gen_loss: 0.7000 - disc_loss: 1.4000\r"
    "5/5 [==============================] - 3s 400ms/step - gen_loss: 0.6931 - disc_loss: 1.3863\n"
    "Epoch 2/2\n"
    "1/5 [=====>........................] - ETA: 1s - gen_loss: 0.9000 - disc_loss: 1.1000\r"
    "5/5 [==============================] - 2s 401ms/step - gen_loss: 0.8123 - disc_loss: 1.2234\n"
)
KERAS_PLAIN = (
    "Epoch 1/3\n5/5 [==============================] - 12s 2s/step - gen_loss: 1.5e-01 - disc_loss: 2.0\n"
    "Epoch 2/3\n5/5 [==============================] - 1s 200ms/step - gen_loss: nan - disc_loss: 1.0\n"
)


def test_parse_keeps_last_value_per_epoch():
    rows = rb.parse_keras_log(KERAS_TTY)
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[0]["gen_loss"] == 0.6931 and rows[0]["disc_loss"] == 1.3863
    assert rows[1]["gen_loss"] == 0.8123 and rows[1]["disc_loss"] == 1.2234
    assert rows[0]["total_epochs"] == 2
    assert rows[0]["epoch_time"] == "3s"


def test_parse_scientific_nan_and_ignores_preamble():
    rows = rb.parse_keras_log("gen_loss: 9.9 noise before training\n" + KERAS_PLAIN)
    assert len(rows) == 2
    assert rows[0]["gen_loss"] == 0.15 and rows[0]["epoch_time"] == "12s"
    assert rows[1]["gen_loss"] != rows[1]["gen_loss"]  # nan


def test_help():
    p = subprocess.run([sys.executable, "-B", str(RUNNER), "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "--smoke-epochs" in p.stdout and "--data-dir" in p.stdout
    assert "--child-stage" not in p.stdout  # internal option hidden


def test_rejects_missing_class_folder(tmp_path):
    (tmp_path / "d" / "brain_tumor").mkdir(parents=True)
    p = subprocess.run([sys.executable, "-B", str(RUNNER), "--data-dir", str(tmp_path / "d"),
                        "--output-dir", str(tmp_path / "o")], capture_output=True, text=True)
    assert p.returncode == 2 and "missing class folder" in p.stderr
    assert not (tmp_path / "o").exists()


def test_rejects_output_inside_data_and_non_empty_output(tmp_path):
    d = tmp_path / "d"
    for c in rb.CLASSES:
        (d / c).mkdir(parents=True)
    p = subprocess.run([sys.executable, "-B", str(RUNNER), "--data-dir", str(d),
                        "--output-dir", str(d / "out")], capture_output=True, text=True)
    assert p.returncode == 2 and "inside --data-dir" in p.stderr
    o = tmp_path / "o"
    o.mkdir()
    (o / "x").write_text("x")
    p = subprocess.run([sys.executable, "-B", str(RUNNER), "--data-dir", str(d),
                        "--output-dir", str(o)], capture_output=True, text=True)
    assert p.returncode == 2 and "new or empty" in p.stderr


def test_dir_link_roundtrip_leaves_target_intact(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    link = tmp_path / "link"
    rb._make_dir_link(target, link)
    assert (link / "keep.txt").read_text() == "keep"
    assert rb._remove_dir_link(link) is True
    assert not link.exists()
    assert (target / "keep.txt").read_text() == "keep"
    assert rb._remove_dir_link(target) is False  # real directories are never removed
    assert target.is_dir()


def test_orchestration_with_fake_stages(tmp_path, monkeypatch):
    """Exercise the parent pipeline end to end with TensorFlow replaced by fakes."""
    import json
    import numpy as np

    data = tmp_path / "data_src"
    for c in rb.CLASSES:
        (data / c).mkdir(parents=True)
        (data / c / "a.txt").write_text(c)
    before = {p: p.read_text() for p in data.rglob("*.txt")}
    out = tmp_path / "out"
    seen = []

    def fake_stage(name, cmd, cwd, env, log_fh, echo):
        seen.append((name, cmd, env))
        assert (cwd / "data" / "brain_tumor" / "a.txt").read_text() == "brain_tumor"
        env_line = rb.ENV_MARK + json.dumps({"stage": name, "tensorflow": "fake"})
        if name == "preprocess":
            np.save(cwd / "data" / "brain_tumor_images.npy", np.zeros((3, 128, 128, 1)))
            np.save(cwd / "data" / "non_brain_tumor_images.npy", np.zeros((2, 128, 128, 1)))
            text = env_line + "\n"
        elif name == "train":
            (cwd / rb.WEIGHTS_NAME).write_bytes(b"fake")
            text = env_line + "\n" + KERAS_TTY
        else:
            text = env_line + "\n"
        log_fh.write(text)
        return 0, 0.01, text

    monkeypatch.setattr(rb, "_run_stage", fake_stage)
    rc = rb.main(["--data-dir", str(data), "--output-dir", str(out), "--seed", "7", "--smoke-epochs", "2",
                  "--quiet", "--delete-intermediate"])
    assert rc == 0
    assert [s[0] for s in seen] == ["preprocess", "train", "export"]
    env = seen[1][2]
    assert env["PYTHONHASHSEED"] == "7" and env["TF_DETERMINISTIC_OPS"] == "1"
    assert env["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert "--smoke-epochs" in seen[1][1] and seen[1][1][seen[1][1].index("--seed") + 1] == "7"
    s = json.loads((out / "run_summary.json").read_text())
    assert s["run_type"] == "smoke_test" and s["epochs_parsed"] == 2
    assert s["array_shapes"] == {"brain_tumor_images.npy": [3, 128, 128, 1],
                                 "non_brain_tumor_images.npy": [2, 128, 128, 1]}
    assert s["final_losses"]["gen_loss"] == 0.8123
    assert s["intermediate_npy_deleted"] is True and not list((out / "work" / "data").glob("*.npy"))
    lines = (out / "losses.csv").read_text().splitlines()
    assert lines[0] == "epoch,total_epochs,gen_loss,disc_loss,epoch_time" and len(lines) == 3
    assert (out / "SYNTHETIC_OUTPUTS_README.txt").exists()
    # links removed, source data untouched
    assert sorted(p.name for p in (out / "work" / "data").iterdir()) == []
    assert {p: p.read_text() for p in data.rglob("*.txt")} == before
