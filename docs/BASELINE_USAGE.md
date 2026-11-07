# Running the frozen baseline with tools/run_baseline.py

`tools/run_baseline.py` runs the existing baseline exactly as committed at
tag `baseline-v0` (`data_preprocessing.py`, then `model_training.py`) without
editing any of its files, and adds seeding, logging and sample export around
it. All images it exports are synthetic research artifacts, not real scans,
with no clinical validity.

## What the baseline does (unchanged)

- Loads every file in `data/brain_tumor/` and `data/non_brain_tumor/` as
  128x128 grayscale, scales to [0, 1], saves `data/*.npy`.
- Trains an unconditional GAN on the brain_tumor array only: 50 epochs,
  batch 64, Adam 1e-4 for both networks, noise dimension 100.
- Saves only the generator weights, `generator_w.weights.h5`.

## What the wrapper adds

- Seeds: `PYTHONHASHSEED`, Python `random`, NumPy, TensorFlow
  (`tf.keras.utils.set_random_seed`), plus
  `tf.config.experimental.enable_op_determinism()`, `TF_DETERMINISTIC_OPS=1`,
  `TF_CUDNN_DETERMINISTIC=1` and `TF_ENABLE_ONEDNN_OPTS=0` (opt out with
  `--keep-onednn`). Default seed 42. Each stage runs in a child process that
  sets these first and then runs the original script with `runpy`.
- Data location without editing the scripts: it creates
  `<output>/work/data/brain_tumor` and `<output>/work/data/non_brain_tumor` as
  directory links (a junction on Windows, a symlink on Linux/WSL) to the
  folders in `--data-dir`, and runs the scripts with `<output>/work` as the
  working directory. The `.npy` arrays and the weights land in
  `<output>/work`. Nothing is written into `--data-dir`. The links are
  removed when the run ends, even on failure.
- `stdout.log`: full combined output of all stages.
- `losses.csv`: per epoch, the last `gen_loss` and `disc_loss` that Keras
  printed for that epoch (Keras shows the running mean over the epoch's
  batches), and the epoch time Keras printed.
- `run_summary.json`: run type, seed, flags, library versions and devices
  seen by TensorFlow, op-determinism status, wall time per stage and total,
  array shapes after preprocessing (counts only), GPU memory (see below),
  git HEAD, whether any baseline file is modified, and SHA-256 of each
  baseline file.
- GPU memory: if `nvidia-smi` exists, it is polled once per second.
  `memory.used` is whole-device usage including other programs, so the
  summary reports the value before the run, the peak during the run, and
  the difference. The wrapper sets `TF_FORCE_GPU_ALLOW_GROWTH=true` so that
  TensorFlow does not reserve almost all GPU memory up front (disable with
  `--no-memory-growth`).
- Samples: after training, the generator is rebuilt with
  `models/generator.py`, the saved weights are loaded, and samples are drawn
  from seeded noise:
  - `samples/SYNTHETIC_grid_8x8.png`: 64 samples with a banner reading
    SYNTHETIC, the run type and the seed.
  - `samples/synthetic_sample_XXXX.png`: `--num-samples` individual images
    (default 64). Pixels are left unaltered so later metrics are not biased
    by a stamp; each file is named `synthetic_...` and carries PNG text
    fields `Synthetic=true`, a description, the run type and the seed.
  - `SYNTHETIC_OUTPUTS_README.txt` in the output folder.
- `--smoke-epochs N` replaces the hard-coded `epochs=50` at runtime by
  wrapping `tf.keras.Model.fit` in the child process. Use it only for smoke
  tests. Such runs are labelled `smoke_test` in the summary, the README and
  the grid banner, and say nothing about model quality.

## Environment

TensorFlow 2.13 (as pinned) supports Python 3.8 to 3.11.

```powershell
py -3.11 -m venv $HOME\.venvs\mis_baseline_py311
& $HOME\.venvs\mis_baseline_py311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

On native Windows, TensorFlow 2.13 has no GPU support (native Windows GPU
support ended with TF 2.10), so the last command will list no GPU and
training runs on the CPU. Your choices for the full 50-epoch run:

1. **CPU on native Windows.** Works with the venv above. Slow but simplest.
   Add `--cpu` to make the choice explicit.
2. **GPU through WSL2.** Needs an Ubuntu (or similar) WSL2 distribution, the
   Windows NVIDIA driver (which exposes the GPU to WSL2), and inside WSL a
   Python 3.11 environment with `tensorflow==2.13.0` plus the CUDA 11.8 and
   cuDNN 8.6 libraries that TF 2.13 was built against. These are large
   downloads and system changes that you would do yourself; the tools here
   do not install them. Inside WSL, Windows drives appear under `/mnt/`, for
   example `/mnt/d/private`. Run the same commands below with Linux paths.

## Smoke test on the synthetic fixture (no real data)

```powershell
python tools/make_synthetic_fixture.py --overwrite
python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run1 --smoke-epochs 2 --seed 42
```

To check that seeding makes runs repeatable, run it a second time into a new
folder and compare the loss logs:

```powershell
python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run2 --smoke-epochs 2 --seed 42
fc.exe outputs\smoke_run1\losses.csv outputs\smoke_run2\losses.csv
```

The fixture contains an unreadable file, a text file and an RGB image on
purpose. The unchanged `data_preprocessing.py` prints an error line for each
file it cannot load and continues; that is expected.

## Full baseline on your private data

`--data-dir` must contain `brain_tumor/` and `non_brain_tumor/`.
`--output-dir` must be new or empty and outside `--data-dir`.

```powershell
python tools/run_baseline.py --data-dir "D:\private" --output-dir "D:\private_runs\baseline_v0_seed42" --seed 42
```

Add `--cpu` on native Windows, `--delete-intermediate` to delete the `.npy`
array copies of your data from `<output>/work/data` after a successful run,
and `--quiet` to stop echoing training output to the console.

## What is safe to share back

Generally safe, after you have looked at them:

- `losses.csv` (numbers only).
- `run_summary.json`. It contains the last component of `--data-dir`, the
  path of the Python interpreter you used, array shapes (image counts) and
  library versions. Remove anything you consider sensitive.

Keep private:

- `work/` (contains `.npy` arrays of your images and the trained weights,
  which were trained on your data).
- `stdout.log` (the unchanged preprocessing script prints the file name of
  every file it cannot load).
- Sample images. They are synthetic, but they come from a model trained on
  your private data and can reveal properties of it. Share them only if you
  decide to, and always as synthetic material.
