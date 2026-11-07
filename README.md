# Medical Image Augmentation and Synthesis

## Table of Contents

- [Overview](#overview)
- [Intended Use and Limits](#intended-use-and-limits)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Using the GUI](#using-the-gui)
- [Data Audit and Baseline Tools](#data-audit-and-baseline-tools)
- [Running the Tests](#running-the-tests)
- [Acknowledgements](#acknowledgements)

## Overview

This project generates synthetic 128x128 grayscale images, intended to resemble the brain tumor class of the training data, with an unconditional Generative Adversarial Network (GAN). It is a research codebase. Augmenting a brain tumor image dataset is a possible future use, but nothing in this repository tests whether synthetic images improve any downstream model.

The original code is preserved at git tag `baseline-v0`.

## Intended Use and Limits

- Research use only.
- Images produced by this project are synthetic. They are not real scans and must never be presented as real patient images.
- No clinical use. Nothing here is validated for diagnosis, treatment, screening or any other clinical purpose.
- No claim is made that augmenting a dataset with these images improves downstream models. That has not been tested.
- The GUI and the sample export tools label generated images as synthetic. Keep that label when you share them.

## Technologies Used

- **Programming Language:** Python (TensorFlow 2.13 requires Python 3.8 to 3.11)
- **Libraries and Frameworks:**
  - TensorFlow 2.13 / Keras
  - NumPy
  - Pillow (PIL)
  - Tkinter (ships with Python; no pip package needed)
- **Development machine:** Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB).
- **GPU note:** TensorFlow 2.11 and later has no native-Windows GPU support. With the pinned TensorFlow 2.13 on native Windows, training runs on the CPU (verified: `tf.config.list_physical_devices('GPU')` returns `[]`). GPU use would need Linux or WSL2.

## Dataset

The `data/` folder is **not** part of this repository and is ignored by git. You must supply your own images.

Expected layout:

```
data/
  brain_tumor/        images of the tumor class (used for training)
  non_brain_tumor/    images of the non-tumor class (preprocessed, not used for training)
```

Dataset details (to be filled in by the project owner):

- Source: _TODO_
- License / terms of use: _TODO_
- Modality and acquisition (for example MRI sequence, slice orientation): _TODO_
- Number of images per class: _TODO_ (see the audit tool below)
- Whether the terms allow publishing a model trained on it: _TODO_

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xyntrae/medical_image_synthesis.git
   cd medical_image_synthesis
   ```

2. **Create a Python 3.11 virtual environment and install the dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows; on Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Place your data** in `data/brain_tumor/` and `data/non_brain_tumor/` (see [Dataset](#dataset)).

## Training the Model

1. **Preprocess the data:**
   ```bash
   python data_preprocessing.py
   ```
   Loads every image file in the two folders as 128x128 grayscale, scales to [0, 1] and writes `data/brain_tumor_images.npy` and `data/non_brain_tumor_images.npy`. Subfolders are skipped; unreadable files are reported and skipped. Optional: `--brain-tumor-dir`, `--non-brain-tumor-dir`, `--output-dir`.

2. **Train the GAN:**
   ```bash
   python model_training.py
   ```
   Trains on the brain tumor array with the original settings (50 epochs, batch size 64, Adam with learning rate 1e-4, noise dimension 100) and saves both the generator weights (`generator_w.weights.h5`) and the discriminator weights (`discriminator_w.weights.h5`) in the current directory.

   Optional arguments (defaults reproduce the original behavior): `--epochs`, `--batch-size`, `--seed`, `--output-dir`, `--checkpoint-every N` (save both networks every N epochs into `checkpoints/`), `--sample-every N` (save an 8x8 grid of synthetic samples every N epochs into `samples/`). Run `python model_training.py --help` for details.

## Using the GUI

```bash
python gui_app.py                     # uses generator_w.weights.h5 in the current directory
python gui_app.py --weights path/to/generator_w.weights.h5
```

The window shows one generated image at a time with a permanent label stating that it is synthetic and not a real scan. If the weights file is missing, the app shows an error message instead of crashing.

## Data Audit and Baseline Tools

Tools added in Phase 0 (they never modify input images and make no network calls):

- `tools/audit_data.py`: read-only audit of your image folders. Produces a markdown report with counts and statistics only (formats, sizes, modes, intensity, exact and near duplicates, identifier-like file names, metadata files). See [docs/AUDIT_USAGE.md](docs/AUDIT_USAGE.md), including which parts are safe to share.
- `tools/run_baseline.py`: seeded wrapper that runs preprocessing and training, logs per-epoch losses to CSV, records wall time and GPU memory, and exports synthetic sample images. See [docs/BASELINE_USAGE.md](docs/BASELINE_USAGE.md).
- `tools/make_synthetic_fixture.py`: generates a small, obviously artificial image set for testing the tools without real data.

## Running the Tests

Install pytest in the same environment (`pip install pytest`; it is a test-only tool, not a runtime dependency), then run the whole suite from the repository root with one command:

```bash
python -m pytest
```

This runs `tools/tests/` and `tests/` (configured in `pytest.ini`). The tests use only synthetic data generated on the fly. GUI tests skip automatically when no display is available.

## Acknowledgements

This project utilizes various open-source libraries and frameworks, without which this implementation would not be possible. Special thanks to the TensorFlow and Keras teams for their continued efforts in providing excellent tools for machine learning and deep learning.
