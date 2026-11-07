# Phase 1 report: Phase 0 gap closure (Part A) and hygiene and bug fixes (Part B)

Date: 2026-11-07. Branch: main. No real data was accessed. Every
measurement in this report comes from synthetic fixture data or random
arrays and is a pipeline smoke test only. None of it is evidence about
model quality on real data.

## 1. Summary

- Starting commit: `26f8c9a5c6cb534b119c96f307735e445b6f2251` (equal to
  `origin/main`). Tag `baseline-v0` still points at `e46df81`, unchanged.
- Part A: TensorFlow 2.13.0 on native Windows sees no GPU (and is not built
  with CUDA). Two identical CPU smoke runs of the baseline runner gave
  identical losses, byte-identical generator weights and pixel-identical
  samples. Sample export and the reduced-epoch option work. No runner fix was
  needed for Part A. Results appended to `docs/reports/phase0_report.md`.
- Part B: all listed defects re-confirmed and fixed with defaults that keep
  the original behavior. Model architecture, loss, optimizer and
  preprocessing behavior are unchanged.
- Behavior preservation: a runner smoke run after all Phase 1 changes
  matches the Part A run exactly (losses, generator weights SHA-256, all 65
  sample PNGs, preprocessed arrays).
- Tests: 45 passed (24 Phase 0 tool tests + 21 new), run with one command,
  `python -m pytest`. py_compile passes on all 18 tracked .py files.
- One small runner change was required by the new CLIs (section 6, B3).

Environment for all runs: Windows 11, venv
`C:\Users\iamab\.venvs\mis_baseline_py311`, Python 3.11.7, tensorflow
2.13.0 (tensorflow-intel 2.13.0), keras 2.13.1, numpy 1.24.3, Pillow 9.4.0,
pytest 9.2.0, CPU only. Runner seed 42, fixture seed 1234.

## 2. Repository inspection

```
$ git status
On branch main
Your branch is up to date with 'origin/main'.
Untracked files:
	docs/medical_image_synthesis_modernization_plan.md
$ git log -8 --oneline
26f8c9a docs: add audit and baseline usage guides and phase 0 report
83c500c feat(tools): add seeded baseline runner
27d3168 feat(tools): add read-only dataset audit
77a1449 feat(tools): add synthetic fixture generator
6305dec chore: add gitignore for data, fixtures and artifacts
e46df81 Update README.md
1774868 Medical Image Augmentation and Synthesis
e1ed8f7 Medical Image Augmentation and Synthesis
$ git ls-remote origin
26f8c9a5c6cb534b119c96f307735e445b6f2251	HEAD
2257a69fa0bb3492e7a8bda577e65553b1fef53e	refs/heads/feature_improvment_gan
26f8c9a5c6cb534b119c96f307735e445b6f2251	refs/heads/main
e46df816eb70f63f65cb1c87a76a6d8150403c8b	refs/tags/baseline-v0
```

Tracked files matched Phase 0. The plan document is still untracked and was
not touched. All files changed in this task were read first, and each known
defect was confirmed in the code before it was fixed.

## 3. Part A: TensorFlow checks (synthetic smoke tests)

### A1. GPU visibility

```
$ python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
2.13.0 []
$ python -c "import tensorflow as tf; print('built_with_cuda', tf.test.is_built_with_cuda())"
built_with_cuda False
```

TensorFlow does **not** see a GPU on native Windows.

### A2. Fixture

```
$ python tools/make_synthetic_fixture.py --overwrite
Synthetic fixture written to fixtures\synthetic_raw: brain_tumor=159 files, non_brain_tumor=153 files
```

### A3. Two runs

```
$ python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run1 --smoke-epochs 2 --seed 42 --cpu
... (trimmed)
[run_baseline] SMOKE TEST: overriding epochs 50 -> 2
3/3 [==============================] - 5s 736ms/step - gen_loss: 0.7448 - disc_loss: 1.3567
3/3 [==============================] - 3s 752ms/step - gen_loss: 0.8537 - disc_loss: 1.2861
[run_baseline] smoke_test finished with return code 0; total wall time 23.831 s; 2 epochs parsed
$ python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run2 --smoke-epochs 2 --seed 42 --cpu --quiet
[run_baseline] smoke_test finished with return code 0; total wall time 19.606 s; 2 epochs parsed
```

Both succeeded on the first attempt; no fix to `tools/run_baseline.py` was
needed for Part A, so the optional commit
"test(tools): fix baseline runner issues found by TensorFlow smoke test" was
not made.

### A4. Programmatic loss comparison

Python script (csv module, absolute float difference per value):

```
epoch 1 gen_loss: 0.7448 vs 0.7448 diff=0.0
epoch 1 disc_loss: 1.3567 vs 1.3567 diff=0.0
epoch 2 gen_loss: 0.8537 vs 0.8537 diff=0.0
epoch 2 disc_loss: 1.2861 vs 1.2861 diff=0.0
losses identical (string): True
losses.csv sha256 equal: False
weights sha256 equal: True
sample pixels identical across runs: True
```

The CSV files differ only in the `epoch_time` column (5s/3s vs 4s/2s,
wall-clock). Keras prints losses to 4 decimals only; the byte-identical
generator weights and pixel-identical samples are the stronger evidence.
Conclusion, for this setting only (CPU, seed 42, 2 epochs, this fixture):
bitwise reproducible. Not tested on GPU or for 50 epochs.

### A5. Sample export and summary

Each run: 64 `synthetic_sample_XXXX.png` (all PNG, 128x128, mode L, PNG text
`Synthetic=true`) plus `SYNTHETIC_grid_8x8.png` (1042x1070, mode L,
`Synthetic=true`, `RunType=smoke_test`). `run_summary.json`: run_type
smoke_test, smoke_epochs 2, epochs_parsed 2, returncode 0.
`SYNTHETIC_OUTPUTS_README.txt` present. Reduced-epoch option works (log
line `overriding epochs 50 -> 2`, Keras `Epoch 1/2`, CSV `total_epochs` 2).

### A6. origin/feature_improvment_gan (read-only)

`git fetch origin`; `git log main..origin/feature_improvment_gan`: one commit,
`2257a69` (2024-05-20, "Added new data agumentation fid is calculator"),
forked from `1774868`. `git diff --stat main...origin/feature_improvment_gan`:
8 files, +151 / -35. Nothing was checked out, merged, cherry-picked or run.

Summary (15 lines):

1. Adds `fid.py`, `inception_score.py`, `data_agumentation.py`, `visualization.py`; rewrites the GAN as a class-conditional GAN (label embeddings in generator and discriminator, Adam beta_1 0.5), loads raw images in `utils/data_loader.py`.
2. FID: Keras `InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))`, 2048-d pooled features, distance from `tensorflow_gan.eval.frechet_classifier_distance_from_activations`.
3. Weights: InceptionV3 uses the default `weights='imagenet'`, so it would download the ImageNet weights (roughly 88 MB, from public knowledge, not measured) from the internet at run time.
4. Preprocessing: `tf.image.resize` to 299x299 then `inception_v3.preprocess_input`; images are 1-channel in [0, 1], but InceptionV3 needs 3 channels and `preprocess_input` expects [0, 255]. As written it would fail on the channel count, and the scaling would be wrong even if channels were repeated.
5. Inception score: passes generated **images** to `tfgan.eval.classifier_score_from_logits`, which expects classifier logits; no classifier is run, so it does not compute an inception score.
6. `tensorflow_gan` is not in requirements.txt and is no longer maintained for current TF.
7. FID is computed against all real images (both classes) while the generator is conditioned, and on full-dataset batches in one call (memory risk).
8. Augmentation: random flips (left-right and up-down), brightness 0.1, contrast 0.9 to 1.1, applied once to the whole array before training (a fixed augmented copy, not per-epoch augmentation). Up-down flips of brain images are a questionable augmentation.
9. `data_agumentation.py` loads the full dataset at import time (side effect when imported by the trainer).
10. `visualization.py` writes `generated_images_epoch_N.png` to the working directory and calls `plt.show()` every epoch (blocks training); matplotlib is not in requirements.txt.
11. `utils/data_loader.py` uses `keras.preprocessing.load_img`, which I expect to fail on Keras 2.13 (not run, not verified).
12. Hard-coded relative paths (`data/brain_tumor`, `data/non_brain_tumor`); no network code besides the implicit weight download.
13. No data files, model files or large blobs in the branch; the Phase 0 history scan already covered commit 2257a69 with 0 secret-pattern matches.
14. Risk summary: run-time network download, missing dependencies, incorrect FID preprocessing and incorrect IS, blocking plots.
15. Recommendation: do not merge; use only as a reference when FID is reimplemented properly in a later phase.

### A7. Addendum

Appended "Addendum: TensorFlow checks" to `docs/reports/phase0_report.md`,
commit `5c45253`.

## 4. Part B: changes per requirement

### B1. .gitignore (done)

Added `*.pth`, `*.zip`, `*.dcm`, `*.nii`, `*.nii.gz`, `venv/`, `env/`,
`.ipynb_checkpoints/`, `.vscode/`, `.idea/`. Verified with
`git check-ignore -v` (all matched). The plan document is not ignored, so it
stays visible as untracked.

### B2. data_preprocessing.py (done)

Confirmed defects: local `load_images_from_folder` shadowed the imported one
and lacked the `os.path.isfile` check; `tensorflow`, `keras` and
`keras.utils` imports were unused. Now uses `utils.image_utils` (with the
isfile check), unused imports removed, `argparse` with `--brain-tumor-dir`
(default `data/brain_tumor`), `--non-brain-tumor-dir` (default
`data/non_brain_tumor`), `--output-dir` (default `data`, created if
missing), code under `main()`.

Byte-identity proof on the synthetic fixture (baseline script extracted with
`git show baseline-v0:data_preprocessing.py` into a temp folder, run with
cwd containing a copy of the fixture under `data/`):

```
brain_tumor_images.npy (158, 128, 128, 1) float32 array_equal= True sha256_equal= True 1ce170dc011bc095
non_brain_tumor_images.npy (152, 128, 128, 1) float32 array_equal= True sha256_equal= True e6fa25268555f68a
```

The same comparison is automated in `tests/test_preprocessing.py`.
Observable difference: the error message path for unreadable files is now
built with `os.path.join`, so it prints `data\brain_tumor\x.png` instead
of `data/brain_tumor\x.png` on Windows. Subfolders are now skipped silently
instead of printing an error.

### B3. model_training.py (done)

Confirmed: hard-coded 50 / 64 / 1e-4 / 100, generator-only save to a
relative path, no seed option. Changes:

- New `config.py` with `NOISE_DIM = 100`, imported by `models/generator.py`
  (`build_generator(noise_dim=NOISE_DIM)`, same default value),
  `model_training.py` (`tf.random.normal([batch_size, NOISE_DIM])`) and
  `gui_app.py`.
- Options with original defaults: `--epochs 50`, `--batch-size 64`,
  `--seed` (none by default, so no seeding unless given; when given,
  `tf.keras.utils.set_random_seed`), `--output-dir .` (same file name
  `generator_w.weights.h5` in the working directory).
- Saves `discriminator_w.weights.h5` next to the generator weights.
- Optional `--checkpoint-every N` (default 0, off): both networks to
  `<output-dir>/checkpoints/epoch_NNNN_*.weights.h5`.
- Optional `--sample-every N` (default 0, off): 8x8 grid to
  `<output-dir>/samples/synthetic_grid_epoch_NNNN.png` with a SYNTHETIC
  banner; uses a private NumPy generator for fixed noise so it does not touch
  TensorFlow or global NumPy random state (a test confirms identical weights
  with and without both callbacks).
- `callbacks=` is passed to `fit` only when a callback is enabled, so the
  default `fit` call is the same as before. Learning rate stays a literal
  1e-4 (no option requested).
- Code moved under `main(argv)`; the `GAN` class is unchanged apart from the
  constant.

Runner change required by the new CLIs: `tools/run_baseline.py` executes the
scripts with `runpy` in a child process whose `sys.argv` held the runner's
own options (`--seed 42 --child-stage ...`). The new `argparse` in the scripts
would have rejected them. Fix: set `sys.argv = [script]` before each
`runpy.run_path` (4 added lines), so the scripts run with their defaults.
Committed with the preprocessing fix (`6d51d68`).

### B4. gui_app.py (done)

Confirmed: nested `self.master.mainloop()` in `generate_images`, local-only
`PhotoImage`, relative weights path without error handling, unused
`filedialog` import, redundant shape check, no synthetic label. Changes: no
nested mainloop; `self.tk_img` holds the image; canvas cleared before each
draw; permanent red label "SYNTHETIC IMAGE generated by a GAN for research
only. Not a real scan. Not for clinical use."; window title mentions
synthetic images; missing or unloadable weights show `messagebox.showerror`
and disable the button instead of crashing; `--weights` option (default
`generator_w.weights.h5`); unused import and explicit shape check removed
(`reshape(128, 128)` still raises on a wrong shape). Pixel conversion is the
same formula, `(x.reshape(128, 128) * 255).astype(np.uint8)`, checked by a
test. Model unchanged.

### B5. requirements.txt (done)

Clean venv outside the repo, `C:\Users\iamab\.venvs\tk_check_py311`
(deleted afterwards), Python 3.11.7, no packages installed:

```
$ python --version
Python 3.11.7
$ python -m pip list
pip        23.2.1
setuptools 65.5.0
$ python -c "import importlib.util as u; print('tk package spec:', u.find_spec('tk'))"
tk package spec: None
$ python -c "import tkinter; print(...); r=tkinter.Tk(); r.withdraw(); ...; r.destroy()"
tkinter OK, TkVersion 8.6 TclVersion 8.6
Tk root created and destroyed
```

`pip show -f tk` in the Phase 0 venv: `tk 0.1.0`, summary "TensorKit is a
deep learning helper between Python and C++", files `tk/__init__.py`,
`tk/structure/...`. It is unrelated to Tkinter and nothing in the repo
imports it (`git grep` for `import tk` / `from tk`: no matches). The
`tk==0.1.0` line was removed; no other pins changed.

### B6. README.md (done)

Removed the RTX 4090 claim; development machine stated as Windows 11 with an
RTX 4060 Laptop GPU (8 GB). States that both generator and discriminator
weights are saved. Adds: data/ is not in the repository and must be
supplied; a Dataset section with TODO placeholders for source, license,
modality, counts and publication terms; Intended Use and Limits (research
only, synthetic images are not real scans, no clinical use, no claim that
augmentation improves downstream models); the Phase 0 tools with links to
`docs/AUDIT_USAGE.md` and `docs/BASELINE_USAGE.md`; the note that
TensorFlow 2.11 and later has no native-Windows GPU support; the new CLI
options; the single test command `python -m pytest`. The overview no longer
states augmentation as an achieved goal. `docs/BASELINE_USAGE.md` got a short
note that the runner now runs the Phase 1 scripts, with the equivalence
result.

### B7. LICENSE (not done, by design: your decision)

Common options:

- **MIT**: very permissive, short; allows any use including commercial, requires keeping the notice.
- **Apache 2.0**: permissive like MIT, plus an explicit patent grant and patent-retaliation clause.
- **BSD 3-Clause**: permissive, similar to MIT, also forbids using your name to endorse derivatives.
- **GPL-3.0**: copyleft; distributed derivatives must be released under GPL-3.0 as well.
- **AGPL-3.0**: like GPL-3.0, and also applies when the software is offered as a network service.
- **MPL 2.0**: file-level copyleft; changed files stay open, the larger work can be under other terms.
- **CC BY-NC 4.0** (for docs or weights, not ideal for code): attribution, non-commercial only.
- **No license**: all rights reserved by default; others may view on GitHub but have no right to reuse.

Note that a code license does not settle the rights to model weights trained
on your data; those depend on the dataset terms.

## 5. Tests

Single command (documented in README): `python -m pytest`
(`pytest.ini`: `testpaths = tools/tests tests`, `addopts = -p no:cacheprovider`).
On this machine I ran it with `-B` and `--basetemp=<scratch folder>`
because pytest's default temp folder raises a Windows permission error
during cleanup in my environment (seen in Phase 0).

```
$ python -m py_compile <all 18 tracked .py files>
py_compile OK: 18 files
$ python -B -m pytest --basetemp=<scratch> -v
tools/tests/test_audit_data.py ... 17 tests PASSED
tools/tests/test_run_baseline.py ... 7 tests PASSED
tests/test_gui_logic.py::test_to_uint8_image_matches_original_formula PASSED
tests/test_gui_logic.py::test_to_uint8_image_rejects_wrong_size PASSED
tests/test_gui_logic.py::test_load_generator_missing_file PASSED
tests/test_gui_logic.py::test_label_text_says_synthetic PASSED
tests/test_gui_logic.py::test_missing_weights_shows_message_and_disables_button PASSED
tests/test_gui_logic.py::test_generate_keeps_persistent_image_and_does_not_block PASSED
tests/test_model_shapes.py::test_noise_dim_constant PASSED
tests/test_model_shapes.py::test_generator_output_shape_and_range PASSED
tests/test_model_shapes.py::test_discriminator_output_shape_and_range PASSED
tests/test_model_shapes.py::test_parameter_counts PASSED
tests/test_preprocessing.py::test_new_script_matches_baseline_v0 PASSED
tests/test_preprocessing.py::test_default_paths_unchanged PASSED
tests/test_preprocessing.py::test_subdirectories_ignored_cleanly PASSED
tests/test_preprocessing.py::test_unreadable_files_reported PASSED
tests/test_train_step_smoke.py::test_two_train_steps_give_finite_losses PASSED
tests/test_train_step_smoke.py::test_fit_passes_plain_float32_batch_tensor PASSED
tests/test_train_step_smoke.py::test_cli_defaults_reproduce_original_configuration PASSED
tests/test_train_step_smoke.py::test_main_saves_both_weight_files PASSED
tests/test_train_step_smoke.py::test_optional_callbacks_write_checkpoints_and_synthetic_grids PASSED
tests/test_train_step_smoke.py::test_callbacks_do_not_change_training_numerics PASSED
tests/test_train_step_smoke.py::test_invalid_epochs_rejected PASSED
====================== 45 passed, 65 warnings in 28.59s =======================
```

(The 24 Phase 0 tool test names are listed in the Phase 0 report; trimmed
here for length.) The 65 warnings are `DeprecationWarning`s from inside
TensorFlow/Keras and the standard library (`sre_constants`, `randrange`
with non-integer arguments), not from project code.

Verified facts:

- Parameter counts match the expected values: generator 2,343,649,
  discriminator 722,369 (`count_params()`), no difference.
- Open question resolved: under `fit()` with a float32 NumPy array,
  `train_step` receives a single `tf.Tensor` (not a tuple), static shape
  `(None, 128, 128, 1)`, dtype float32; with 10 samples and batch size 4 the
  dynamic batch sizes were `[4, 4, 2]`. So `tf.shape(real_images)[0]` in the
  baseline is the true batch size, including the smaller last batch.
- GUI tests ran with a display (not skipped) on this machine.

Test issue found and fixed: in one full-suite run a GUI test was skipped
because the second `tk.Tk()` in the same process failed Tcl initialisation
("This probably means that Tcl wasn't installed properly"). It was
intermittent (6 separate runs of the GUI file passed). The fixture now
creates one Tk root per module and a `Toplevel` per test; afterwards 6
GUI-file runs and 3 full-suite runs all gave 0 skips (45 passed each time).
This fix is in a second commit with the same test commit message
(`6c044aa`).

## 6. Behavior-preservation check (synthetic smoke test)

After all code changes:

```
$ python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/phase1_check --smoke-epochs 2 --seed 42 --cpu
[run_baseline] SMOKE TEST: overriding epochs 50 -> 2
3/3 [==============================] - 5s 738ms/step - gen_loss: 0.7448 - disc_loss: 1.3567
3/3 [==============================] - 3s 788ms/step - gen_loss: 0.8537 - disc_loss: 1.2861
[run_baseline] smoke_test finished with return code 0; total wall time 21.054 s; 2 epochs parsed
```

Compared programmatically with Part A `outputs/smoke_run1`:

```
epoch 1 gen_loss: 0.7448 vs 0.7448 diff=0.0
epoch 1 disc_loss: 1.3567 vs 1.3567 diff=0.0
epoch 2 gen_loss: 0.8537 vs 0.8537 diff=0.0
epoch 2 disc_loss: 1.2861 vs 1.2861 diff=0.0
losses identical (string): True
losses.csv sha256 equal: True
weights sha256 equal: True
sample pixels identical across runs: True
brain_tumor_images.npy True
non_brain_tumor_images.npy True
```

Result: **identical**. Losses, `losses.csv` (this time including epoch
times), generator weights (SHA-256), all 65 sample PNGs and both
preprocessed arrays match. The work folder now also contains
`discriminator_w.weights.h5` (new, expected). Wall times (synthetic, CPU):
preprocess 4.348 s, train 11.687 s, export 4.87 s.

## 7. Files added, changed, removed

- Added: `config.py`, `pytest.ini`, `tests/conftest.py`,
  `tests/test_preprocessing.py`, `tests/test_model_shapes.py`,
  `tests/test_train_step_smoke.py`, `tests/test_gui_logic.py`,
  `docs/reports/phase1_report.md`.
- Changed: `.gitignore`, `data_preprocessing.py`, `model_training.py`,
  `models/generator.py`, `gui_app.py`, `requirements.txt`, `README.md`,
  `tools/run_baseline.py`, `docs/reports/phase0_report.md`,
  `docs/BASELINE_USAGE.md`.
- Removed: none. No file moves. `models/discriminator.py`, `utils/` unchanged.

## 8. Requirement status

| item | status |
|---|---|
| A1 GPU check | Done: no GPU on native Windows |
| A2 fixture | Done |
| A3 two runs | Done, no fix needed |
| A4 comparison | Done: identical losses, byte-identical weights |
| A5 export check | Done |
| A6 branch review | Done (read-only) |
| A7 addendum | Done (`5c45253`) |
| B1 gitignore | Done |
| B2 preprocessing | Done, byte-identical outputs proven |
| B3 training | Done; runner needed a 4-line argv fix |
| B4 GUI | Done |
| B5 tk | Done: removed after verification on Python 3.11.7 |
| B6 README | Done |
| B7 LICENSE | Not done by design, options listed, your decision |
| Tests | Done, 45 passed, one command |
| Behavior check | Done, identical |

## 9. Pre-push checklist

Results at the time of writing (before the final docs commit; re-run after
it and printed in the terminal summary).

| item | result |
|---|---|
| a. clean apart from plan file, all committed | Pass (only `docs/medical_image_synthesis_modernization_plan.md` untracked) |
| b. tests and py_compile | Pass: 45 passed; py_compile OK on 18 files |
| c. no dataset-like or artifact files tracked | Pass: 0 |
| d. no file over 1 MB, no secret patterns | Pass: 0 files over 1 MB (largest added/modified file 30,664 bytes); 0 secret-pattern matches in added lines |
| e. diff stat shown | Below (before the docs commit) and in the terminal summary |
| f. fast-forward push | Done after this commit, see terminal summary |
| g. no Claude/Anthropic/Co-Authored-By/Claude-Session lines in `git log --format=%B origin/main..main` | Pass: 0 matching lines |

`git diff --stat origin/main..main` before the docs commit:

```
 .gitignore                     |  12 +++++
 README.md                      | 102 ++++++++++++++++++++++++++++++-------
 config.py                      |   4 ++
 data_preprocessing.py          |  45 ++++++++++-------
 docs/reports/phase0_report.md  | 112 +++++++++++++++++++++++++++++++++++++++++
 gui_app.py                     |  80 ++++++++++++++++++++++-------
 model_training.py              | 107 +++++++++++++++++++++++++++++++++++++--
 models/generator.py            |   3 +-
 pytest.ini                     |   3 ++
 requirements.txt               |   1 -
 tests/conftest.py              |  29 +++++++++++
 tests/test_gui_logic.py        |  81 +++++++++++++++++++++++++++++
 tests/test_model_shapes.py     |  31 ++++++++++++
 tests/test_preprocessing.py    |  75 +++++++++++++++++++++++++++
 tests/test_train_step_smoke.py | 108 +++++++++++++++++++++++++++++++++++++++
 tools/run_baseline.py          |   4 ++
 16 files changed, 733 insertions(+), 64 deletions(-)
```

The pushed commit hash is that of the commit containing this report, so it
cannot be written here; it is verified with `git ls-remote` and given in the
terminal summary.

Commits in this task:

```
5c45253 docs: add phase 0 addendum with TensorFlow checks
7061187 chore: extend gitignore for archives, medical formats and checkpoints
6d51d68 fix(preprocessing): remove shadowed loader and add path options
74034fa feat(training): add seed, output dir, discriminator save and optional callbacks
f6a65f8 fix(gui): remove nested mainloop, keep image reference, handle missing weights, label images synthetic
a952613 build: drop unneeded tk requirement
8596af7 docs: correct README and add intended-use section
01de9a6 test: add preprocessing, shape, train-step and gui logic tests
6c044aa test: add preprocessing, shape, train-step and gui logic tests
(next)  docs: add phase 1 report
```

## 10. Measurements (all synthetic smoke tests, CPU)

| measurement | value |
|---|---|
| Part A run 1 total / preprocess / train / export (s) | 23.831 / 6.9 / 12.007 / 4.759 |
| Part A run 2 total / preprocess / train / export (s) | 19.606 / 4.147 / 10.756 / 4.561 |
| Phase 1 check total / preprocess / train / export (s) | 21.054 / 4.348 / 11.687 / 4.87 |
| losses (all three runs) | epoch 1: gen 0.7448, disc 1.3567; epoch 2: gen 0.8537, disc 1.2861 |
| arrays | brain_tumor (158, 128, 128, 1), non_brain_tumor (152, 128, 128, 1), float32 |
| full test suite | 45 passed in 28.59 s |

Seeds: runner 42, fixture 1234; tests use fixed seeds in the test code.
These loss values describe 2 epochs on procedural images and mean nothing
about model quality.

## 11. Deviations

- `tests/test_gui_logic.py` fix landed in a second commit reusing the test
  commit message (I did not amend, to avoid rewriting history).
- Runner change (`sys.argv` reset) committed with the preprocessing fix,
  since that is the change that required it.
- Added `pytest.ini` and `tests/conftest.py` (needed for the one-command
  suite); added an extra test group in `tests/test_train_step_smoke.py`
  for the trainer CLI, saving and callbacks.
- `gui_app.py` got a `--weights` option (default unchanged) so the GUI can
  use weights saved with `--output-dir`.
- `docs/BASELINE_USAGE.md` got a short accuracy note (not in the suggested
  file list).
- Previous Phase 0 commits contain a Co-Authored-By line; per your rule they
  were not rewritten. No commit in this task contains one.

## 12. Open problems and risks

- The runner now runs the current scripts. Re-running the literal
  `baseline-v0` files needs a separate checkout of the tag (for example
  `git worktree add ../mis_baseline_v0 baseline-v0`), and the runner has no
  option to point at another checkout. The equivalence check covers the
  default path only.
- Bitwise reproducibility is shown only for CPU, 2 epochs, this fixture.
- Keras prints losses to 4 decimals; `losses.csv` cannot show smaller
  differences.
- The baseline still resizes non-square images with distortion and trains on
  the brain_tumor class only; unchanged by design.
- The GUI draws noise with `tf.random.normal` without a seed (as before).
- Intermittent Tcl initialisation failure when creating several Tk roots in
  one process was worked around in the tests only.

## 13. Rollback

- Revert specific commits without rewriting history, for example
  `git revert a952613` (tk removal) or `git revert --no-edit 5c45253^..main`
  for the whole task, then push normally.
- Original code at any time, local only: `git switch --detach baseline-v0`
  or `git worktree add ../mis_baseline_v0 baseline-v0`.
- To return a local checkout to the Phase 0 state without touching the
  remote: `git switch --detach 26f8c9a`.

## 14. Questions for you

1. Which license do you want (section 4, B7), or none for now?
2. Do your dataset terms allow publishing a model (weights) trained on it,
   or synthetic samples from such a model?
3. Should a later phase add a runner option to run the literal
   `baseline-v0` checkout, for side-by-side comparisons?
4. Please fill in the Dataset placeholders in README.md (source, license,
   modality), or tell me the values to enter.
