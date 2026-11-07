# Phase 0 report: baseline freeze and data audit tooling (code-only)

Date: 2026-11-07. Branch: main. All work was done without access to the
private dataset. Every measurement below comes from synthetic fixture data
and is a pipeline smoke test only. None of it is evidence about model
quality or about real data.

## 1. Summary

- Starting commit: `e46df816eb70f63f65cb1c87a76a6d8150403c8b` (e46df81,
  "Update README.md"), equal to `origin/main` at the start.
- Local tag `baseline-v0` created on that exact commit before any change.
- History scan: no blobs over 1 MB, no dataset or artifact extensions, no
  secret-pattern matches in any of the 5 commits on any branch.
- Added: synthetic fixture generator, read-only dataset audit, seeded
  baseline runner, 24 automated tests (all pass), usage guides, this report.
- No existing file was modified. `models/`, `utils/`, `model_training.py`,
  `data_preprocessing.py`, `gui_app.py`, `requirements.txt` and `README.md`
  are byte-identical to `baseline-v0`.
- `requirements.txt` installed cleanly in a new Python 3.11.7 venv outside
  the repo (TensorFlow install approved by you, 276.6 MB wheel).
- TensorFlow was **not run** in this phase, per your answer "Do not run TF
  at all". The baseline smoke run, the GPU visibility check and the
  determinism check are therefore not done; exact commands for you are in
  section 9.

## 2. Repository inspection (requirement: standing rule 1)

The repository matched the prompt's description. Small differences:

- Untracked file not created by me: `docs/medical_image_synthesis_modernization_plan.md`
  (126 KB). Per your answer it was left untracked and untouched; it is not
  committed.
- A remote branch exists that the prompt did not mention:
  `origin/feature_improvment_gan` at `2257a69` ("Added new data agumentation
  fid is calculator"), which adds `data_agumentation.py`, `fid.py`,
  `inception_score.py`, `visualization.py`. It was included in the history
  scan and was not touched.
- `README.md` lists an RTX 4090; this machine has an RTX 4060 Laptop GPU
  (8 GB). README not edited (out of scope).

Commands and output:

```
$ git status
On branch main
Your branch is up to date with 'origin/main'.
Untracked files:
	docs/
nothing added to commit but untracked files present

$ git log -5 --oneline
e46df81 Update README.md
1774868 Medical Image Augmentation and Synthesis
e1ed8f7 Medical Image Augmentation and Synthesis
2b152fe Initial commit

$ git log --all --oneline --graph
* e46df81 Update README.md
| * 2257a69 Added new data agumentation fid is calculator
|/
* 1774868 Medical Image Augmentation and Synthesis
* e1ed8f7 Medical Image Augmentation and Synthesis
* 2b152fe Initial commit

$ git remote -v
origin  https://github.com/xyntrae/medical_image_synthesis.git (fetch)
origin  https://github.com/xyntrae/medical_image_synthesis.git (push)
```

Files at start (excluding `.git`): `data_preprocessing.py`, `gui_app.py`,
`model_training.py`, `models/generator.py`, `models/discriminator.py`,
`utils/image_utils.py`, `utils/data_loader.py`, `requirements.txt`
(`tensorflow==2.13.0`, `numpy==1.24.3`, `Pillow==9.4.0`, `tk==0.1.0`),
`README.md`, plus the untracked plan document. All tracked source files were
read. Code facts confirmed: unconditional Keras GAN, 128x128x1, noise dim 100,
Adam 1e-4 for both networks, `gan.fit(..., epochs=50, batch_size=64)`, only
`generator_w.weights.h5` saved, data read from `data/brain_tumor/` and
`data/non_brain_tumor/` relative to the working directory.

## 3. Requirement 1: baseline tag and environment

```
$ git tag baseline-v0 e46df816eb70f63f65cb1c87a76a6d8150403c8b
$ git show-ref --tags -d
e46df816eb70f63f65cb1c87a76a6d8150403c8b refs/tags/baseline-v0
```

| item | value |
|---|---|
| OS | Windows 11 Pro, `Microsoft Windows [Version 10.0.26200.9711]` |
| Default `python` | 3.14.6 (`C:\Python314`), too new for TF 2.13 |
| Other Pythons | 3.12, 3.11.7, uv CPython 3.12.13 |
| Python used for the venv | 3.11.7 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver model WDDM |
| nvidia-smi | NVIDIA-SMI 610.47, KMD 610.47, CUDA UMD 13.3; 1431 MiB in use by desktop apps at idle |
| WSL2 | installed, default version 2; only distribution is `docker-desktop` (running). No Ubuntu or other general-purpose distribution |

nvidia-smi header (process list of 25 desktop apps trimmed for length):

```
| NVIDIA-SMI 610.47                 KMD Version: 610.47        CUDA UMD Version: 13.3     |
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   44C    P5              7W /   95W |    1431MiB /   8188MiB |      3%      Default |
```

WSL:

```
$ wsl --status
Default Distribution: docker-desktop
Default Version: 2
$ wsl -l -v
* docker-desktop    Running         2
```

Nothing about WSL2, CUDA or drivers was installed or changed.

## 4. Requirement 2: history scan

Scope: `git rev-list --objects --all` (all branches including
`origin/feature_improvment_gan`), 34 objects, 5 commits.

| check | result |
|---|---|
| blobs larger than 1 MB | 0 (largest blob is 3488 bytes, `model_training.py`) |
| paths with .npy .npz .h5 .hdf5 .pt .pth .ckpt .zip .dcm .nii .nii.gz, image extensions, .env .pem .key | 0 |
| secret patterns (`api_key`, `secret`, `token`, `password`/`passwd`, `pwd`, `access_key`, `private_key`, `auth` followed by `=` or `:`; AWS `AKIA...`, `ghp_...`, `sk-...`, `BEGIN ... PRIVATE KEY`) per commit via `git grep` | 0 matches in each of e46df81, 2257a69, 1774868, e1ed8f7, 2b152fe |
| same patterns in diffs via `git log --all -G` | 0 commits |

All paths ever committed: `README.md`, `data_agumentation.py`,
`data_preprocessing.py`, `fid.py`, `gui_app.py`, `inception_score.py`,
`model_training.py`, `models/discriminator.py`, `models/generator.py`,
`requirements.txt`, `utils/data_loader.py`, `utils/image_utils.py`,
`visualization.py`. Nothing dataset-like or secret-like was found, so there
was no reason to stop.

## 5. Requirement 3: virtual environment and TensorFlow

- Venv: `C:\Users\iamab\.venvs\mis_baseline_py311` (outside the repo),
  Python 3.11.7. TF 2.13 supports Python 3.8 to 3.11.
- Before installing I checked PyPI sizes: `tensorflow-intel==2.13.0`
  cp311 win_amd64 wheel is 276.6 MB (Apache 2.0). You approved the install.
- `pip install -r requirements.txt` result: success. Final lines:

```
Successfully installed absl-py-2.5.1 ... h5py-3.16.0 ... keras-2.13.1 ... protobuf-4.25.9 ...
tensorboard-2.13.0 ... tensorflow-2.13.0 tensorflow-estimator-2.13.0 tensorflow-intel-2.13.0
tensorflow-io-gcs-filesystem-0.31.0 ... (trimmed for length)
$ pip check
No broken requirements found.
```

- Installed versions: tensorflow 2.13.0, tensorflow-intel 2.13.0, keras
  2.13.1, numpy 1.24.3, Pillow 9.4.0, tk 0.1.0 (a PyPI placeholder package,
  not Tkinter; Tkinter ships with CPython), h5py 3.16.0, protobuf 4.25.9.
  Extra test-only package in the venv: pytest 9.2.0 (not added to
  requirements.txt).
- GPU visibility: **not checked by running TensorFlow**, because you chose
  not to run TF in this phase. Expected, not measured: TF 2.13 on native
  Windows is CPU only (last native Windows GPU release was TF 2.10), so it
  will list no GPU. You chose "Do not run TF at all"; the GPU route for the
  full run is still open (section 12).

## 6. Requirement 4: tools/make_synthetic_fixture.py

Generates `fixtures/synthetic_raw/` (gitignored) with `brain_tumor/` and
`non_brain_tumor/`. Images are procedural: gradient, sinusoidal texture,
soft patches, a rotated ellipse "shell" with small ellipses, Gaussian noise,
and a bright Gaussian blob in the brain_tumor class only. Each base image has
a black box with the text "SYN" in the top-left corner. Seeded
(`numpy.random.default_rng`, default seed 1234), deterministic (tested), and
refuses to write into a non-empty folder that is not a previous fixture.

Default fixture: 150 base images per class (10% at 256x256, 20% JPEG), plus:

| planted case | count | where |
|---|---|---|
| exact duplicates | 3 extra copies (3 groups) | 2 in brain_tumor, 1 copied into non_brain_tumor |
| near-duplicates | 3 | brain_tumor: 1 px shift, additive noise (sigma 4), JPEG re-encode at quality 70 |
| non-square images | 3 | brain_tumor: 160x128, 128x200, 96x64 |
| unreadable file | 1 | brain_tumor: PNG signature followed by random bytes |
| RGB image | 1 | non_brain_tumor: true colour, tinted channels |
| non-image file | 1 | non_brain_tumor: .txt |
| identifier-like names | 2 | non_brain_tumor: fake `patient_00000_scan.png`, `19000101_fixture.png` |

```
$ python tools/make_synthetic_fixture.py --overwrite
Synthetic fixture written to fixtures\synthetic_raw: brain_tumor=159 files, non_brain_tumor=153 files
```

Design note: in a first version some pairs of independent base images were
within Hamming distance 8 of each other, and the 2 px shifted copy was at
distance 8 from its original. I added more varied low-frequency structure to
the base images and changed the shift to 1 px. Measured on the default
fixture: planted near-duplicates are at distances 0, 0 and 4; the closest
pair of unrelated images is at distance 10.

## 7. Requirement 5: tools/audit_data.py

Read-only audit; takes folders as positional arguments; writes a markdown
report (`--output`, or stdout) and optionally JSON (`--json`). Reports every
item listed in the requirement: file counts, formats, extension mismatches,
size and mode distributions, grayscale vs colour (including colour files
with identical channels), intensity min/max/mean/std and per-image-mean
percentiles per folder on the 8-bit grayscale conversion, unreadable files,
non-image files, SHA-256 exact duplicates (with cross-folder groups),
perceptual-hash (64-bit DCT pHash, implemented with NumPy, no extra
dependency) near-duplicate clusters at thresholds 0, 2, 4, 6, 8, 10, 12,
identifier-like file name pattern counts and anonymised name shapes,
embedded metadata counts (EXIF, PNG text, ICC; values not read), and
metadata/label files in the folders and their parent folder(s).

Safety: opens inputs read-only, writes only its output files, refuses an
output path inside an audited folder, imports no network modules, disables
bytecode writing, shows folders by last path component only, and includes
raw file names only with `--show-filenames` (that report is marked DO NOT
SHARE). Usage guide: `docs/AUDIT_USAGE.md`.

Smoke test on synthetic fixture (report sections trimmed to the planted
cases; full report is in the gitignored `outputs/audit/audit_report.md`).
**SMOKE TEST ON SYNTHETIC DATA. Not real data.**

```
$ python -B tools/audit_data.py fixtures/synthetic_raw/brain_tumor fixtures/synthetic_raw/non_brain_tumor -o outputs/audit/audit_report.md --json outputs/audit/audit_report.json
Report written to outputs\audit\audit_report.md
JSON written to outputs\audit\audit_report.json
real	0m3.314s
```

| folder | files | images | unreadable | non-image |
|---|---|---|---|---|
| brain_tumor | 159 | 158 | 1 | 0 |
| non_brain_tumor | 153 | 152 | 0 | 1 |

- brain_tumor sizes: 128x128: 140, 256x256: 15, 128x200: 1, 160x128: 1, 96x64: 1; non-square 3
- non_brain_tumor modes: L: 151, RGB: 1; colour class: grayscale 151, color 1
- exact duplicates: 3 groups, 6 files, 3 extra copies, 1 cross-folder group
- identifier patterns: keyword 1, date-like 1, long digit run 1 (the date
  string also has 8 digits), files matching any pattern 2 of 312
- metadata files: `(parent): .json: 1` (the fixture manifest),
  `non_brain_tumor: .txt: 1`
- intensity (8-bit grayscale): brain_tumor min 0, max 255, mean 106.80,
  std 49.11; non_brain_tumor min 0, max 255, mean 103.21, std 43.33

| Hamming threshold | pairs | clusters | images in clusters | cross-folder clusters |
|---|---|---|---|---|
| <= 0 | 2 | 2 | 4 | 0 |
| <= 2 | 2 | 2 | 4 | 0 |
| <= 4 | 3 | 3 | 6 | 0 |
| <= 6 | 3 | 3 | 6 | 0 |
| <= 8 | 3 | 3 | 6 | 0 |
| <= 10 | 5 | 5 | 10 | 2 |
| <= 12 | 15 | 14 | 29 | 8 |

The extra pairs at thresholds 10 and 12 are unrelated procedural images that
happen to have similar low-frequency structure. Similar effects should be
expected on any set of look-alike images, which is why the report shows
several thresholds.

## 8. Requirement 6: tools/run_baseline.py

External wrapper; edits no baseline file. Seeding is done without editing
the existing files, so no edit was needed or made. How it works:

- Each stage (preprocess, train, export) runs in a child Python process
  that sets `PYTHONHASHSEED`, Python `random`, NumPy and TF seeds
  (`tf.keras.utils.set_random_seed`), calls
  `tf.config.experimental.enable_op_determinism()`, with environment
  `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`,
  `TF_ENABLE_ONEDNN_OPTS=0` (opt out `--keep-onednn`),
  `TF_FORCE_GPU_ALLOW_GROWTH=true` (opt out `--no-memory-growth`), then runs
  the original script with `runpy`.
- Because the scripts use hard-coded relative paths, the wrapper creates
  `<output>/work/data/{brain_tumor,non_brain_tumor}` as directory links
  (Windows junctions, POSIX symlinks) to `--data-dir`, runs with
  `<output>/work` as working directory, and removes the links afterwards.
  Nothing is written to `--data-dir`.
- Captures combined stdout/stderr to `stdout.log`, parses per-epoch
  `gen_loss` and `disc_loss` into `losses.csv`, records wall time per stage
  and total, polls `nvidia-smi` once per second for device memory (before,
  peak, difference), and writes `run_summary.json` (seeds, flags, versions,
  TF devices, op-determinism status, array shapes, git HEAD, baseline file
  SHA-256 and modification status).
- Exports `samples/SYNTHETIC_grid_8x8.png` (banner: SYNTHETIC, run type,
  seed) and `--num-samples` individual `samples/synthetic_sample_XXXX.png`
  files (named synthetic, PNG text fields `Synthetic=true` and a description;
  pixels unaltered), plus `SYNTHETIC_OUTPUTS_README.txt`.
- `--smoke-epochs N` overrides `epochs=50` at runtime by wrapping
  `tf.keras.Model.fit` in the child; runs are labelled `smoke_test`.
- Usage guide: `docs/BASELINE_USAGE.md`.

Status: implemented and tested without TensorFlow (log parser, argument
validation, junction create/remove leaving the target intact, and the full
parent pipeline with the three stages replaced by fakes). **Not run end to
end with TensorFlow**, so these parts are unverified: the child seeding code,
the `fit` override, the sample export, GPU memory sampling during real
training, and run-to-run determinism.

## 9. Requirement 7: end-to-end tests

- Audit on the synthetic fixture: done (section 7), smoke test on
  synthetic data.
- Baseline short run: **not done**, because you chose not to run TF in this
  phase. Commands for you to run (from the repo root, PowerShell):

```powershell
& $HOME\.venvs\mis_baseline_py311\Scripts\Activate.ps1
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python tools/make_synthetic_fixture.py --overwrite
python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run1 --smoke-epochs 2 --seed 42 --cpu
python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run2 --smoke-epochs 2 --seed 42 --cpu
fc.exe outputs\smoke_run1\losses.csv outputs\smoke_run2\losses.csv
```

  Expected: both runs finish with return code 0, `losses.csv` has 2 rows,
  `samples/` has the grid and 64 PNGs, and `fc` reports no differences if
  seeding is effective. The first command answers the GPU question.
  Sharing `run_summary.json` and `losses.csv` from the synthetic runs back to
  me is safe (no private data involved).

## 10. Tests

All commands run in the venv (Python 3.11.7, numpy 1.24.3, Pillow 9.4.0,
pytest 9.2.0), on CPU. TensorFlow is installed but no test imports it.

```
$ python -m py_compile data_preprocessing.py gui_app.py model_training.py models/discriminator.py models/generator.py tools/audit_data.py tools/make_synthetic_fixture.py tools/tests/test_audit_data.py utils/data_loader.py utils/image_utils.py tools/run_baseline.py tools/tests/test_run_baseline.py
py_compile: OK (12 files)
```

(`__pycache__` folders created by py_compile were removed afterwards.)

```
$ python -B -m pytest tools/tests -v -p no:cacheprovider --basetemp=<scratch dir>
tools/tests/test_audit_data.py::test_file_status_counts PASSED
tools/tests/test_audit_data.py::test_exact_duplicates PASSED
tools/tests/test_audit_data.py::test_near_duplicates[4] PASSED
tools/tests/test_audit_data.py::test_near_duplicates[6] PASSED
tools/tests/test_audit_data.py::test_near_duplicates[8] PASSED
tools/tests/test_audit_data.py::test_near_duplicate_exact_copies_excluded PASSED
tools/tests/test_audit_data.py::test_rgb_and_non_square PASSED
tools/tests/test_audit_data.py::test_identifier_patterns PASSED
tools/tests/test_audit_data.py::test_report_contains_no_file_names PASSED
tools/tests/test_audit_data.py::test_show_filenames_flag PASSED
tools/tests/test_audit_data.py::test_metadata_file_detection PASSED
tools/tests/test_audit_data.py::test_fixture_is_deterministic PASSED
tools/tests/test_audit_data.py::test_fixture_refuses_foreign_directory PASSED
tools/tests/test_audit_data.py::test_refuses_output_inside_audited_folder PASSED
tools/tests/test_audit_data.py::test_help_text PASSED
tools/tests/test_audit_data.py::test_audit_writes_only_its_output_and_no_network PASSED
tools/tests/test_audit_data.py::test_guard_actually_catches_writes PASSED
tools/tests/test_run_baseline.py::test_parse_keeps_last_value_per_epoch PASSED
tools/tests/test_run_baseline.py::test_parse_scientific_nan_and_ignores_preamble PASSED
tools/tests/test_run_baseline.py::test_help PASSED
tools/tests/test_run_baseline.py::test_rejects_missing_class_folder PASSED
tools/tests/test_run_baseline.py::test_rejects_output_inside_data_and_non_empty_output PASSED
tools/tests/test_run_baseline.py::test_dir_link_roundtrip_leaves_target_intact PASSED
tools/tests/test_run_baseline.py::test_orchestration_with_fake_stages PASSED
============================= 24 passed in 8.30s ==============================
```

The no-writes test runs the audit in a subprocess under a Python audit hook
(`sys.addaudithook`) that blocks and counts any file opened for writing
other than the two output files, and any `os.remove`, `rename`, `mkdir`,
`shutil` copy/move, `socket.connect`/`getaddrinfo`, `subprocess.Popen`
and similar events. It asserts 0 violations, an unchanged fixture tree
(size, mtime, SHA-256 of every file), an empty working directory and an
output folder containing only the two outputs. A second test confirms the
hook does catch a forbidden write.

Test failures during development, and fixes: (1) pytest's default temp
folder under `%TEMP%\pytest-of-iamab` raised `PermissionError: [WinError 5]`
during cleanup, fixed by `--basetemp` in a scratch folder (environment
issue, not a code issue); (2) the write-guard harness only caught
`SystemExit`, so an uncaught `PermissionError` skipped its summary line;
fixed in the test harness.

## 11. Pre-push checklist

Results at the time of writing, over `origin/main..main` (4 tool commits
plus the docs commit that contains this report; the checks are re-run after
that commit and printed in the terminal summary).

| item | result |
|---|---|
| a. git status clean, all work committed | Pass for all phase work. One pre-existing untracked file remains, `docs/medical_image_synthesis_modernization_plan.md`, which you told me to leave untracked |
| b. all phase tests pass | Pass: 24 passed, py_compile OK on 12 files |
| c. no dataset-like or artifact files tracked | Pass: 0 tracked files with .npy .npz .h5 .hdf5 .pt .pth .ckpt .zip .dcm .nii .nii.gz or image extensions, 0 under data/ fixtures/ outputs/ cache/ |
| d. no file over 1 MB added, no secret patterns | Pass: largest added file 25600 bytes (`tools/audit_data.py`); 0 secret-pattern matches in added lines |
| e. `git diff --stat origin/main..main` shown | See terminal summary (final) and below (before the docs commit) |
| f. normal fast-forward push with `git push origin main` | Done after this commit; result in terminal summary |

`git diff --stat origin/main..main` before the docs commit:

```
 .gitignore                       |  13 +
 tools/audit_data.py              | 560 +++++++++++++++++++++++++++++++++++++++
 tools/make_synthetic_fixture.py  | 222 ++++++++++++++++
 tools/run_baseline.py            | 479 +++++++++++++++++++++++++++++++++
 tools/tests/test_audit_data.py   | 248 +++++++++++++++++
 tools/tests/test_run_baseline.py | 137 ++++++++++
 6 files changed, 1659 insertions(+)
```

The pushed commit hash cannot be written inside the commit it identifies; it
is the commit that adds this report, and it is given in the terminal summary.

## 12. Files added, changed, removed

- Added: `.gitignore`, `tools/make_synthetic_fixture.py`,
  `tools/audit_data.py`, `tools/run_baseline.py`,
  `tools/tests/test_audit_data.py`, `tools/tests/test_run_baseline.py`,
  `docs/AUDIT_USAGE.md`, `docs/BASELINE_USAGE.md`,
  `docs/reports/phase0_report.md`.
- Changed: none. Removed: none.

Commits (all on main, additive):

```
6305dec chore: add gitignore for data, fixtures and artifacts
77a1449 feat(tools): add synthetic fixture generator
27d3168 feat(tools): add read-only dataset audit
83c500c feat(tools): add seeded baseline runner
(next)  docs: add audit and baseline usage guides and phase 0 report
```

## 13. Requirement status

| # | requirement | status |
|---|---|---|
| 1 | tag + environment record | Done |
| 2 | history scan | Done, clean |
| 3 | venv + install + GPU check | Partly done: install succeeded; GPU visibility not checked because TF was not run (your choice) |
| 4 | fixture generator | Done |
| 5 | audit tool + usage doc | Done |
| 6 | baseline runner + usage doc | Done as code; not run with TF |
| 7 | end-to-end tests | Audit: done. Baseline smoke run: not done (TF not run); commands in section 9 |
| 8 | checklist, push main, push tag | See terminal summary |

## 14. Measurements

All are synthetic-fixture smoke tests on CPU (Windows 11, Python 3.11.7,
numpy 1.24.3, Pillow 9.4.0). Fixture seed 1234. No TensorFlow measurements
exist for this phase.

| measurement | value |
|---|---|
| fixture size | 312 files (159 + 153) |
| audit wall time on the fixture (bash `time`, real) | 3.314 s |
| pytest suite wall time | 8.30 s |
| baseline training metrics, GPU memory, determinism | not measured (TF not run) |

## 15. Deviations from the prompt

- Tests live in `tools/tests/` (the constraints allow new files only under
  `tools/` and `docs/`).
- The baseline runner was not executed with TF and no TF GPU check was run,
  following your instruction.
- The near-duplicate fixture uses a 1 px shift (not 2 px) so the planted
  case stays clearly separated from unrelated images under pHash.
- `.gitignore` follows the list in the prompt exactly. It does not list
  `.pth`, `.zip`, `.dcm` or `.nii` (checklist item c covers them, and none
  are tracked).
- The fixture also writes `fixture_manifest.json` at the fixture root
  (gitignored, outside the class folders).
- `run_baseline.py` sets `TF_ENABLE_ONEDNN_OPTS=0` and
  `TF_FORCE_GPU_ALLOW_GROWTH=true` by default; both are recorded in
  `run_summary.json` and can be turned off.

## 16. Open problems and risks

- The runner's TF-dependent code path is untested. The most likely failure
  points: `enable_op_determinism()` can make some ops raise on GPU (on CPU
  this is not expected); the `fit` override assumes `epochs` is passed as a
  keyword, which is true for the current `model_training.py`.
- Baseline loader behaviour that the audit helps quantify but this phase
  does not change: `data_preprocessing.py` silently resizes non-square
  images to 128x128 (distortion), converts colour to grayscale, and prints
  file names of unreadable files.
- The unchanged `model_training.py` loads the full array into memory; with
  a large real dataset that is a RAM consideration.
- Keras-printed losses are running means over an epoch, not end-of-epoch
  values; the CSV records exactly what Keras printed.
- `nvidia-smi` memory is whole-device usage (desktop apps used about
  1.4 GB at idle), so the reported peak includes other programs.
- Pixel intensity stats use the 8-bit grayscale conversion; 16-bit sources
  would be reduced to 8-bit in those numbers (the mode counts show if any
  exist).
- The name-identifier check is heuristic and can miss identifiers or flag
  harmless names.

## 17. Rollback

- Revert individual commits without rewriting history, for example
  `git revert 83c500c` (runner) or the full range
  `git revert --no-edit 6305dec^..main`, then push normally.
- To get the original code locally without touching the remote:
  `git switch --detach baseline-v0` (or `git worktree add ../baseline baseline-v0`).
- The venv is outside the repo and can be deleted on its own:
  `C:\Users\iamab\.venvs\mis_baseline_py311`.

## 18. Questions for you

1. Please run the commands in section 9 and share `run_summary.json` and
   `losses.csv` from the synthetic runs (or tell me the error), so the
   runner can be marked as verified.
2. Which compute route do you want for the full 50-epoch run on your data:
   CPU on native Windows, or GPU through WSL2 (needs a WSL Ubuntu distro and
   CUDA 11.8 / cuDNN 8.6 libraries inside it, which you would install)?
3. Should `.gitignore` also list `*.pth`, `*.zip`, `*.dcm`, `*.nii`,
   `*.nii.gz` in a later phase?
4. Should the remote branch `origin/feature_improvment_gan` (FID, inception
   score, augmentation) be considered in a later phase, or ignored?
5. After you run the audit on your real data, please share the default
   report (sections 1 to 7 only, generated without `--show-filenames`).

## Addendum: TensorFlow checks

Date: 2026-11-07. Run after you authorised TensorFlow on synthetic fixtures,
CPU only. **Every result in this addendum is a smoke test on synthetic
fixture data. None of it says anything about model quality on real data.**

Environment: venv `C:\Users\iamab\.venvs\mis_baseline_py311`, Python 3.11.7,
tensorflow 2.13.0 (tensorflow-intel 2.13.0), keras 2.13.1, numpy 1.24.3,
Pillow 9.4.0, Windows 11, runner option `--cpu` (`CUDA_VISIBLE_DEVICES=-1`).
Seed 42 for the runner, fixture seed 1234.

### A1. GPU visibility

```
$ python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
2.13.0 []
$ python -c "import tensorflow as tf; print('built_with_cuda', tf.test.is_built_with_cuda())"
built_with_cuda False
```

TensorFlow 2.13 on native Windows does **not** see the GPU; the Windows
build is not built with CUDA. The TensorFlow baseline stays on CPU (your
decision; GPU work moves to PyTorch in Phase 4).

### A2. Fixture

```
$ python tools/make_synthetic_fixture.py --overwrite
Synthetic fixture written to fixtures\synthetic_raw: brain_tumor=159 files, non_brain_tumor=153 files
```

### A3. Two identical smoke runs

```
$ python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run1 --smoke-epochs 2 --seed 42 --cpu
$ python tools/run_baseline.py --data-dir fixtures/synthetic_raw --output-dir outputs/smoke_run2 --smoke-epochs 2 --seed 42 --cpu --quiet
```

Both finished with return code 0. No runner fix was needed. Trimmed
output of run 1 (TF CPU-feature info lines and progress-bar updates
omitted):

```
Error loading image data/brain_tumor\corrupt_0000.png: cannot identify image file <_io.BytesIO object at 0x...>
Error loading image data/non_brain_tumor\stray_notes.txt: cannot identify image file <_io.BytesIO object at 0x...>
[run_baseline] SMOKE TEST: overriding epochs 50 -> 2
Epoch 1/2
3/3 [==============================] - 5s 736ms/step - gen_loss: 0.7448 - disc_loss: 1.3567
Epoch 2/2
3/3 [==============================] - 3s 752ms/step - gen_loss: 0.8537 - disc_loss: 1.2861
[run_baseline] exported 64 samples and 8x8 grid to ...\outputs\smoke_run1\samples
[run_baseline] smoke_test finished with return code 0; total wall time 23.831 s; 2 epochs parsed
```

The two error lines are the planted unreadable file and the planted text
file; the unchanged preprocessing script skips them, as expected.
Child environment reported by every stage: `tf_gpus: []`,
`op_determinism: true`, `PYTHONHASHSEED=42`, `TF_DETERMINISTIC_OPS=1`,
`TF_CUDNN_DETERMINISTIC=1`, `TF_ENABLE_ONEDNN_OPTS=0`,
`CUDA_VISIBLE_DEVICES=-1`.

| (synthetic smoke test, CPU) | run 1 | run 2 |
|---|---|---|
| total wall time (s) | 23.831 | 19.606 |
| preprocess / train / export (s) | 6.9 / 12.007 / 4.759 | 4.147 / 10.756 / 4.561 |
| array shapes | brain_tumor (158, 128, 128, 1), non_brain_tumor (152, 128, 128, 1) | same |
| GPU memory before / peak (MiB, whole device) | 1321 / 1321 | 1321 / 1321 |

The GPU memory figures only show desktop usage; training ran on CPU.

### A4. Loss comparison (programmatic)

Compared with a Python script (csv module, float difference per value):

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

- Loss values are identical in both runs. The CSV files differ only in the
  `epoch_time` column (run 1: 5s, 3s; run 2: 4s, 2s), which is wall-clock
  time printed by Keras.
- Keras prints losses with 4 decimals, so the CSV alone cannot show
  differences below 1e-4. Stronger evidence: the saved generator weights
  files are byte-identical (SHA-256), the preprocessed arrays are equal
  (`np.array_equal`), and all 65 exported PNGs are pixel-identical between
  the runs.
- Conclusion for this setting only (CPU, seed 42, 2 epochs, this fixture):
  the runs were bitwise reproducible. This was not tested on GPU or for
  50 epochs.

### A5. Sample export and summary

- `samples/` in each run: 64 `synthetic_sample_XXXX.png` plus
  `SYNTHETIC_grid_8x8.png`.
- All 64 individual PNGs: format PNG, size 128x128, mode L (grayscale),
  PNG text `Synthetic=true`; file names start with `synthetic_`.
- Grid: 1042x1070, mode L, PNG text `Synthetic=true`, `RunType=smoke_test`,
  visible SYNTHETIC banner.
- `run_summary.json`: `run_type` smoke_test, `smoke_epochs` 2,
  `epochs_parsed` 2, `returncode` 0, `git_baseline_files_modified` empty.
  `SYNTHETIC_OUTPUTS_README.txt` present.
- Reduced-epoch option works: the log shows `overriding epochs 50 -> 2`,
  Keras prints `Epoch 1/2` and `Epoch 2/2`, and `losses.csv` has
  `total_epochs` 2.
