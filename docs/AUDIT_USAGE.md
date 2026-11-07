# Dataset audit: how to run it on private data

`tools/audit_data.py` is a read-only audit of image folders. It produces a
markdown report that contains only counts and statistics, so that you can
share the report without sharing any image, file name or path.

## What the tool does and does not do

- It opens every file in the folders you pass, read-only, and computes
  statistics in memory.
- It never copies, moves, renames, modifies, deletes or uploads any input
  file. The only files it writes are the report (`--output`) and, if you ask
  for it, a JSON copy (`--json`). Without `--output` it prints to the screen.
- It refuses to write its output inside any folder it is auditing.
- It makes no network calls. It imports no network modules.
- It does not create `__pycache__` folders (the tool disables bytecode
  writing; running Python with `-B` also prevents it).
- These guarantees are checked by the automated test
  `test_audit_writes_only_its_output_and_no_network` in
  `tools/tests/test_audit_data.py`, which runs the tool under a Python audit
  hook that blocks and counts any file write outside the output files and
  any socket, subprocess or file-system mutation call.

## Requirements

Python 3.9 or later with `numpy` and `Pillow` (both pinned in
`requirements.txt`). TensorFlow is not needed for the audit.

## Running it

From the repository root, with your virtual environment active. Replace the
two folder paths with the real locations of your class folders, and keep the
report outside those folders:

```powershell
python -B tools/audit_data.py "D:\private\brain_tumor" "D:\private\non_brain_tumor" --output "$HOME\audit_report.md"
```

Optional flags:

| flag | effect |
|---|---|
| `--json PATH` | also write the same statistics as JSON |
| `--thresholds 0,2,4,6,8,10,12` | Hamming thresholds for near-duplicate counts |
| `--no-recursive` | only look at files directly inside each folder |
| `--show-filenames` | PRIVATE. Adds raw file names of flagged files (identifier-like names, metadata files, unreadable files). The report is then marked DO NOT SHARE. Use only for your own cleanup. |

`python tools/audit_data.py --help` prints the full option list.

## What the report contains

1. Files per folder: totals, readable images, unreadable files, non-image
   files, system files (Thumbs.db and similar), subfolders.
2. Per folder: detected formats, extensions, extension/format mismatches,
   image modes, colour class (grayscale, colour file with identical channels,
   true colour), size distribution, non-square and multi-frame counts, and how
   many images carry embedded metadata (EXIF, PNG text chunks, ICC). Metadata
   values are never read into the report.
3. Pixel intensity per folder (min, max, mean, std, and percentiles of the
   per-image mean) on the 8-bit grayscale conversion, which is what the
   baseline loader uses.
4. Exact duplicates by SHA-256, including groups that span both folders
   (possible class leakage).
5. Near-duplicate clusters by 64-bit perceptual hash at several Hamming
   thresholds, after collapsing exact duplicates. Lower thresholds are
   stricter. Expect some matches at the higher thresholds for any set of
   similar-looking images; the low thresholds are the meaningful ones.
6. File names that may contain identifiers: counts per pattern (keywords
   such as patient, subject, mrn, name, dob, case; date-like strings; runs of
   6 or more digits; @; two capitalised words) and anonymised name shapes
   (letters become `a`, digit runs become `9`, for example `a_9`).
7. Metadata or label files: counts by location and extension for non-image
   files in the folders and files in their parent folder(s) whose name or
   extension suggests labels, annotations, metadata, masks or splits. Their
   contents are never opened.

## What is safe to paste back

Safe to paste, if the report was generated **without** `--show-filenames`:

- Sections 1 to 7 of the markdown report, as produced. They contain counts,
  statistics, the last path component of each audited folder (for example
  `brain_tumor`), and anonymised name shapes.
- The `--json` file from the same run contains the same information.

Read the report once before pasting it. In particular:

- Section header lines show the folder names you passed. If a folder name
  itself is sensitive, rename it in the pasted text.
- Name shapes in section 6 contain no letters or digits from your file
  names, only structure. If even the structure is sensitive, omit that line.

Never paste:

- Any report generated with `--show-filenames` (it says DO NOT SHARE at the
  top and has a section 8 with raw names).
- Any image, file name, path, metadata value or file from the dataset.

## Testing the tool without real data

```powershell
python tools/make_synthetic_fixture.py            # writes fixtures/synthetic_raw (gitignored)
python -B tools/audit_data.py fixtures/synthetic_raw/brain_tumor fixtures/synthetic_raw/non_brain_tumor
python -B -m pytest tools/tests -q -p no:cacheprovider
```

The fixture is procedural and obviously artificial. It contains planted
duplicates, near-duplicates, non-square images, an unreadable file, an RGB
image, a non-image file and two fake identifier-like file names, and the
tests check that the audit finds each of them.
