#!/usr/bin/env python
"""Read-only image dataset audit that produces a shareable statistics report.

Usage (see docs/AUDIT_USAGE.md for the full guide):

    python -B tools/audit_data.py PATH/TO/brain_tumor PATH/TO/non_brain_tumor \
        --output audit_report.md

What it reports, per folder and overall:
  * file counts, file formats (detected and by extension), extension mismatches
  * image size and mode distributions, grayscale vs colour (including colour
    files whose channels are identical), non-square images, bit depth
  * pixel intensity min, max, mean, std per folder, computed on the 8-bit
    grayscale ('L') conversion, which is what the baseline loader sees
  * unreadable files and non-image files
  * exact duplicates (SHA-256), including duplicates across folders
  * near-duplicate clusters (64-bit perceptual hash) at several Hamming
    distance thresholds, excluding byte-identical copies
  * file names that look like they may contain identifiers: pattern counts
    and anonymised name shapes only (raw names only with --show-filenames)
  * embedded metadata (EXIF, PNG text chunks) counts, without values
  * presence of metadata or label files in the folders and their parent

Safety guarantees:
  * Never writes, copies, moves, renames or deletes any input file. Input
    files are opened read-only. The only file written is --output (and
    --json if given). With no --output the report goes to stdout.
  * Refuses to write its output inside any audited folder.
  * Makes no network calls (no network modules are imported).
  * By default the report contains no file names and shows folders by their
    last path component only.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # do not create __pycache__ next to this tool

import argparse
import hashlib
import io
import json
import os
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

TOOL_VERSION = "1.0.0"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".pgm", ".ppm", ".pnm"}
SYSTEM_FILES = {"thumbs.db", ".ds_store", "desktop.ini"}
METADATA_EXTS = {".csv", ".tsv", ".json", ".xml", ".xls", ".xlsx", ".txt", ".yaml", ".yml", ".dcm",
                 ".nii", ".gz", ".mat", ".h5", ".hdf5", ".npy", ".npz", ".pkl", ".parquet"}
METADATA_NAME_RE = re.compile(r"label|annot|meta|mask|seg|manifest|split|train|test|valid|readme",
                              re.IGNORECASE)
GRAY_MODES = {"1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "F"}
DEFAULT_THRESHOLDS = (0, 2, 4, 6, 8, 10, 12)

ID_PATTERNS = {
    "keyword (patient/subject/mrn/name/dob/case)":
        re.compile(r"patient|subject|subj|mrn|name|dob|birth|case", re.IGNORECASE),
    "date-like (YYYYMMDD or YYYY-MM-DD)":
        re.compile(r"(?<!\d)(19|20)\d{2}[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)"),
    "long digit run (6 or more digits)": re.compile(r"\d{6,}"),
    "email-like (@)": re.compile(r"@"),
    "two capitalised words (Name_Surname)": re.compile(r"[A-Z][a-z]{2,}[ _\-.][A-Z][a-z]{2,}"),
}


# ----------------------------------------------------------------------------
# Hashing helpers
# ----------------------------------------------------------------------------
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_DCT32 = _dct_matrix(32)


def phash(gray: Image.Image) -> int:
    """64-bit perceptual hash: 32x32 resize, 2D DCT, 8x8 low band vs median."""
    small = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = _DCT32 @ small @ _DCT32.T
    low = d[:8, :8].flatten()
    med = np.median(low[1:])  # exclude DC term from the median
    bits = low > med
    return int("".join("1" if b else "0" for b in bits), 2)


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix_rows(hashes: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Hamming distances between hashes[start:stop] and all hashes (uint64)."""
    x = hashes[start:stop, None] ^ hashes[None, :]
    return _POP8[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ----------------------------------------------------------------------------
# Name helpers
# ----------------------------------------------------------------------------
def name_shape(stem: str) -> str:
    """Anonymised shape of a file stem: letters->'a', digits->'9', runs collapsed."""
    s = re.sub(r"[A-Za-z]+", "a", stem)
    s = re.sub(r"\d+", "9", s)
    return s


# ----------------------------------------------------------------------------
# Core audit
# ----------------------------------------------------------------------------
def _iter_files(folder: Path, recursive: bool):
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for f in sorted(files):
                yield Path(root) / f
    else:
        for p in sorted(folder.iterdir()):
            if p.is_file():
                yield p


def _inspect_file(path: Path):
    """Return (record, gray_image_or_None). Opens the file read-only."""
    rec = {"ext": path.suffix.lower(), "size_bytes": path.stat().st_size}
    data = path.read_bytes()
    rec["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rec["format"] = im.format or "unknown"
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            rec["n_frames"] = getattr(im, "n_frames", 1)
            info_keys = set(k.lower() for k in im.info.keys())
            rec["has_exif"] = bool("exif" in info_keys or (hasattr(im, "getexif") and len(im.getexif()) > 0))
            text_keys = info_keys - {"exif", "icc_profile", "dpi", "gamma", "transparency", "aspect",
                                     "jfif", "jfif_version", "jfif_unit", "jfif_density", "progressive",
                                     "progression", "interlace", "srgb", "adobe", "adobe_transform",
                                     "compression", "duration", "loop", "background", "version",
                                     "chromaticity"}
            rec["has_text_meta"] = bool(text_keys)
            rec["has_icc"] = "icc_profile" in info_keys
            if im.mode in GRAY_MODES:
                rec["color_class"] = "grayscale"
            else:
                rgb = np.asarray(im.convert("RGB"))
                same = bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))
                rec["color_class"] = "color_mode_gray_content" if same else "color"
            gray = im.convert("L")
        rec["status"] = "image"
        return rec, gray
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
        name = path.name.lower()
        if name in SYSTEM_FILES:
            rec["status"] = "system_file"
        elif rec["ext"] in IMAGE_EXTS:
            rec["status"] = "unreadable"
            rec["error_type"] = type(e).__name__
        else:
            rec["status"] = "non_image"
        return rec, None


def run_audit(folders, recursive=True, thresholds=DEFAULT_THRESHOLDS, show_filenames=False):
    folders = [Path(f) for f in folders]
    for f in folders:
        if not f.is_dir():
            raise SystemExit(f"Not a directory: {f}")
    labels = []
    for i, f in enumerate(folders):
        lab = f.resolve().name or f"folder_{i}"
        labels.append(lab if lab not in labels else f"{lab}_{i}")

    per_folder = {}
    all_records = []  # (folder_label, rel_name, rec)
    phashes = {}      # sha256 -> (phash, set(folder labels))
    for lab, folder in zip(labels, folders):
        fstats = {
            "files_total": 0, "subdirectories": sum(1 for _ in folder.rglob("*") if _.is_dir()) if recursive else 0,
            "status": Counter(), "format": Counter(), "ext": Counter(), "mode": Counter(),
            "size": Counter(), "color_class": Counter(), "ext_format_mismatch": 0, "non_square": 0,
            "multi_frame": 0, "has_exif": 0, "has_text_meta": 0, "has_icc": 0,
            "unreadable_error_types": Counter(), "non_image_ext": Counter(),
            "px_min": None, "px_max": None, "px_sum": 0.0, "px_sumsq": 0.0, "px_count": 0,
            "img_means": [],
        }
        for path in _iter_files(folder, recursive):
            fstats["files_total"] += 1
            rec, gray = _inspect_file(path)
            rel = str(path.relative_to(folder))
            all_records.append((lab, rel, rec))
            fstats["status"][rec["status"]] += 1
            fstats["ext"][rec["ext"] or "(none)"] += 1
            if rec["status"] == "unreadable":
                fstats["unreadable_error_types"][rec["error_type"]] += 1
            if rec["status"] == "non_image":
                fstats["non_image_ext"][rec["ext"] or "(none)"] += 1
            if rec["status"] != "image":
                continue
            fstats["format"][rec["format"]] += 1
            fstats["mode"][rec["mode"]] += 1
            fstats["size"][f"{rec['width']}x{rec['height']}"] += 1
            fstats["color_class"][rec["color_class"]] += 1
            fmt = rec["format"].upper()
            ext_ok = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}, "TIFF": {".tif", ".tiff"},
                      "BMP": {".bmp"}, "GIF": {".gif"}, "WEBP": {".webp"}}.get(fmt)
            if ext_ok is not None and rec["ext"] not in ext_ok:
                fstats["ext_format_mismatch"] += 1
            if rec["width"] != rec["height"]:
                fstats["non_square"] += 1
            if rec["n_frames"] > 1:
                fstats["multi_frame"] += 1
            for k in ("has_exif", "has_text_meta", "has_icc"):
                fstats[k] += int(rec[k])
            arr = np.asarray(gray, dtype=np.float64)
            mn, mx = float(arr.min()), float(arr.max())
            fstats["px_min"] = mn if fstats["px_min"] is None else min(fstats["px_min"], mn)
            fstats["px_max"] = mx if fstats["px_max"] is None else max(fstats["px_max"], mx)
            fstats["px_sum"] += float(arr.sum())
            fstats["px_sumsq"] += float((arr ** 2).sum())
            fstats["px_count"] += arr.size
            fstats["img_means"].append(float(arr.mean()))
            if rec["sha256"] not in phashes:
                phashes[rec["sha256"]] = [phash(gray), set()]
            phashes[rec["sha256"]][1].add(lab)
        per_folder[lab] = fstats

    # Finalise intensity statistics.
    for fs in per_folder.values():
        n = fs["px_count"]
        if n:
            mean = fs["px_sum"] / n
            fs["px_mean"] = mean
            fs["px_std"] = float(np.sqrt(max(fs["px_sumsq"] / n - mean ** 2, 0.0)))
            im = np.array(fs["img_means"])
            fs["img_mean_p5"], fs["img_mean_p50"], fs["img_mean_p95"] = (float(v) for v in np.percentile(im, [5, 50, 95]))
        for k in ("px_sum", "px_sumsq", "img_means"):
            fs.pop(k)

    # Exact duplicates across all files (any status, by content).
    by_sha = defaultdict(list)
    for lab, rel, rec in all_records:
        by_sha[rec["sha256"]].append((lab, rel, rec["status"]))
    dup_groups = [v for v in by_sha.values() if len(v) > 1]
    exact = {
        "groups": len(dup_groups),
        "files_in_groups": sum(len(g) for g in dup_groups),
        "extra_copies": sum(len(g) - 1 for g in dup_groups),
        "cross_folder_groups": sum(1 for g in dup_groups if len({x[0] for x in g}) > 1),
        "image_groups": sum(1 for g in dup_groups if g[0][2] == "image"),
    }

    # Near duplicates on distinct image contents.
    shas = list(phashes.keys())
    hashes = np.array([phashes[s][0] for s in shas], dtype=np.uint64)
    n = len(shas)
    max_t = max(thresholds) if thresholds else 0
    pairs = []  # (i, j, dist) with dist <= max_t, i < j
    chunk = 512
    for s in range(0, n, chunk):
        d = hamming_matrix_rows(hashes, s, min(s + chunk, n))
        ii, jj = np.nonzero(d <= max_t)
        for a, b in zip(ii, jj):
            gi = s + int(a)
            if gi < int(b):
                pairs.append((gi, int(b), int(d[a, b])))
    near = []
    for t in sorted(thresholds):
        uf = _UnionFind(n)
        npairs = 0
        for i, j, dist in pairs:
            if dist <= t:
                uf.union(i, j)
                npairs += 1
        clusters = defaultdict(list)
        for i in range(n):
            clusters[uf.find(i)].append(i)
        multi = [c for c in clusters.values() if len(c) > 1]
        cross = sum(1 for c in multi if len(set().union(*(phashes[shas[i]][1] for i in c))) > 1)
        near.append({
            "threshold": t, "pairs": npairs, "clusters": len(multi),
            "images_in_clusters": sum(len(c) for c in multi),
            "largest_cluster": max((len(c) for c in multi), default=0),
            "cross_folder_clusters": cross,
        })

    # File name identifier heuristics.
    id_counts = Counter()
    id_examples = defaultdict(list)
    shapes = Counter()
    any_flag = 0
    for lab, rel, rec in all_records:
        stem = Path(rel).stem
        shapes[name_shape(stem)] += 1
        hit = False
        for pname, rx in ID_PATTERNS.items():
            if rx.search(stem):
                id_counts[pname] += 1
                hit = True
                if show_filenames:
                    id_examples[pname].append(f"{lab}/{rel}")
        any_flag += int(hit)

    # Metadata / label files in folders and their parents (names inspected, files not opened).
    meta = Counter()
    meta_names = []
    parents = {f.resolve().parent for f in folders}
    candidates = [(lab, rel) for lab, rel, rec in all_records if rec["status"] in ("non_image", "unreadable")]
    for p in parents:
        for q in sorted(p.iterdir()):
            if q.is_file():
                candidates.append(("(parent)", q.name))
    for lab, rel in candidates:
        ext = Path(rel).suffix.lower()
        if ext in METADATA_EXTS or METADATA_NAME_RE.search(Path(rel).stem):
            meta[f"{lab}: {ext or '(none)'}"] += 1
            if show_filenames:
                meta_names.append(f"{lab}/{rel}")

    unreadable_names = [f"{lab}/{rel}" for lab, rel, rec in all_records if rec["status"] == "unreadable"] \
        if show_filenames else []
    return {
        "tool_version": TOOL_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "pillow": Image.__version__ if hasattr(Image, "__version__") else __import__("PIL").__version__,
        "numpy": np.__version__,
        "recursive": recursive,
        "show_filenames": show_filenames,
        "folders": labels,
        "per_folder": per_folder,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "identifier_patterns": dict(id_counts),
        "identifier_any": any_flag,
        "name_shapes_top": shapes.most_common(10),
        "name_shapes_distinct": len(shapes),
        "metadata_files": dict(meta),
        "filenames": {"identifier_matches": dict(id_examples), "metadata_files": meta_names,
                      "unreadable": unreadable_names} if show_filenames else None,
    }


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def _fmt_counter(c, limit=12):
    if not c:
        return "none"
    items = sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))
    s = ", ".join(f"{k}: {v}" for k, v in items[:limit])
    if len(items) > limit:
        s += f", ... ({len(items) - limit} more)"
    return s


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_markdown(r: dict) -> str:
    L = []
    L.append("# Dataset audit report")
    L.append("")
    L.append(f"Generated by tools/audit_data.py v{r['tool_version']} at {r['generated_utc']} UTC "
             f"(Python {r['python']}, Pillow {r['pillow']}, NumPy {r['numpy']}).")
    if r["show_filenames"]:
        L.append("")
        L.append("**WARNING: this report was generated with --show-filenames and contains raw file "
                 "names. Do not share it.**")
    else:
        L.append("This report contains only counts and statistics. It includes no file names and "
                 "shows folders by their last path component only.")
    L.append("")
    L.append("## 1. Files per folder")
    L.append("")
    L.append("| folder | files | images | unreadable | non-image | system files | subdirectories |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in r["folders"]:
        fs = r["per_folder"][lab]
        st = fs["status"]
        L.append(f"| {lab} | {fs['files_total']} | {st.get('image', 0)} | {st.get('unreadable', 0)} | "
                 f"{st.get('non_image', 0)} | {st.get('system_file', 0)} | {fs['subdirectories']} |")
    L.append("")
    L.append("## 2. Formats, sizes and modes (readable images only)")
    for lab in r["folders"]:
        fs = r["per_folder"][lab]
        L.append("")
        L.append(f"### {lab}")
        L.append("")
        L.append(f"- detected formats: {_fmt_counter(fs['format'])}")
        L.append(f"- file extensions (all files): {_fmt_counter(fs['ext'])}")
        L.append(f"- extension does not match detected format: {fs['ext_format_mismatch']}")
        L.append(f"- image modes: {_fmt_counter(fs['mode'])}")
        L.append(f"- colour class: {_fmt_counter(fs['color_class'])}")
        L.append(f"- image sizes (WxH): {_fmt_counter(fs['size'])}")
        L.append(f"- distinct sizes: {len(fs['size'])}; non-square images: {fs['non_square']}; "
                 f"multi-frame images: {fs['multi_frame']}")
        L.append(f"- embedded metadata (values not read into report): EXIF {fs['has_exif']}, "
                 f"PNG/other text chunks {fs['has_text_meta']}, ICC profile {fs['has_icc']}")
        if fs["unreadable_error_types"]:
            L.append(f"- unreadable error types: {_fmt_counter(fs['unreadable_error_types'])}")
        if fs["non_image_ext"]:
            L.append(f"- non-image file extensions: {_fmt_counter(fs['non_image_ext'])}")
    L.append("")
    L.append("Colour class meanings: grayscale = single-channel mode; color_mode_gray_content = "
             "RGB/RGBA/P file whose R, G and B channels are identical; color = true colour content.")
    L.append("")
    L.append("## 3. Pixel intensity (8-bit grayscale conversion, as the baseline loader sees it)")
    L.append("")
    L.append("| folder | min | max | mean | std | per-image mean p5 | p50 | p95 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lab in r["folders"]:
        fs = r["per_folder"][lab]
        L.append(f"| {lab} | {_f(fs['px_min'], 0)} | {_f(fs['px_max'], 0)} | {_f(fs.get('px_mean'))} | "
                 f"{_f(fs.get('px_std'))} | {_f(fs.get('img_mean_p5'))} | {_f(fs.get('img_mean_p50'))} | "
                 f"{_f(fs.get('img_mean_p95'))} |")
    L.append("")
    e = r["exact_duplicates"]
    L.append("## 4. Exact duplicates (SHA-256 of file bytes)")
    L.append("")
    L.append(f"- duplicate groups: {e['groups']} (of which image groups: {e['image_groups']})")
    L.append(f"- files in duplicate groups: {e['files_in_groups']}")
    L.append(f"- redundant extra copies: {e['extra_copies']}")
    L.append(f"- groups spanning more than one folder: {e['cross_folder_groups']}")
    L.append("")
    L.append("## 5. Near-duplicates (64-bit perceptual hash, distinct file contents only)")
    L.append("")
    L.append("Byte-identical copies are collapsed first, so these counts are in addition to section 4. "
             "Lower thresholds are stricter. Clusters are connected components of pairs within the "
             "threshold.")
    L.append("")
    L.append("| Hamming threshold | pairs | clusters | images in clusters | largest cluster | cross-folder clusters |")
    L.append("|---|---|---|---|---|---|")
    for row in r["near_duplicates"]:
        L.append(f"| <= {row['threshold']} | {row['pairs']} | {row['clusters']} | {row['images_in_clusters']} | "
                 f"{row['largest_cluster']} | {row['cross_folder_clusters']} |")
    L.append("")
    L.append("## 6. File names that may contain identifiers (heuristic)")
    L.append("")
    total_files = sum(r["per_folder"][lab]["files_total"] for lab in r["folders"])
    L.append(f"- files whose name matches at least one pattern: {r['identifier_any']} of {total_files}")
    for pname in ID_PATTERNS:
        L.append(f"- {pname}: {r['identifier_patterns'].get(pname, 0)}")
    L.append(f"- distinct anonymised name shapes: {r['name_shapes_distinct']} "
             "(letters shown as 'a', digit runs as '9')")
    L.append("- most common name shapes: " +
             (", ".join(f"`{s}`: {c}" for s, c in r["name_shapes_top"]) or "none"))
    L.append("")
    L.append("## 7. Metadata or label files")
    L.append("")
    L.append("Non-image files inside the audited folders, and files in their parent folder(s), whose "
             "extension or name suggests labels, annotations, metadata, masks or splits. Contents are "
             "never opened.")
    L.append("")
    L.append(f"- matches by location and extension: {_fmt_counter(r['metadata_files'])}")
    if r["filenames"] is not None:
        L.append("")
        L.append("## 8. Raw file names (--show-filenames). PRIVATE, DO NOT SHARE")
        L.append("")
        for k, v in r["filenames"]["identifier_matches"].items():
            L.append(f"- {k}: " + ", ".join(v))
        L.append("- metadata/label files: " + (", ".join(r["filenames"]["metadata_files"]) or "none"))
        L.append("- unreadable files: " + (", ".join(r["filenames"]["unreadable"]) or "none"))
    L.append("")
    return "\n".join(L)


def _json_ready(r: dict) -> dict:
    def conv(o):
        if isinstance(o, Counter):
            return dict(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    return conv(r)


def _check_output_location(out: Path, folders) -> None:
    out_r = out.resolve()
    for f in folders:
        fr = Path(f).resolve()
        if out_r == fr or fr in out_r.parents:
            raise SystemExit(f"Refusing to write output inside an audited folder: {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="audit_data.py",
        description=("Read-only audit of image folders. Prints or writes a markdown report with "
                     "counts and statistics only (no file names unless --show-filenames). Never "
                     "modifies, copies or uploads input files and makes no network calls."),
        epilog=("Example: python -B tools/audit_data.py D:/private/brain_tumor "
                "D:/private/non_brain_tumor --output audit_report.md   "
                "See docs/AUDIT_USAGE.md for what is safe to share."),
    )
    ap.add_argument("folders", nargs="+", help="one or more image folders to audit (for example the "
                                                "brain_tumor and non_brain_tumor folders)")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="markdown report path (default: print to stdout); must be outside the audited folders")
    ap.add_argument("--json", type=Path, default=None,
                    help="optional machine-readable copy of the same statistics")
    ap.add_argument("--thresholds", type=str, default=",".join(map(str, DEFAULT_THRESHOLDS)),
                    help="comma-separated Hamming thresholds for near-duplicates (default: %(default)s)")
    ap.add_argument("--no-recursive", action="store_true", help="only audit files directly in each folder")
    ap.add_argument("--show-filenames", action="store_true",
                    help="PRIVATE: include raw file names of flagged files. The report is then not safe to share.")
    args = ap.parse_args(argv)

    thresholds = tuple(sorted({int(t) for t in args.thresholds.split(",") if t.strip()}))
    if any(t < 0 or t > 64 for t in thresholds):
        ap.error("thresholds must be between 0 and 64")
    for out in (args.output, args.json):
        if out is not None:
            _check_output_location(out, args.folders)

    result = run_audit(args.folders, recursive=not args.no_recursive, thresholds=thresholds,
                       show_filenames=args.show_filenames)
    md = render_markdown(result)
    if args.output is None:
        sys.stdout.write(md)
    else:
        args.output.write_text(md, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    if args.json is not None:
        args.json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"JSON written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
