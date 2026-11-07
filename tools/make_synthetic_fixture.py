#!/usr/bin/env python
"""Generate a small, deterministic, obviously artificial image fixture.

The fixture mimics only the folder layout expected by the baseline code
(<out>/brain_tumor/ and <out>/non_brain_tumor/). Every image is built from
procedural shapes (ellipses, blobs, gradients, sinusoidal texture, noise) and
carries a "SYN" mark in one corner. Nothing here is derived from real scans,
and none of it resembles clinical material.

Planted cases, so that tools/audit_data.py has something to find:
  * exact duplicates     : 3 byte-identical copies (2 inside brain_tumor,
                           1 copied across into non_brain_tumor)
  * near-duplicates      : 3 altered copies (1 px shift, additive noise,
                           JPEG re-encode), all in brain_tumor
  * non-square images    : 3
  * unreadable file      : 1 (.png with a valid signature then garbage)
  * RGB image            : 1 (true colour, tinted)
  * non-image file       : 1 (.txt)
  * identifier-like names: 2 (fake keyword and date patterns, no real data)

A JSON manifest describing the planted cases is written to the fixture root
(outside the class folders). Use --help for options.

The output directory must be new, empty, or a previous fixture (detected by
its manifest). The script refuses to overwrite anything else.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MANIFEST_NAME = "fixture_manifest.json"
CLASSES = ("brain_tumor", "non_brain_tumor")


def _texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Background: linear gradient plus sinusoidal texture, values in [0, 1]."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(angle) * xx / w + np.sin(angle) * yy / h)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    tex = np.zeros((h, w), np.float32)
    for _ in range(3):
        fx, fy = rng.uniform(0.5, 6, size=2)
        ph = rng.uniform(0, 2 * np.pi)
        tex += np.sin(2 * np.pi * (fx * xx / w + fy * yy / h) + ph)
    # Large soft bright/dark patches add varied low-frequency structure.
    for _ in range(int(rng.integers(2, 5))):
        py, px = rng.uniform(0, h), rng.uniform(0, w)
        sig = rng.uniform(0.1, 0.3) * min(h, w)
        tex += rng.choice([-1.5, 1.5]) * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sig ** 2))
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-6)
    return rng.uniform(0.1, 0.4) * grad + rng.uniform(0.3, 0.5) * tex


def _ellipse_mask(h, w, cy, cx, ry, rx, theta):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y, x = yy - cy, xx - cx
    c, s = np.cos(theta), np.sin(theta)
    xr, yr = c * x + s * y, -s * x + c * y
    return ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0


def make_image(rng: np.random.Generator, h: int, w: int, with_blob: bool) -> np.ndarray:
    """Return a uint8 grayscale procedural image of shape (h, w)."""
    img = _texture(rng, h, w)
    # Large ellipse "shell" with a random offset, size and rotation.
    cy = h / 2 + rng.uniform(-0.15, 0.15) * h
    cx = w / 2 + rng.uniform(-0.15, 0.15) * w
    ry = rng.uniform(0.25, 0.42) * h
    rx = rng.uniform(0.25, 0.42) * w
    theta = rng.uniform(0, np.pi)
    shell = _ellipse_mask(h, w, cy, cx, ry, rx, theta)
    img = np.where(shell, img * 0.5 + rng.uniform(0.25, 0.5), img)
    # Random small ellipses inside.
    for _ in range(int(rng.integers(1, 5))):
        m = _ellipse_mask(h, w, cy + rng.uniform(-0.5, 0.5) * ry, cx + rng.uniform(-0.5, 0.5) * rx,
                          rng.uniform(0.05, 0.15) * h, rng.uniform(0.05, 0.15) * w, rng.uniform(0, np.pi))
        img = np.where(m & shell, img + rng.uniform(-0.3, 0.3), img)
    if with_blob:
        # One bright Gaussian blob, the only systematic class difference.
        by = cy + rng.uniform(-0.5, 0.5) * ry
        bx = cx + rng.uniform(-0.5, 0.5) * rx
        sig = rng.uniform(0.04, 0.1) * min(h, w)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = img + 0.6 * np.exp(-((yy - by) ** 2 + (xx - bx) ** 2) / (2 * sig ** 2))
    img = img + rng.normal(0, 0.03, size=(h, w))
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return _stamp(out)


def _stamp(arr: np.ndarray) -> np.ndarray:
    """Draw a small 'SYN' mark in the top-left corner so images look artificial."""
    pil = Image.fromarray(arr, mode="L")
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 21, 10], fill=0)
    d.text((1, 0), "SYN", fill=255)
    return np.asarray(pil)


def _save(arr_or_img, path: Path, fmt: str, **kw) -> None:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img.save(path, format=fmt, **kw)


def _prepare_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not (out / MANIFEST_NAME).is_file():
            sys.exit(f"Refusing to write into non-empty directory without a fixture manifest: {out}")
        if not overwrite:
            sys.exit(f"Fixture already exists at {out}. Pass --overwrite to regenerate it.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def generate(out: Path, per_class: int = 150, seed: int = 1234, overwrite: bool = False) -> dict:
    if per_class < 12:
        raise ValueError("per_class must be at least 12 so all planted cases fit")
    _prepare_out(out, overwrite)
    rng = np.random.default_rng(seed)
    dirs = {c: out / c for c in CLASSES}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    base = {c: [] for c in CLASSES}  # list of (path, array)
    for c in CLASSES:
        for i in range(per_class):
            side = 256 if i % 10 == 9 else 128  # small size variety
            arr = make_image(rng, side, side, with_blob=(c == "brain_tumor"))
            fmt, ext = ("JPEG", "jpg") if i % 5 == 4 else ("PNG", "png")
            p = dirs[c] / f"img_{i:04d}.{ext}"
            _save(arr, p, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
            base[c].append((p, arr))

    bt, nbt = dirs["brain_tumor"], dirs["non_brain_tumor"]
    bt_png = [(p, a) for p, a in base["brain_tumor"] if p.suffix == ".png" and a.shape == (128, 128)]

    # Exact duplicates: originals bt_png[0], bt_png[1] (within folder), bt_png[2] (cross folder).
    shutil.copyfile(bt_png[0][0], bt / "dup_copy_a.png")
    shutil.copyfile(bt_png[1][0], bt / "dup_copy_b.png")
    shutil.copyfile(bt_png[2][0], nbt / "dup_cross_copy.png")

    # Near-duplicates from distinct originals bt_png[3..5].
    a = bt_png[3][1]
    shifted = np.roll(a, shift=(1, 1), axis=(0, 1))
    _save(shifted, bt / "near_shift.png", "PNG")
    b = bt_png[4][1].astype(np.int16)
    noisy = np.clip(b + rng.normal(0, 4, size=b.shape), 0, 255).astype(np.uint8)
    _save(noisy, bt / "near_noise.png", "PNG")
    _save(bt_png[5][1], bt / "near_jpeg.jpg", "JPEG", quality=70)

    # Non-square images (fresh content).
    for name, (h, w) in {"wide_0000.png": (128, 160), "tall_0000.png": (200, 128),
                         "small_rect_0000.png": (64, 96)}.items():
        _save(make_image(rng, h, w, with_blob=True), bt / name, "PNG")

    # Unreadable file: PNG signature followed by garbage.
    (bt / "corrupt_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n" + rng.bytes(200))

    # True-colour RGB image (fresh content, tinted channels).
    g = make_image(rng, 128, 128, with_blob=False).astype(np.float32)
    rgb = np.stack([g, g * 0.6, np.clip(g * 0.3 + 90, 0, 255)], axis=-1).astype(np.uint8)
    _save(Image.fromarray(rgb, mode="RGB"), nbt / "color_0000.png", "PNG")

    # Non-image file.
    (nbt / "stray_notes.txt").write_text("Synthetic fixture: this is not an image.\n", encoding="utf-8")

    # Identifier-like names (fake). Renames two existing base images in place.
    renames = [(base["non_brain_tumor"][0][0], nbt / "patient_00000_scan.png"),
               (base["non_brain_tumor"][1][0], nbt / "19000101_fixture.png")]
    for src, dst in renames:
        src.rename(dst)

    manifest = {
        "description": "Synthetic procedural fixture. Not real data. Not medical images.",
        "seed": seed,
        "per_class_base_images": per_class,
        "planted": {
            "exact_duplicate_groups": 3,
            "exact_duplicate_extra_copies": 3,
            "exact_duplicate_cross_folder_groups": 1,
            "near_duplicate_pairs": 3,
            "non_square_images": 3,
            "unreadable_files": 1,
            "rgb_true_color_images": 1,
            "non_image_files": 1,
            "identifier_like_names": {"keyword": 1, "date_like": 1},
        },
        "files": {c: sorted(p.name for p in dirs[c].iterdir()) for c in CLASSES},
    }
    manifest["counts"] = {c: len(v) for c, v in manifest["files"].items()}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate a deterministic synthetic image fixture (procedural shapes, "
                    "not real data) with planted audit cases.")
    ap.add_argument("--out", type=Path, default=Path("fixtures/synthetic_raw"),
                    help="output directory (default: fixtures/synthetic_raw, gitignored)")
    ap.add_argument("--per-class", type=int, default=150,
                    help="base images per class folder before planted extras (default: 150)")
    ap.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate if a previous fixture (with manifest) exists at --out")
    args = ap.parse_args(argv)
    m = generate(args.out, args.per_class, args.seed, args.overwrite)
    print(f"Synthetic fixture written to {args.out}: " +
          ", ".join(f"{c}={n} files" for c, n in m["counts"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
