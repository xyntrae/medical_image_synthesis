"""Tests for tools/audit_data.py, run against the synthetic fixture only.

Run from the repository root:
    python -B -m pytest tools/tests -q -p no:cacheprovider
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
AUDIT = TOOLS / "audit_data.py"
FIXTURE = TOOLS / "make_synthetic_fixture.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_data", AUDIT)
fixgen = _load("make_synthetic_fixture", FIXTURE)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fx") / "synthetic_raw"
    fixgen.generate(out, per_class=150, seed=1234)
    return out


@pytest.fixture(scope="session")
def result(fixture_dir):
    return audit.run_audit([fixture_dir / "brain_tumor", fixture_dir / "non_brain_tumor"])


def _tree_snapshot(root: Path):
    snap = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        if p.is_file():
            snap[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns,
                                              hashlib.sha256(p.read_bytes()).hexdigest())
        else:
            snap[str(p.relative_to(root))] = ("dir", st.st_mtime_ns)
    return snap


def test_file_status_counts(result):
    bt = result["per_folder"]["brain_tumor"]
    nbt = result["per_folder"]["non_brain_tumor"]
    # 150 base + 2 dup copies + 3 near + 3 non-square + 1 corrupt
    assert bt["files_total"] == 159
    assert bt["status"]["image"] == 158
    assert bt["status"]["unreadable"] == 1
    assert bt["status"].get("non_image", 0) == 0
    # 150 base + 1 cross dup + 1 rgb + 1 txt
    assert nbt["files_total"] == 153
    assert nbt["status"]["image"] == 152
    assert nbt["status"]["non_image"] == 1
    assert nbt["status"].get("unreadable", 0) == 0
    assert bt["unreadable_error_types"] == {"UnidentifiedImageError": 1}
    assert nbt["non_image_ext"] == {".txt": 1}


def test_exact_duplicates(result):
    e = result["exact_duplicates"]
    assert e["groups"] == 3
    assert e["extra_copies"] == 3
    assert e["files_in_groups"] == 6
    assert e["cross_folder_groups"] == 1
    assert e["image_groups"] == 3


@pytest.mark.parametrize("t", [4, 6, 8])
def test_near_duplicates(result, t):
    row = next(r for r in result["near_duplicates"] if r["threshold"] == t)
    assert row["pairs"] == 3
    assert row["clusters"] == 3
    assert row["images_in_clusters"] == 6
    assert row["largest_cluster"] == 2
    assert row["cross_folder_clusters"] == 0


def test_near_duplicate_exact_copies_excluded(result):
    # At threshold 0 only perceptually identical but byte-different files count
    # (noise and JPEG re-encode); byte-identical copies are not counted again.
    row = next(r for r in result["near_duplicates"] if r["threshold"] == 0)
    assert row["pairs"] == 2


def test_rgb_and_non_square(result):
    bt = result["per_folder"]["brain_tumor"]
    nbt = result["per_folder"]["non_brain_tumor"]
    assert nbt["color_class"]["color"] == 1
    assert nbt["mode"]["RGB"] == 1
    assert bt["color_class"].get("color", 0) == 0
    assert bt["non_square"] + nbt["non_square"] == 3
    assert bt["size"]["160x128"] == 1 and bt["size"]["128x200"] == 1 and bt["size"]["96x64"] == 1


def test_identifier_patterns(result):
    ip = result["identifier_patterns"]
    assert ip["keyword (patient/subject/mrn/name/dob/case)"] == 1
    assert ip["date-like (YYYYMMDD or YYYY-MM-DD)"] == 1
    assert result["identifier_any"] == 2
    assert result["filenames"] is None


def test_report_contains_no_file_names(result, fixture_dir):
    md = audit.render_markdown(result)
    manifest = json.loads((fixture_dir / fixgen.MANIFEST_NAME).read_text())
    names = [n for files in manifest["files"].values() for n in files]
    leaked = [n for n in names if n in md or Path(n).stem in md]
    assert leaked == []
    assert str(fixture_dir) not in md
    assert "contains only counts and statistics" in md


def test_show_filenames_flag(fixture_dir):
    r = audit.run_audit([fixture_dir / "brain_tumor", fixture_dir / "non_brain_tumor"], show_filenames=True)
    md = audit.render_markdown(r)
    assert "patient_00000_scan.png" in md
    assert "corrupt_0000.png" in md
    assert "DO NOT SHARE" in md


def test_metadata_file_detection(tmp_path):
    from PIL import Image
    d = tmp_path / "parent" / "imgs"
    d.mkdir(parents=True)
    Image.new("L", (8, 8), 7).save(d / "a.png")
    (d / "labels.csv").write_text("x\n")
    (tmp_path / "parent" / "metadata.json").write_text("{}")
    r = audit.run_audit([d])
    assert r["metadata_files"] == {"imgs: .csv": 1, "(parent): .json": 1}


def test_fixture_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fixgen.generate(a, per_class=20, seed=7)
    fixgen.generate(b, per_class=20, seed=7)
    sa = {k: v[2] for k, v in _tree_snapshot(a).items() if v[0] != "dir"}
    sb = {k: v[2] for k, v in _tree_snapshot(b).items() if v[0] != "dir"}
    assert sa == sb and len(sa) > 40


def test_fixture_refuses_foreign_directory(tmp_path):
    (tmp_path / "keep.txt").write_text("not a fixture")
    with pytest.raises(SystemExit):
        fixgen.generate(tmp_path, per_class=20, seed=1)
    assert (tmp_path / "keep.txt").read_text() == "not a fixture"


def test_refuses_output_inside_audited_folder(fixture_dir):
    proc = subprocess.run([sys.executable, "-B", str(AUDIT), str(fixture_dir / "brain_tumor"),
                           "-o", str(fixture_dir / "brain_tumor" / "r.md")], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "Refusing" in proc.stderr
    assert not (fixture_dir / "brain_tumor" / "r.md").exists()


def test_help_text():
    proc = subprocess.run([sys.executable, "-B", str(AUDIT), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Read-only audit" in proc.stdout and "--show-filenames" in proc.stdout


GUARD = r"""
import os, sys, runpy
allowed = {os.path.normcase(os.path.abspath(p)) for p in sys.argv[1:3]}
violations = []
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
BLOCK = {"os.remove", "os.unlink", "os.rename", "os.replace", "os.mkdir", "os.rmdir", "os.symlink",
         "os.link", "os.truncate", "os.chmod", "os.utime", "shutil.copyfile", "shutil.copymode",
         "shutil.copystat", "shutil.move", "shutil.rmtree", "socket.connect", "socket.bind",
         "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto", "urllib.Request",
         "http.client.connect", "subprocess.Popen", "os.system", "os.startfile"}
def hook(event, args):
    if event == "open":
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                  (mode is None and isinstance(flags, int) and flags & WRITE_FLAGS)
        if writing and isinstance(path, (str, bytes, os.PathLike)):
            p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
            if p not in allowed:
                violations.append(("open-write", p))
                raise PermissionError("guard: write blocked " + p)
    elif event in BLOCK:
        violations.append((event, repr(args)[:200]))
        raise PermissionError("guard: blocked " + event)
sys.addaudithook(hook)
audit_path = sys.argv[3]
sys.argv = [audit_path] + sys.argv[4:]
try:
    runpy.run_path(audit_path, run_name="__main__")
except SystemExit as e:
    code = e.code
except BaseException as e:
    sys.stderr.write("GUARD_EXCEPTION %s: %s\n" % (type(e).__name__, e))
    code = 4
else:
    code = 0
sys.stderr.write("GUARD_VIOLATIONS=%d\n" % len(violations))
for v in violations:
    sys.stderr.write("GUARD %r\n" % (v,))
sys.exit(code or (3 if violations else 0))
"""


def test_audit_writes_only_its_output_and_no_network(fixture_dir, tmp_path):
    before = _tree_snapshot(fixture_dir.parent)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    outdir = tmp_path / "out"
    outdir.mkdir()
    md, js = outdir / "report.md", outdir / "report.json"
    proc = subprocess.run(
        [sys.executable, "-B", "-c", GUARD, str(md), str(js), str(AUDIT),
         str(fixture_dir / "brain_tumor"), str(fixture_dir / "non_brain_tumor"),
         "-o", str(md), "--json", str(js)],
        cwd=cwd, capture_output=True, text=True)
    print(proc.stderr)
    assert proc.returncode == 0, proc.stderr
    assert "GUARD_VIOLATIONS=0" in proc.stderr
    assert _tree_snapshot(fixture_dir.parent) == before, "fixture tree changed"
    assert list(cwd.iterdir()) == []
    assert sorted(p.name for p in outdir.iterdir()) == ["report.json", "report.md"]
    assert json.loads(js.read_text())["exact_duplicates"]["groups"] == 3


def test_guard_actually_catches_writes(tmp_path):
    """Sanity check that the guard used above detects a forbidden write."""
    bad = tmp_path / "bad.py"
    bad.write_text("open(r'%s', 'w').write('x')\n" % (tmp_path / "forbidden.txt"))
    proc = subprocess.run([sys.executable, "-B", "-c", GUARD, str(tmp_path / "a"), str(tmp_path / "b"),
                           str(bad)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "GUARD_VIOLATIONS=1" in proc.stderr
    assert not (tmp_path / "forbidden.txt").exists()
