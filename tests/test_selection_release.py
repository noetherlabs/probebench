"""Trust-boundary tests for the fixed classical-selection reference release."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_selection_release", ROOT / "scripts" / "verify_selection_release.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path.write_bytes(json_bytes(value))


def packet_inventory(root):
    return {path.relative_to(root).as_posix(): verifier.digest(path.read_bytes())
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "packet-manifest.json"}


def rehash_packet(root):
    write_json(root / "packet-manifest.json", {
        "schema": 1, "purpose": verifier.PACKET_PURPOSE, "files": packet_inventory(root)})


def release_identity():
    return {"schema": 1, "purpose": verifier.RELEASE_PURPOSE,
            "archive_manifest_sha256": verifier.NATIVE_MANIFEST_SHA256,
            "source_inventory_sha256": verifier.SOURCE_INVENTORY_SHA256}


@pytest.fixture
def synthetic_packet(tmp_path, monkeypatch):
    """Small byte-level fixture; no synthetic source is ever imported."""
    root = tmp_path.resolve() / "packet"
    root.mkdir()
    source_name = "src/probebench/__init__.py"
    source_path = root / "source" / source_name
    source_path.parent.mkdir(parents=True)
    source_path.write_text("raise RuntimeError('untrusted source was executed')\n")
    source_hashes = {source_name: verifier.digest(source_path.read_bytes())}
    write_json(root / "study.json", {"configuration": {}, "source_sha256": source_hashes})
    for name in verifier.ORIGINAL_JSON[1:]:
        write_json(root / name, {})
    native = {"schema": 1, "purpose": verifier.NATIVE_PURPOSE,
              "episode_count": 4032, "exposure_count": 30912,
              "source_sha256": source_hashes, "files": packet_inventory(root)}
    write_json(root / "archive-manifest.json", native)
    monkeypatch.setattr(verifier, "NATIVE_MANIFEST_SHA256",
                        verifier.digest((root / "archive-manifest.json").read_bytes()))
    monkeypatch.setattr(verifier, "SOURCE_INVENTORY_SHA256", verifier.canonical_hash(source_hashes))
    write_json(root / "release.json", release_identity())
    rehash_packet(root)
    return root


def test_byte_validation_does_not_import_packet_source(synthetic_packet):
    loaded_before = {key for key in sys.modules if key.startswith("probebench")}
    checked = verifier.validate_packet(synthetic_packet)
    assert checked["native"]["episode_count"] == 4032
    assert {key for key in sys.modules if key.startswith("probebench")} == loaded_before
    assert not list(synthetic_packet.rglob("__pycache__"))


@pytest.mark.parametrize("name", verifier.ORIGINAL_JSON)
def test_rehashed_packet_cannot_replace_original_metadata(synthetic_packet, name):
    (synthetic_packet / name).write_bytes(b"{}\n" + b" " * 3)
    rehash_packet(synthetic_packet)
    with pytest.raises(ValueError, match="Original archive file changed"):
        verifier.validate_packet(synthetic_packet)


def test_rehashed_packet_cannot_replace_source(synthetic_packet):
    source = synthetic_packet / "source/src/probebench/__init__.py"
    source.write_text("raise SystemExit('changed code must not execute')\n")
    rehash_packet(synthetic_packet)
    with pytest.raises(ValueError, match="Original archive file changed"):
        verifier.validate_packet(synthetic_packet)


def test_packet_manifest_cannot_repin_native_manifest(synthetic_packet):
    native = json.loads((synthetic_packet / "archive-manifest.json").read_text())
    native["episode_count"] = 1
    write_json(synthetic_packet / "archive-manifest.json", native)
    rehash_packet(synthetic_packet)
    with pytest.raises(ValueError, match="independent release fingerprint"):
        verifier.validate_packet(synthetic_packet)


def test_source_inventory_has_independent_pin(synthetic_packet, monkeypatch):
    monkeypatch.setattr(verifier, "SOURCE_INVENTORY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="independent source fingerprint"):
        verifier.validate_packet(synthetic_packet)


@pytest.mark.parametrize("name", ["../outside", "/outside", "source/../outside",
                                  "source\\outside", "source//outside", "./outside"])
def test_unsafe_manifest_names_are_rejected(synthetic_packet, name):
    manifest = json.loads((synthetic_packet / "packet-manifest.json").read_text())
    manifest["files"][name] = "0" * 64
    write_json(synthetic_packet / "packet-manifest.json", manifest)
    with pytest.raises(ValueError, match="Unsafe relative artifact path"):
        verifier.validate_packet(synthetic_packet)


@pytest.mark.parametrize("entry", ["file", "directory", "manifest", "root"])
def test_symbolic_links_are_rejected(synthetic_packet, entry, tmp_path):
    outside = tmp_path.resolve() / "outside"
    if entry == "root":
        outside.symlink_to(synthetic_packet, target_is_directory=True)
        target = outside
    else:
        name = {"file": "summary.json", "directory": "source",
                "manifest": "packet-manifest.json"}[entry]
        source = synthetic_packet / name
        source.rename(outside)
        source.symlink_to(outside, target_is_directory=entry == "directory")
        target = synthetic_packet
    with pytest.raises((ValueError, OSError)):
        verifier.validate_packet(target)


def test_fifo_manifest_is_rejected_without_reading(synthetic_packet):
    manifest = synthetic_packet / "packet-manifest.json"
    manifest.unlink()
    os.mkfifo(manifest)
    with pytest.raises(ValueError, match="bounded regular file"):
        verifier.validate_packet(synthetic_packet)


@pytest.mark.parametrize("mutation", ["missing_file", "unlisted_file", "missing_required",
                                      "extra_source", "self_entry", "bad_hash", "empty_inventory"])
def test_inventory_must_be_exact_and_well_formed(synthetic_packet, mutation):
    manifest_path = synthetic_packet / "packet-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "missing_file":
        (synthetic_packet / "summary.json").unlink()
    elif mutation == "unlisted_file":
        (synthetic_packet / "extra.txt").write_text("unlisted")
    elif mutation == "missing_required":
        (synthetic_packet / "summary.json").unlink()
        del manifest["files"]["summary.json"]
    elif mutation == "extra_source":
        (synthetic_packet / "source/untrusted.py").write_text("pass\n")
        manifest["files"] = packet_inventory(synthetic_packet)
    elif mutation == "self_entry":
        manifest["files"]["packet-manifest.json"] = "0" * 64
    elif mutation == "bad_hash":
        manifest["files"]["summary.json"] = "not a digest"
    elif mutation == "empty_inventory":
        manifest["files"] = {}
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        verifier.validate_packet(synthetic_packet)


@pytest.mark.parametrize("content", [b"[]", b'{"schema":1,"schema":1}',
                                     b'{"schema":true}', b'{"schema":NaN}'])
def test_malformed_manifest_is_rejected(synthetic_packet, content):
    (synthetic_packet / "packet-manifest.json").write_bytes(content)
    with pytest.raises(ValueError):
        verifier.validate_packet(synthetic_packet)


def test_release_metadata_cannot_rebind_selection(synthetic_packet):
    release = release_identity()
    release["archive_manifest_sha256"] = "0" * 64
    write_json(synthetic_packet / "release.json", release)
    rehash_packet(synthetic_packet)
    with pytest.raises(ValueError, match="Release identity"):
        verifier.validate_packet(synthetic_packet)


@pytest.fixture(scope="module")
def reference_packet(tmp_path_factory):
    """Copy only the published compact metadata, never camera/case archives."""
    original = ROOT / "results/classical-selection-v1-20261016"
    root = tmp_path_factory.mktemp("selection-reference").resolve()
    for name in (*verifier.ORIGINAL_JSON, "archive-manifest.json"):
        shutil.copyfile(original / name, root / name)
    shutil.copytree(original / "source", root / "source")
    write_json(root / "release.json", release_identity())
    rehash_packet(root)
    return root


def test_reference_packet_reduces_in_isolation_without_camera_replay(reference_packet, monkeypatch, tmp_path):
    # A hostile import path cannot shadow the independently verified package.
    shadow = tmp_path.resolve() / "shadow"
    shadow.mkdir()
    (shadow / "probebench.py").write_text("raise SystemExit('untrusted import path')\n")
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    # System temporary directories may use aliases, such as /var on macOS.
    temporary = tmp_path.resolve() / "temporary"
    temporary.mkdir()
    temporary_alias = tmp_path.resolve() / "temporary-alias"
    temporary_alias.symlink_to(temporary, target_is_directory=True)
    monkeypatch.setenv("TMPDIR", str(temporary_alias))
    before = packet_inventory(reference_packet)
    result = verifier.verify(reference_packet)
    assert result["packet_verified"] is True
    assert result["selection_reduction_verified"] is True
    assert result["full_camera_replay_verified"] is False
    assert result["episodes"] == 4032
    assert result["exposures"] == 30912
    assert result["candidates"] == 14
    assert result["selected_families"] == 6
    assert "camera data not checked" in result["scope"]
    assert "not held-out performance" in result["limitations"]
    assert packet_inventory(reference_packet) == before
    assert not list(reference_packet.rglob("__pycache__"))


def test_compact_packet_cannot_be_reported_as_full_archive(reference_packet):
    with pytest.raises(ValueError, match="manifest.json"):
        verifier.verify(reference_packet, archive=reference_packet)


def test_wrong_archive_identity_is_rejected_before_replay(reference_packet, tmp_path):
    archive = tmp_path.resolve() / "archive"
    archive.mkdir()
    write_json(archive / "manifest.json", {"purpose": verifier.NATIVE_PURPOSE})
    with pytest.raises(ValueError, match="Full archive manifest differs"):
        verifier.verify(reference_packet, archive=archive)


def test_changed_source_cannot_execute_in_worker(reference_packet, tmp_path):
    packet = tmp_path.resolve() / "packet"
    shutil.copytree(reference_packet, packet)
    sentinel = tmp_path.resolve() / "executed"
    (packet / "source/src/probebench/__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n")
    rehash_packet(packet)
    with pytest.raises(ValueError, match="Original archive file changed"):
        verifier.verify(packet)
    assert not sentinel.exists()
