"""Small-archive export tests; scientific replay is isolated at its trusted boundary."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_selection_imagefolder.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC = importlib.util.spec_from_file_location("selection_imagefolder_export", SCRIPT)
    exporter = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(exporter)
finally:
    sys.path.pop(0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def snapshot(root):
    return {path.relative_to(root).as_posix(): digest(path.read_bytes())
            for path in sorted(root.rglob("*")) if path.is_file()}


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def archive_fixture(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    packet, archive, output = (root / name for name in ("packet", "archive", "export"))
    packet.mkdir()
    (packet / "fixture.txt").write_text("Synthetic release boundary.\n")
    archive.mkdir()
    (archive / ".lock").touch()
    (archive / "cases").mkdir()
    (archive / "frames").mkdir()
    study = {"configuration": {"cohort": {"specification": {
        "optics": {"pupil_pixels": 2, "oversampling": 2}}}},
        "source_sha256": {"src/probebench/__init__.py": digest(b"fixture")}}
    settings = {"schema": 1, "purpose": "classical_parameter_selection_v1", "families": {
        "random_probe_v2": {"candidate_id": "random-small", "policy": {}},
        "coordinate_search_v2": {"candidate_id": "coordinate", "policy": {}}}}
    designs = [
        ("random_probe_v2", "random-small", "base-a", "stratum-a", 0, 0, 100, 2),
        ("random_probe_v2", "random-large", "base-a", "stratum-a", 0, 0, 100, 2),
        ("random_probe_v2", "random-small", "base-a", "stratum-a", 1, 1, 200, 1),
        ("coordinate_search_v2", "coordinate", "base-b", "stratum-b", 0, 0, 200, 2),
    ]
    identity_keys = ("family", "candidate_id", "base_wavefront_id", "stratum_id",
                     "member", "replicate", "photons_per_exposure", "exposure_budget")
    rows, records, arrays = [], {}, {}
    for number, values in enumerate(designs):
        identity = {"id": "case_" + f"{number + 1:064x}", **dict(zip(identity_keys, values))}
        rows.append(identity)
        case = {**identity, "truth_opd_nm": [13, 0, 0, 0, 0, 0],
                "detector_seed": 71 + number, "policy_seed": 91 + number}
        trajectory = []
        frame_root = archive / "frames" / identity["id"]
        frame_root.mkdir()
        for index in range(identity["exposure_budget"]):
            measured = {"index": index, "expected_photons": identity["photons_per_exposure"],
                "cumulative_expected_photons": (index + 1) * identity["photons_per_exposure"],
                "applied_opd_nm": [float(index), 0, 0, 0, 0, 0],
                "residual_wfe_nm": 12.5 - index, "strehl_peak_ratio": 0.8 + 0.01 * index}
            trajectory.append(measured)
            image = np.arange(16, dtype=np.float64).reshape(4, 4) - 5.5 - number - index
            arrays[(identity["id"], index)] = image.copy()
            camera = {"observation": measured, "image_sha256": digest(image.tobytes())}
            camera_json = json.dumps(camera, sort_keys=True, separators=(",", ":"))
            np.savez_compressed(frame_root / f"frame_{index:04d}.npz", image_electrons=image,
                record_json=np.array(camera_json), record_sha256=np.array(digest(camera_json.encode())))
        # Include both an unmeasured final command and one observed before the last frame.
        observed_earlier = number == 1
        record = {"case": case, "study_sha256": digest(json_bytes(study)),
            "result": {"trajectory": trajectory, "exposures": identity["exposure_budget"],
                "hidden_opd_nm": case["truth_opd_nm"],
                "final_applied_opd_nm": [0 if observed_earlier else 19, 0, 0, 0, 0, 0],
                "final_command_has_camera_observation": observed_earlier,
                "final_command_observation_indices": [0] if observed_earlier else [],
                "final_command_is_last_observation": False,
                "final": {"residual_wfe_nm": 7.0}, "fit_records": []},
            "receipt": {"retained": True}}
        records[identity["id"]] = record
        (archive / "cases" / (identity["id"] + ".json")).write_bytes(json_bytes(record))
    originals = {"study.json": json_bytes(study), "summary.json": json_bytes(rows),
                 "selected-settings.json": json_bytes(settings)}
    for name, data in originals.items():
        (archive / name).write_bytes(data)
    native = {"schema": 1, "purpose": "classical_parameter_selection_v1",
        "episode_count": len(rows), "exposure_count": len(arrays),
        "source_sha256": study["source_sha256"], "files": snapshot(archive)}
    native_bytes = json_bytes(native)
    (archive / "manifest.json").write_bytes(native_bytes)
    checked = {"bytes": {"archive-manifest.json": native_bytes, **originals},
               "native": native, "study": study, "packet_manifest_sha256": "a" * 64}
    monkeypatch.setattr(exporter.release, "NATIVE_MANIFEST_SHA256", digest(native_bytes))
    monkeypatch.setattr(exporter.release, "SOURCE_INVENTORY_SHA256",
                        exporter.release.canonical_hash(study["source_sha256"]))
    verification = {"schema": 1, "packet_verified": True, "selection_reduction_verified": True,
        "full_camera_replay_verified": True, "archive_manifest_sha256": digest(native_bytes),
        "source_inventory_sha256": exporter.release.canonical_hash(study["source_sha256"]),
        "packet_manifest_sha256": checked["packet_manifest_sha256"],
        "episodes": len(rows), "exposures": len(arrays), "candidates": 3, "selected_families": 2}
    calls = []

    def verify(packet_arg, archive=None):
        assert Path(packet_arg) == packet
        assert Path(archive) == root / "archive"
        assert not output.exists()
        calls.append("verify")
        return verification.copy()

    def validate(packet_arg):
        assert Path(packet_arg) == packet
        assert calls == ["verify"]
        calls.append("validate")
        return checked

    monkeypatch.setattr(exporter.release, "verify", verify)
    monkeypatch.setattr(exporter.release, "validate_packet", validate)
    monkeypatch.setattr(exporter.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=100 * 1024**3))
    return SimpleNamespace(packet=packet, archive=archive, output=output, native=native,
        checked=checked, verification=verification, rows=rows, records=records, arrays=arrays,
        settings=settings, study=study, calls=calls, before=snapshot(archive),
        packet_before=snapshot(packet))


def run(fixture):
    return exporter.export(fixture.packet, fixture.archive, fixture.output)


def test_export_preserves_all_real_exposures_signed_arrays_and_complete_records(archive_fixture):
    item = archive_fixture
    manifest = run(item)
    assert item.calls == ["verify", "validate"]
    assert snapshot(item.archive) == item.before
    assert snapshot(item.packet) == item.packet_before
    assert manifest == json.loads((item.output / "manifest.json").read_text())
    assert manifest["episode_count"] == len(item.rows)
    assert manifest["exposure_count"] == len(item.arrays)
    assert manifest["candidate_count"] == 3
    assert manifest["independent_base_count"] == 2
    assert manifest["input_manifest_sha256"] == digest(item.checked["bytes"]["archive-manifest.json"])
    assert manifest["packet_manifest_sha256"] == item.checked["packet_manifest_sha256"]
    actual_files = snapshot(item.output)
    del actual_files["manifest.json"]
    assert manifest["files"] == actual_files
    assert json.loads((item.output / "configuration.json").read_text()) == item.study["configuration"]
    assert json.loads((item.output / "selected-settings.json").read_text()) == item.settings
    provenance = json.loads((item.output / "provenance.json").read_text())
    assert provenance["verification"] == item.verification
    assert provenance["display"] == {
        "transform": "log1p(max(electrons,0))/log1p(photons_per_exposure)",
        "minimum": 0, "maximum": 1, "colormap": "magma", "origin": "lower",
        "raw_dtype": "float64", "raw_units": "electrons", "raw_shape": [4, 4]}
    for name in ("LICENSE", "NOTICE"):
        assert (item.output / name).read_bytes() == (ROOT / name).read_bytes()
    assert (item.output / "README.md").is_file()
    rows = json_lines(item.output / "images/metadata.jsonl")
    assert len(rows) == len(item.arrays)
    keys = {"episode_id", "candidate_id", "family", "photons_per_exposure", "exposure_budget",
        "base_wavefront_id", "member", "stratum_id", "replicate", "exposure_index", "file_name",
        "raw_array", "raw_array_sha256", "source_frame_sha256", "selected_reference", "split",
        "applied_opd_nm", "expected_photons", "cumulative_expected_photons", "evaluator_wfe_nm",
        "evaluator_strehl_peak_ratio"}
    assert {(row["episode_id"], row["exposure_index"]) for row in rows} == set(item.arrays)
    assert {row["candidate_id"] for row in rows} == {"random-small", "random-large", "coordinate"}
    assert any(not row["selected_reference"] for row in rows)
    for row in rows:
        assert set(row) == keys
        case = item.records[row["episode_id"]]["case"]
        for name in ("candidate_id", "family", "photons_per_exposure", "exposure_budget",
                     "base_wavefront_id", "member", "stratum_id", "replicate"):
            assert row[name] == case[name]
        assert row["split"] == "public_tuning"
        selected = item.settings["families"][case["family"]]["candidate_id"]
        assert row["selected_reference"] is (case["candidate_id"] == selected)
        index = row["exposure_index"]
        identity = f"{case['id']}_frame_{index:04d}"
        assert row["file_name"] == identity + ".png"
        assert row["raw_array"] == f"raw/{identity}.npy"
        raw = item.output / row["raw_array"]
        actual = np.load(raw, allow_pickle=False)
        assert actual.dtype == np.dtype("float64")
        np.testing.assert_array_equal(actual, item.arrays[(case["id"], index)])
        assert actual.min() < 0
        assert row["raw_array_sha256"] == digest(raw.read_bytes())
        frame_name = f"frames/{case['id']}/frame_{index:04d}.npz"
        assert row["source_frame_sha256"] == item.native["files"][frame_name]
        assert (item.output / "images" / row["file_name"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        display = np.log1p(np.maximum(actual, 0)) / np.log1p(case["photons_per_exposure"])
        expected_rgba = exporter.matplotlib.colormaps["magma"](display, bytes=True)[::-1]
        with Image.open(item.output / "images" / row["file_name"]) as rendered:
            np.testing.assert_array_equal(np.asarray(rendered.convert("RGBA")), expected_rgba)
        measured = item.records[case["id"]]["result"]["trajectory"][index]
        for field in ("applied_opd_nm", "expected_photons", "cumulative_expected_photons"):
            assert row[field] == measured[field]
        assert row["evaluator_wfe_nm"] == measured["residual_wfe_nm"]
        assert row["evaluator_strehl_peak_ratio"] == measured["strehl_peak_ratio"]
    episodes = json_lines(item.output / "episodes.jsonl")
    assert episodes == [{"episode_id": row["id"], "record": item.records[row["id"]]} for row in item.rows]
    first, second = (episodes[index]["record"]["result"] for index in (0, 1))
    assert first["final_command_observation_indices"] == []
    assert not first["final_command_has_camera_observation"]
    assert second["final_command_observation_indices"] == [0]
    assert second["final_command_has_camera_observation"]
    assert not second["final_command_is_last_observation"]
    assert len(list((item.output / "images").glob("*.png"))) == len(item.arrays)
    assert len(list((item.output / "raw").glob("*.npy"))) == len(item.arrays)


@pytest.mark.parametrize("destination", ["archive", "packet", "inside_archive", "inside_packet",
                                          "ancestor", "existing", "missing_parent", "linked_parent"])
def test_unsafe_output_is_rejected_before_verification(archive_fixture, destination):
    item = archive_fixture
    if destination in {"archive", "packet"}:
        item.output = getattr(item, destination)
    elif destination.startswith("inside_"):
        item.output = getattr(item, destination.removeprefix("inside_")) / "export"
    elif destination == "ancestor":
        item.output = item.archive.parent
    elif destination == "existing":
        item.output.mkdir()
    elif destination == "missing_parent":
        item.output = item.output / "missing" / "export"
    else:
        link = item.archive.parent / "linked-parent"
        link.symlink_to(item.archive.parent, target_is_directory=True)
        item.output = link / "export"
    with pytest.raises((ValueError, OSError)):
        run(item)
    assert item.calls == []
    assert snapshot(item.archive) == item.before
    assert snapshot(item.packet) == item.packet_before


@pytest.mark.parametrize("source_name", ["packet", "archive"])
def test_parent_traversal_source_alias_cannot_hide_output_overlap(archive_fixture, source_name):
    item = archive_fixture
    source = getattr(item, source_name)
    child = source / "child"
    child.mkdir()
    setattr(item, source_name, child / "..")
    item.output = source / "new-export"
    with pytest.raises(ValueError, match="disjoint"):
        run(item)
    assert item.calls == []
    assert not item.output.exists()
    assert snapshot(source) == (item.packet_before if source_name == "packet" else item.before)


@pytest.mark.parametrize("source_name", ["packet", "archive"])
def test_case_alias_cannot_hide_output_overlap(archive_fixture, source_name):
    item = archive_fixture
    source = getattr(item, source_name)
    alias = source.with_name(source.name.upper())
    if not alias.exists() or not alias.samefile(source):
        pytest.skip("Filesystem does not provide this case-insensitive directory alias")
    item.output = alias / "new-export"
    with pytest.raises(ValueError, match="disjoint"):
        run(item)
    assert item.calls == []
    assert not item.output.exists()
    assert snapshot(source) == (item.packet_before if source_name == "packet" else item.before)


def test_busy_archive_lock_is_nonblocking_and_readonly(archive_fixture):
    item = archive_fixture
    with (item.archive / ".lock").open("rb") as locked:
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((ValueError, OSError)):
            run(item)
    assert item.calls == []
    assert not item.output.exists()
    assert snapshot(item.archive) == item.before


def test_missing_archive_lock_is_never_created(archive_fixture):
    item = archive_fixture
    lock = item.archive / ".lock"
    lock.unlink()
    before = snapshot(item.archive)
    with pytest.raises((ValueError, OSError)):
        run(item)
    assert not lock.exists()
    assert not item.output.exists()
    assert item.calls == []
    assert snapshot(item.archive) == before


def test_export_coexists_with_another_reader_and_readonly_archive(archive_fixture):
    item = archive_fixture
    for path in item.archive.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    with (item.archive / ".lock").open("rb") as locked:
        fcntl.flock(locked, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = run(item)
    assert manifest["exposure_count"] == len(item.arrays)
    assert snapshot(item.archive) == item.before


@pytest.mark.parametrize("entry", ["lock_link", "lock_fifo", "extra_link", "extra_directory_link", "extra_fifo"])
def test_links_and_special_archive_entries_fail_before_verification(archive_fixture, entry):
    item = archive_fixture
    outside = item.archive.parent / "outside"
    if entry == "extra_directory_link":
        outside.mkdir()
    else:
        outside.write_text("outside sentinel")
    path = item.archive / (".lock" if entry.startswith("lock_") else "extra")
    if path.exists():
        path.unlink()
    if entry.endswith("fifo"):
        os.mkfifo(path)
    else:
        path.symlink_to(outside, target_is_directory=entry == "extra_directory_link")
    with pytest.raises((ValueError, OSError)):
        run(item)
    assert item.calls == []
    assert not item.output.exists()


def test_unlisted_regular_archive_file_cannot_enter_export(archive_fixture):
    item = archive_fixture
    (item.archive / "unexpected.json").write_text("{}\n")
    with pytest.raises(ValueError):
        run(item)
    assert not item.output.exists()


def test_failed_full_verification_does_not_create_output(archive_fixture, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("camera replay failed")

    monkeypatch.setattr(exporter.release, "verify", reject)
    with pytest.raises(ValueError, match="camera replay failed"):
        run(archive_fixture)
    assert not archive_fixture.output.exists()
    assert snapshot(archive_fixture.archive) == archive_fixture.before


def test_invalid_packet_is_refused_by_real_isolated_verifier(tmp_path):
    root = tmp_path.resolve()
    packet, archive, output = (root / name for name in ("packet", "archive", "export"))
    packet.mkdir()
    (packet / "packet-manifest.json").write_text("{}\n")
    archive.mkdir()
    (archive / ".lock").touch()
    (archive / "manifest.json").write_text("{}\n")
    before_packet, before_archive = snapshot(packet), snapshot(archive)
    with pytest.raises(ValueError, match="reference-packet manifest"):
        exporter.export(packet, archive, output)
    assert not output.exists()
    assert snapshot(packet) == before_packet
    assert snapshot(archive) == before_archive


@pytest.mark.parametrize("field,value", [
    ("full_camera_replay_verified", False), ("archive_manifest_sha256", "0" * 64),
    ("source_inventory_sha256", "0" * 64), ("packet_manifest_sha256", "0" * 64),
    ("episodes", 3), ("exposures", 6),
])
def test_verification_receipt_cannot_substitute_packet_only_or_other_evidence(archive_fixture, field, value):
    archive_fixture.verification[field] = value
    with pytest.raises(ValueError):
        run(archive_fixture)
    assert not archive_fixture.output.exists()


def test_native_manifest_has_independent_release_pin(archive_fixture, monkeypatch):
    monkeypatch.setattr(exporter.release, "NATIVE_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        run(archive_fixture)
    assert not archive_fixture.output.exists()


def test_native_manifest_bytes_are_rechecked_after_successful_replay(archive_fixture, monkeypatch):
    item = archive_fixture
    original = exporter.release.verify

    def change_native_manifest(*args, **kwargs):
        receipt = original(*args, **kwargs)
        path = item.archive / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        return receipt

    monkeypatch.setattr(exporter.release, "verify", change_native_manifest)
    with pytest.raises(ValueError, match="Native input changed after verification: manifest.json"):
        run(item)
    assert item.calls == ["verify", "validate"]
    assert not item.output.exists()


@pytest.mark.parametrize("entry", ["case", "frame"])
def test_consumed_bytes_are_rehashed_after_full_verification(archive_fixture, monkeypatch, entry):
    item = archive_fixture
    original = exporter.release.verify
    identity = item.rows[0]["id"]
    path = item.archive / (f"cases/{identity}.json" if entry == "case"
                           else f"frames/{identity}/frame_0000.npz")

    def mutate_after_verify(*args, **kwargs):
        receipt = original(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b" ")
        return receipt

    monkeypatch.setattr(exporter.release, "verify", mutate_after_verify)
    with pytest.raises(ValueError):
        run(item)
    assert not (item.output / "manifest.json").exists()


def test_insufficient_space_refuses_before_creating_output(archive_fixture, monkeypatch):
    monkeypatch.setattr(exporter.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=exporter.RESERVE_BYTES - 1))
    with pytest.raises((ValueError, OSError)):
        run(archive_fixture)
    assert not archive_fixture.output.exists()
    assert snapshot(archive_fixture.archive) == archive_fixture.before


def test_space_check_includes_pending_write_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=exporter.RESERVE_BYTES + 9))
    exporter.check_space(tmp_path, required_bytes=9)
    with pytest.raises((ValueError, OSError)):
        exporter.check_space(tmp_path, required_bytes=10)


def test_space_floor_is_checked_again_after_output_creation(archive_fixture, monkeypatch):
    item = archive_fixture
    checks = []

    def available_space(path):
        checks.append(item.output.exists())
        return SimpleNamespace(free=exporter.RESERVE_BYTES - 1 if item.output.exists()
                               else 100 * 1024**3)

    monkeypatch.setattr(exporter.shutil, "disk_usage", available_space)
    with pytest.raises((ValueError, OSError)):
        run(item)
    assert checks[0] is False
    assert any(checks)
    assert item.output.is_dir()
    assert not (item.output / "manifest.json").exists()
    assert snapshot(item.archive) == item.before


def test_interrupted_export_retains_partial_output_without_completion_manifest(archive_fixture, monkeypatch):
    save = exporter.np.save
    calls = []

    def fail_second_save(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("synthetic disk write failure")
        return save(*args, **kwargs)

    monkeypatch.setattr(exporter.np, "save", fail_second_save)
    with pytest.raises(OSError, match="synthetic disk write failure"):
        run(archive_fixture)
    assert archive_fixture.output.is_dir()
    assert list((archive_fixture.output / "raw").glob("*.npy"))
    assert not (archive_fixture.output / "manifest.json").exists()
    assert snapshot(archive_fixture.archive) == archive_fixture.before
