"""Small immutable archives exercise scalar export semantics and failure boundaries."""
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_scalar_gain_imagefolder.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC = importlib.util.spec_from_file_location("scalar_imagefolder_export", SCRIPT)
    exporter = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(exporter)
finally:
    sys.path.pop(0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): digest(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def case_bytes(record, images, *, compressed=False):
    stream = io.BytesIO()
    save = np.savez_compressed if compressed else np.savez
    save(stream, images=images, metadata=np.frombuffer(exporter.encoded(record), dtype=np.uint8))
    return stream.getvalue()


@pytest.fixture
def small_archive(tmp_path, monkeypatch):
    archive, output = tmp_path / "archive", tmp_path / "export"
    archive.mkdir(); (archive / "cases").mkdir()
    (archive / "source").mkdir(); (archive / "reviewed-sources").mkdir()
    (archive / ".lock").touch()
    source = b'raise RuntimeError("Archived source must never execute")\n'
    (archive / "source/never_execute.py").write_bytes(source)
    reviewed = exporter.encoded({"fixture": "retained evidence"})
    (archive / "reviewed-sources/fixture.json").write_bytes(reviewed)
    sources = {"never_execute.py": digest(source)}
    protocol = {"reviewed_fixture_sources": {"fixture.json": digest(reviewed)}, "fixture": True}
    runtime = {"python": "fixture-runtime"}
    config_hash = exporter.release.canonical_hash(protocol)
    study = {"configuration": {"protocol": protocol, "admission": {"status": "FROZEN",
        "source_revision": exporter.SOURCE_REVISION, "configuration_sha256": config_hash,
        "source_sha256": sources, "runtime": runtime}}, "source_sha256": sources, "runtime": runtime}
    cases, records, arrays = [], {}, {}
    for member in exporter.MEMBERS:
        for gain in exporter.GAINS:
            for candidate, family in exporter.CANDIDATES.items():
                identity = {"cohort_id": exporter.COHORT, "base_index": 0, "stratum_id": "low", "member": member,
                    "replicate": 0, "base_wavefront_id": f"{exporter.COHORT}:low:000000",
                    "gain": gain, "candidate_id": candidate, "family": family}
                case = {**identity, "id": "case_" + exporter.release.canonical_hash(identity),
                        "truth_opd_nm": [13., 0., 0., 0., 0., 0.], "detector_seed": 19, "policy_seed": 23}
                cases.append(case)
                observations, trajectory, images = [], [], []
                for index in range(7):
                    command = [float((0, 80, -80)[index % 3]), 0., 0., 0., 0., 0.]
                    observations.append({"schema": exporter.OBSERVATION_SCHEMA, "index": index,
                        "commanded_opd_nm": command, "nominal_applied_opd_nm": command,
                        "nominal_calibration": dict(exporter.NOMINAL_CALIBRATION), "expected_photons": 2000,
                        "cumulative_expected_photons": 2000*(index+1)})
                    trajectory.append({"index": index, "applied_opd_nm": (gain*np.array(command)).tolist(),
                        "expected_photons": 2000, "cumulative_expected_photons": 2000*(index+1),
                        "residual_wfe_nm": 12.5-index, "strehl_peak_ratio": .7+.01*index})
                    image = (np.arange(16, dtype="<f8").reshape(4, 4)-5.5-index)
                    image[0, 0] = -0.0
                    images.append(image)
                images = np.stack(images)
                record = {"schema": 1, "purpose": exporter.NATIVE_PURPOSE, "case": case,
                    "case_binding": exporter.release.canonical_hash({"study_sha256": exporter.release.canonical_hash(study), "case": case}),
                    "observations": observations, "failed_observation": None, "images_sha256": digest(images.tobytes()),
                    "evaluation": {"schema": "probebench.scalar-gain-evaluation/1", "status": "submitted",
                        "command_gain": gain, "nominal_calibration": dict(exporter.NOMINAL_CALIBRATION),
                        "nominal_observations": observations, "final_commanded_opd_nm": [19., 0., 0., 0., 0., 0.],
                        "physical": {"trajectory": trajectory, "exposures": 7, "expected_photons_used": 14000,
                            "hidden_opd_nm": case["truth_opd_nm"], "final_applied_opd_nm": [19.*gain, 0., 0., 0., 0., 0.],
                            "final_command_has_camera_observation": False, "final_command_observation_indices": [],
                            "final_command_is_last_observation": False}},
                    "fit_records": [{"optimizer_success": False, "fit_status": "finite", "reason": "retained fixture status"}],
                    "elapsed_seconds": .1}
                records[case["id"]], arrays[case["id"]] = record, images
                (archive / "cases" / (case["id"] + ".npz")).write_bytes(case_bytes(record, images))
    for name, value in {"study.json": study, "cases.json": cases, "preparation.json": {"cohort": "fixture"},
                        "summary.json": [{"case_id": case["id"]} for case in cases], "analysis.json": {"status": "fixture"}}.items():
        (archive / name).write_bytes(exporter.encoded(value))
    manifest = {"schema": 1, "purpose": exporter.NATIVE_PURPOSE, "episode_count": len(cases),
        "exposure_count": 7*len(cases), "source_sha256": sources, "files": snapshot(archive)}
    raw_manifest = exporter.encoded(manifest)
    (archive / "manifest.json").write_bytes(raw_manifest)
    for name, value in {"MANIFEST_SHA256": digest(raw_manifest), "CONFIG_SHA256": config_hash,
                        "STRATA": ("low",), "BASES_PER_STRATUM": 1, "EPISODES": len(cases),
                        "EXPOSURES": 7*len(cases), "SHAPE": (4, 4)}.items():
        monkeypatch.setattr(exporter, name, value)
    monkeypatch.setattr(exporter.shared.shutil, "disk_usage", lambda _: SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(exporter.shared.release, "verify", lambda *a, **k: pytest.fail("Archived source verifier must not run"))
    return SimpleNamespace(archive=archive, output=output, study=study, cases=cases, records=records,
                           arrays=arrays, manifest=manifest, before=snapshot(archive))


def repin(item, monkeypatch):
    item.manifest["files"] = {name: value for name, value in snapshot(item.archive).items() if name != "manifest.json"}
    raw = exporter.encoded(item.manifest)
    (item.archive / "manifest.json").write_bytes(raw)
    monkeypatch.setattr(exporter, "MANIFEST_SHA256", digest(raw))


def test_release_fingerprint_and_declared_complete_grid():
    assert exporter.MANIFEST_SHA256 == "ee56441561fa50aeb80f1e33e516cf9151248445f75d997e0d46e9b5002c3322"
    assert exporter.EPISODES == 288 and exporter.EXPOSURES == 2016
    assert exporter.CONFIG_NAME == "scalar_command_response_v1" and exporter.SPLIT == "development"


def test_round_trip_retains_every_signed_sample_and_native_record(small_archive):
    item = small_archive
    result = exporter.export(item.archive, item.output)
    assert snapshot(item.archive) == item.before
    assert result == json.loads((item.output / "manifest.json").read_text())
    files = snapshot(item.output); files.pop("manifest.json")
    assert files == result["files"]
    assert result["exposure_count"] == 84 and result["expected_photons_consumed"] == 168000
    assert result["episode_count"] == 12
    rows = json_lines(item.output / "images/metadata.jsonl")
    assert len(rows) == 84
    assert {(row["episode_id"], row["exposure_index"]) for row in rows} == {
        (case["id"], index) for case in item.cases for index in range(7)}
    for row in rows:
        payload = (item.output / row["raw_array"]).read_bytes()
        image = np.load(io.BytesIO(payload), allow_pickle=False)
        original = item.arrays[row["episode_id"]][row["exposure_index"]]
        assert image.dtype == np.dtype("<f8") and image.tobytes() == original.tobytes()
        assert digest(payload) == row["raw_array_sha256"]
        assert digest(image.tobytes()) == row["image_electrons_sha256"]
        assert row["source_case_sha256"] == item.manifest["files"][f"cases/{row['episode_id']}.npz"]
        with Image.open(item.output / "images" / row["file_name"]) as preview:
            preview.load()
            assert preview.size == (4, 4)
    assert result["negative_detector_samples_preserved"] == sum(int(np.count_nonzero(a < 0)) for a in item.arrays.values())
    episodes = json_lines(item.output / "episodes.jsonl")
    assert {entry["episode_id"]: entry["record"] for entry in episodes} == item.records
    assert all(not entry["record"]["fit_records"][0]["optimizer_success"] for entry in episodes)
    assert all(not entry["record"]["evaluation"]["physical"]["final_command_has_camera_observation"] for entry in episodes)
    assert (item.output / "source-manifest.json").read_bytes() == (item.archive / "manifest.json").read_bytes()
    for name in ("LICENSE", "NOTICE"):
        assert (item.output / name).read_bytes() == (ROOT / name).read_bytes()


def test_gain_mismatch_remains_evaluator_only_and_nominal_schema_is_explicit(small_archive):
    item = small_archive
    exporter.export(item.archive, item.output)
    rows = json_lines(item.output / "images/metadata.jsonl")
    for row in rows:
        assert row["observation_schema"] == "probebench.nominal-observation/1"
        assert row["commanded_opd_nm"] == row["nominal_applied_opd_nm"]
        assert row["nominal_calibration"] == exporter.NOMINAL_CALIBRATION
        assert not {"gain", "command_gain", "applied_opd_nm", "truth_opd_nm", "detector_seed", "policy_seed", "selected_reference"} & row.keys()
        np.testing.assert_array_equal(row["evaluator_applied_opd_nm"], row["evaluator_command_gain"]*np.array(row["commanded_opd_nm"]))
        if row["exposure_index"] == 1 and row["evaluator_command_gain"] != 1:
            assert row["commanded_opd_nm"] != row["evaluator_applied_opd_nm"]
    card = (item.output / "README.md").read_text()
    assert "config_name: scalar_command_response_v1" in card
    assert "split: development" in card and "path: images/**" in card
    assert "split: public_tuning" not in card


@pytest.mark.parametrize("damage", ["extra", "missing", "corrupt", "manifest"])
def test_damaged_native_inventory_refuses_before_output(small_archive, damage):
    item = small_archive
    first = item.archive / "cases" / (item.cases[0]["id"] + ".npz")
    if damage == "extra":
        (item.archive / "case.pending").write_bytes(b"interrupted")
    elif damage == "missing":
        first.unlink()
    elif damage == "corrupt":
        first.write_bytes(b"corrupt")
    else:
        (item.archive / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        exporter.export(item.archive, item.output)
    assert not item.output.exists()


def test_repeated_case_grid_refuses_before_output(small_archive, monkeypatch):
    item = small_archive
    (item.archive / "cases.json").write_bytes(exporter.encoded([item.cases[0]]*len(item.cases)))
    repin(item, monkeypatch)
    with pytest.raises(ValueError, match="grid"):
        exporter.export(item.archive, item.output)
    assert not item.output.exists()


@pytest.mark.parametrize("damage", ["compressed", "object", "wrong_shape", "extra_member", "nonfinite"])
def test_bounded_npz_decoder_rejects_incompatible_arrays(small_archive, damage):
    item = small_archive
    record = item.records[item.cases[0]["id"]]
    images = item.arrays[item.cases[0]["id"]].copy()
    if damage == "object":
        images = images.astype(object)
    elif damage == "wrong_shape":
        images = images[:-1]
    elif damage == "nonfinite":
        images[0, 0, 0] = np.nan
    if damage == "extra_member":
        stream = io.BytesIO(); np.savez(stream, images=images, metadata=np.zeros(1, dtype=np.uint8), extra=np.array([0]))
        data = stream.getvalue()
    else:
        data = case_bytes(record, images, compressed=damage == "compressed")
    with pytest.raises(ValueError):
        exporter.decode_case(data)


@pytest.mark.parametrize("damage", ["legacy_alias", "nominal_alias", "physical_gain", "photon", "image_digest"])
def test_observation_semantics_reject_silent_legacy_projection(small_archive, damage):
    item = small_archive
    case = item.cases[0]
    record = json.loads(json.dumps(item.records[case["id"]]))
    if damage == "legacy_alias":
        record["observations"][0]["applied_opd_nm"] = [0.]*6
    elif damage == "nominal_alias":
        record["observations"][1]["nominal_applied_opd_nm"] = [72., 0., 0., 0., 0., 0.]
    elif damage == "physical_gain":
        record["evaluation"]["physical"]["trajectory"][1]["applied_opd_nm"] = [80., 0., 0., 0., 0., 0.]
    elif damage == "photon":
        record["observations"][0]["cumulative_expected_photons"] = 4000
    else:
        record["images_sha256"] = "0"*64
    record["evaluation"]["nominal_observations"] = record["observations"]
    with pytest.raises(ValueError):
        exporter.check_record(item.study, case, record, item.arrays[case["id"]])


def test_existing_or_overlapping_output_is_refused(small_archive):
    item = small_archive
    item.output.mkdir(); (item.output / "keep").write_bytes(b"untouched")
    with pytest.raises(ValueError, match="new directory"):
        exporter.export(item.archive, item.output)
    with pytest.raises(ValueError, match="disjoint"):
        exporter.export(item.archive, item.archive / "new-output")
    assert (item.output / "keep").read_bytes() == b"untouched"


def test_symlink_and_writer_lock_are_refused(small_archive):
    item = small_archive
    (item.archive / "linked").symlink_to(item.archive / "study.json")
    with pytest.raises(ValueError, match="regular"):
        exporter.export(item.archive, item.output)
    (item.archive / "linked").unlink()
    with (item.archive / ".lock").open("r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="busy"):
            exporter.export(item.archive, item.output)
    assert not item.output.exists()


def test_storage_refusal_precedes_output_and_input_remains_unchanged(small_archive, monkeypatch):
    item = small_archive
    monkeypatch.setattr(exporter.shared.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="storage"):
        exporter.export(item.archive, item.output)
    assert not item.output.exists() and snapshot(item.archive) == item.before


def test_interrupted_export_never_publishes_a_complete_manifest(small_archive, monkeypatch):
    item = small_archive
    original, calls = exporter.shared.check_space, []
    def fail_after_some_writes(*args):
        calls.append(args)
        if len(calls) > 6:
            raise ValueError("simulated disk reserve failure")
        return original(*args)
    monkeypatch.setattr(exporter.shared, "check_space", fail_after_some_writes)
    with pytest.raises(ValueError, match="simulated"):
        exporter.export(item.archive, item.output)
    assert item.output.exists() and list((item.output / "raw").iterdir())
    assert not (item.output / "manifest.json").exists()
    assert snapshot(item.archive) == item.before
    with pytest.raises(ValueError, match="new directory"):
        exporter.export(item.archive, item.output)


@pytest.mark.parametrize("target", ["input", "output"])
def test_late_input_or_output_change_prevents_completion(small_archive, monkeypatch, target):
    item = small_archive
    original, changed = exporter.shared.write_new, []
    def mutate_after_write(path, *args):
        result = original(path, *args)
        if not changed:
            changed.append(True)
            if target == "input":
                (item.archive / "source/never_execute.py").write_bytes(b"changed after initial inventory")
            else:
                path.write_bytes(b"changed output")
        return result
    monkeypatch.setattr(exporter.shared, "write_new", mutate_after_write)
    with pytest.raises(ValueError, match="changed|differ"):
        exporter.export(item.archive, item.output)
    assert not (item.output / "manifest.json").exists()
