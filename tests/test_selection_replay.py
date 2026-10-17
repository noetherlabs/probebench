"""Bounded offline replay tests against a small, independently pinned export."""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_selection_replay.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC = importlib.util.spec_from_file_location("selection_replay_generator", SCRIPT)
    replay = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(replay)
finally:
    sys.path.pop(0)


CASE_FIELDS = {"id", "candidate_id", "family", "photons_per_exposure", "exposure_budget",
               "base_wavefront_id", "member", "stratum_id", "replicate"}
TEMPLATE = '''<!doctype html><html><head><meta http-equiv="Content-Security-Policy"
content="default-src 'none'; script-src 'sha256-%%SCRIPT_HASH%%'; img-src 'self' data:;"></head>
<body><script id="experiment" type="application/json">%%DATA%%</script>
<script id="replay-script">const experiment = JSON.parse(document.getElementById("experiment").textContent);</script>
</body></html>'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_json(path, value):
    path.write_bytes(encoded(value))


def write_lines(path, values):
    path.write_bytes(b"".join(encoded(value) for value in values))


def snapshot(root):
    return {path.relative_to(root).as_posix(): digest(path.read_bytes())
            for path in sorted(root.rglob("*")) if path.is_file()}


def repin(item):
    item.manifest["files"] = {name: sha for name, sha in snapshot(item.export).items()
                              if name != "manifest.json"}
    write_json(item.export / "manifest.json", item.manifest)
    item.pin = digest((item.export / "manifest.json").read_bytes())


def write_episodes(item):
    for episode in item.episodes:
        record = episode["record"]
        record["record_sha256"] = replay.release.canonical_hash(
            {key: value for key, value in record.items() if key != "record_sha256"})
    write_lines(item.export / "episodes.jsonl", item.episodes)


@pytest.fixture
def exported(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    source, output = root / "imagefolder", root / "replay"
    source.mkdir()
    (source / "images").mkdir()
    (source / "raw").mkdir()
    template = root / "trusted-template.html"
    template.write_text(TEMPLATE)
    monkeypatch.setattr(replay, "TEMPLATE", template)
    monkeypatch.setattr(replay.imagefolder.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=100 * 1024**3))
    designs = [
        ("random_probe_v2", "random-small", "base-a", "stratum-a", 0, 0, 100, 2),
        ("random_probe_v2", "random-large", "base-a", "stratum-a", 0, 0, 100, 2),
        ("random_probe_v2", "random-small", "base-a", "stratum-a", 1, 1, 200, 1),
        ("coordinate_search_v2", "coordinate", "base-b", "stratum-b", 0, 0, 200, 2),
    ]
    fields = ("family", "candidate_id", "base_wavefront_id", "stratum_id", "member",
              "replicate", "photons_per_exposure", "exposure_budget")
    candidates = [{"id": "random-small", "family": "random_probe_v2", "parameters": {}},
                  {"id": "random-large", "family": "random_probe_v2", "parameters": {}},
                  {"id": "coordinate", "family": "coordinate_search_v2", "parameters": {}}]
    configuration = {"cohort": {"specification": {"optics": {"pupil_pixels": 2, "oversampling": 2}}},
        "candidates": candidates, "photons_per_exposure": [100, 200], "exposure_budgets": [1, 2],
        "replicates": 2, "common_policy": {}}
    settings = {"schema": 1, "purpose": "classical_parameter_selection_v1", "families": {
        "random_probe_v2": {"candidate_id": "random-small", "policy": {}},
        "coordinate_search_v2": {"candidate_id": "coordinate", "policy": {}}}}
    episodes, metadata = [], []
    for number, values in enumerate(designs):
        case = {"id": "case_" + f"{number + 1:064x}", **dict(zip(fields, values)),
            "truth_opd_nm": [31, 0, 0, 0, 0, 0], "detector_seed": 123 + number,
            "policy_seed": 321 + number, "private_receipt": "not-for-browser"}
        case["member"] = "primary" if case["member"] == 0 else "conjugate"
        trajectory, frame_hashes = [], {}
        for index in range(case["exposure_budget"]):
            measured = {"index": index, "applied_opd_nm": [float(index), 0, 0, 0, 0, 0],
                "expected_photons": case["photons_per_exposure"],
                "cumulative_expected_photons": (index + 1) * case["photons_per_exposure"],
                "realized_photoelectrons": 95 + index,
                "residual_wfe_nm": 30.5 - index, "strehl_peak_ratio": 0.7 + index * 0.01}
            trajectory.append(measured)
            basename = f"{case['id']}_frame_{index:04d}"
            raw_name, image_name = f"raw/{basename}.npy", basename + ".png"
            array = np.arange(16, dtype=np.float64).reshape(4, 4) - 5.5 - number - index
            np.save(source / raw_name, array, allow_pickle=False)
            display = np.log1p(np.maximum(array, 0)) / np.log1p(case["photons_per_exposure"])
            rgba = replay.imagefolder.matplotlib.colormaps["magma"](display, bytes=True)[::-1]
            Image.fromarray(rgba).save(source / "images" / image_name)
            frame_hash = digest(f"native retained frame {number} {index}".encode())
            frame_hashes[f"frame_{index:04d}.npz"] = frame_hash
            metadata.append({"episode_id": case["id"], **{key: case[key] for key in fields},
                "split": "public_tuning", "exposure_index": index, "file_name": image_name,
                "raw_array": raw_name, "raw_array_sha256": digest((source / raw_name).read_bytes()),
                "source_frame_sha256": frame_hash,
                "selected_reference": settings["families"][case["family"]]["candidate_id"] == case["candidate_id"],
                **{key: measured[key] for key in ("applied_opd_nm", "expected_photons", "cumulative_expected_photons")},
                "evaluator_wfe_nm": measured["residual_wfe_nm"],
                "evaluator_strehl_peak_ratio": measured["strehl_peak_ratio"]})
        measured_indices = [] if number == 0 else [0] if number == 1 else [case["exposure_budget"] - 1]
        final_command = [19, 0, 0, 0, 0, 0] if not measured_indices else trajectory[measured_indices[0]]["applied_opd_nm"]
        final_metrics = ({"residual_wfe_nm": 20.0, "strehl_peak_ratio": 0.85} if not measured_indices
                         else {key: trajectory[measured_indices[0]][key]
                               for key in ("residual_wfe_nm", "strehl_peak_ratio")})
        result = {"trajectory": trajectory, "exposures": case["exposure_budget"],
            "expected_photons_used": case["exposure_budget"] * case["photons_per_exposure"],
            "initial": {key: trajectory[0][key] for key in ("residual_wfe_nm", "strehl_peak_ratio")},
            "final": final_metrics,
            "final_applied_opd_nm": final_command, "hidden_opd_nm": case["truth_opd_nm"],
            "final_command_observation_indices": measured_indices,
            "final_command_has_camera_observation": bool(measured_indices),
            "final_command_is_last_observation": case["exposure_budget"] - 1 in measured_indices,
            "fit_records": [{"fit_status": "finite", "optimizer_success": True, "private_fit": "secret-a"},
                            {"fit_status": "finite", "optimizer_success": False, "private_fit": "secret-b"},
                            {"fit_status": "numerical_failure", "optimizer_success": False, "private_fit": "secret-c"}]}
        record = {"case": case, "result": result, "frame_sha256": frame_hashes,
                  "study_binding": "b" * 64}
        record["record_sha256"] = replay.release.canonical_hash(record)
        episodes.append({"episode_id": case["id"], "record": record})
    write_lines(source / "images/metadata.jsonl", metadata)
    write_lines(source / "episodes.jsonl", episodes)
    write_json(source / "configuration.json", configuration)
    write_json(source / "selected-settings.json", settings)
    provenance = {"schema": 1, "purpose": replay.imagefolder.PURPOSE, "split": "public_tuning",
        "verification": {"schema": 1, "packet_verified": True, "selection_reduction_verified": True,
            "full_camera_replay_verified": True, "archive_manifest_sha256": replay.release.NATIVE_MANIFEST_SHA256,
            "source_inventory_sha256": replay.release.SOURCE_INVENTORY_SHA256,
            "packet_manifest_sha256": "a" * 64, "episodes": len(episodes), "exposures": len(metadata),
            "candidates": 3, "selected_families": 2}}
    write_json(source / "provenance.json", provenance)
    for name in ("LICENSE", "NOTICE"):
        (source / name).write_bytes((ROOT / name).read_bytes())
    (source / "README.md").write_text("Public parameter-selection measurements.\n")
    manifest = {"schema": 1, "purpose": replay.imagefolder.PURPOSE, "split": "public_tuning",
        "input_manifest_sha256": replay.release.NATIVE_MANIFEST_SHA256,
        "packet_manifest_sha256": "a" * 64, "episode_count": len(episodes),
        "exposure_count": len(metadata), "candidate_count": 3, "independent_base_count": 2}
    item = SimpleNamespace(export=source, output=output, pin=None, manifest=manifest,
        episodes=episodes, metadata=metadata, configuration=configuration, settings=settings,
        provenance=provenance, template=template)
    repin(item)
    return item


def generate(item):
    return replay.generate(item.export, item.pin, item.output)


def browser_data(output):
    html = (output / "index.html").read_text()
    match = re.search(r'<script\b[^>]*\bid="experiment"[^>]*>(.*?)</script>', html, re.S)
    assert match
    return html, json.loads(match.group(1))


def test_portable_replay_retains_all_identities_measurements_and_final_states(exported):
    item = exported
    before = snapshot(item.export)
    manifest = generate(item)
    assert snapshot(item.export) == before
    assert manifest == json.loads((item.output / "manifest.json").read_text())
    files = snapshot(item.output)
    del files["manifest.json"]
    assert manifest["files"] == files
    assert set(files) == {"index.html", "provenance.json", "LICENSE", "NOTICE",
                          *("images/" + row["file_name"] for row in item.metadata)}
    html, data = browser_data(item.output)
    assert set(data) == {"schema", "split", "episodes", "manifest_sha256", "native_manifest_sha256",
                         "exposure_count", "independent_base_count", "candidate_count"}
    assert data["schema"] == 1 and data["split"] == "public_tuning"
    assert data["manifest_sha256"] == item.pin
    assert data["native_manifest_sha256"] == replay.release.NATIVE_MANIFEST_SHA256
    assert data["exposure_count"] == len(item.metadata)
    assert data["candidate_count"] == 3 and data["independent_base_count"] == 2
    assert len(data["episodes"]) == len(item.episodes)
    for shown, original in zip(data["episodes"], item.episodes, strict=True):
        record = original["record"]
        case, result = record["case"], record["result"]
        assert set(shown) == {"case", "selected_reference", "frames", "final", "final_command",
                              "measured_indices", "initial", "fit_count", "failed_fits", "unconverged_fits"}
        assert shown["case"] == {key: case[key] for key in CASE_FIELDS}
        assert shown["selected_reference"] is (case["candidate_id"] == item.settings["families"][case["family"]]["candidate_id"])
        assert len(shown["frames"]) == case["exposure_budget"]
        for index, frame in enumerate(shown["frames"]):
            image = f"images/{case['id']}_frame_{index:04d}.png"
            assert frame == {"image": image, "measurement": result["trajectory"][index]}
            assert (item.output / image).read_bytes() == (item.export / image).read_bytes()
        for target, source in (("final", "final"), ("initial", "initial"),
                               ("final_command", "final_applied_opd_nm"),
                               ("measured_indices", "final_command_observation_indices")):
            assert shown[target] == result[source]
        assert (shown["fit_count"], shown["failed_fits"], shown["unconverged_fits"]) == (3, 1, 1)
    assert [episode["measured_indices"] for episode in data["episodes"]] == [[], [0], [0], [1]]
    assert sum(len(episode["frames"]) for episode in data["episodes"]) == len(item.metadata)
    assert not any(marker in html for marker in ("truth_opd_nm", "detector_seed", "policy_seed",
                                                "hidden_opd_nm", "private_fit", "not-for-browser"))
    for name in ("LICENSE", "NOTICE"):
        assert (item.output / name).read_bytes() == (item.export / name).read_bytes()
    script = re.search(r'<script id="replay-script">(.*?)</script>', html, re.S).group(1)
    expected_csp = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert "'sha256-" + expected_csp + "'" in html
    assert "%%DATA%%" not in html and "%%SCRIPT_HASH%%" not in html


def test_markup_in_labels_is_escaped_without_changing_browser_data(exported):
    item = exported
    label = '</script><script id="injected">window.injected=true</script> %%SCRIPT_HASH%%'
    for episode in item.episodes:
        record = episode["record"]
        if record["case"]["base_wavefront_id"] == "base-a":
            record["case"]["stratum_id"] = label
            record.pop("record_sha256")
            record["record_sha256"] = replay.release.canonical_hash(record)
    for row in item.metadata:
        if row["base_wavefront_id"] == "base-a":
            row["stratum_id"] = label
    write_episodes(item)
    write_lines(item.export / "images/metadata.jsonl", item.metadata)
    repin(item)
    generate(item)
    html, data = browser_data(item.output)
    assert label not in html
    assert '<script id="injected">' not in html
    assert "\\u003c/script>" in html
    assert data["episodes"][0]["case"]["stratum_id"] == label
    assert len(re.findall(r"<script\b", html)) == 2


@pytest.mark.parametrize("pin", ["0" * 64, "not-a-sha256", "", "../manifest.json"])
def test_untrusted_manifest_pin_refuses_before_output(exported, pin):
    exported.pin = pin
    with pytest.raises((ValueError, OSError)):
        generate(exported)
    assert not exported.output.exists()


@pytest.mark.parametrize("field,value", [("schema", True), ("purpose", "other_export"),
    ("split", "held_out"), ("input_manifest_sha256", "0" * 64),
    ("episode_count", 0), ("exposure_count", 999999), ("candidate_count", 2)])
def test_manifest_semantics_are_checked_even_with_trusted_hash(exported, field, value):
    exported.manifest[field] = value
    repin(exported)
    with pytest.raises(ValueError):
        generate(exported)
    assert not exported.output.exists()


@pytest.mark.parametrize("field,value", [("full_camera_replay_verified", False),
    ("archive_manifest_sha256", "0" * 64), ("source_inventory_sha256", "0" * 64),
    ("packet_manifest_sha256", "0" * 64), ("exposures", 3)])
def test_provenance_cannot_claim_a_different_or_incomplete_native_replay(exported, field, value):
    exported.provenance["verification"][field] = value
    write_json(exported.export / "provenance.json", exported.provenance)
    repin(exported)
    with pytest.raises(ValueError):
        generate(exported)
    assert not exported.output.exists()


@pytest.mark.parametrize("mutation", ["missing", "unlisted", "changed", "unsafe_name", "self_entry"])
def test_exact_inventory_and_hashes_are_required(exported, mutation):
    item = exported
    image = item.export / "images" / item.metadata[0]["file_name"]
    if mutation == "missing":
        image.unlink()
    elif mutation == "unlisted":
        (item.export / "unexpected.txt").write_text("extra")
    elif mutation == "changed":
        image.write_bytes(image.read_bytes() + b"changed")
    else:
        item.manifest["files"]["../outside" if mutation == "unsafe_name" else "manifest.json"] = "0" * 64
        write_json(item.export / "manifest.json", item.manifest)
        item.pin = digest((item.export / "manifest.json").read_bytes())
    with pytest.raises((ValueError, OSError)):
        generate(item)
    assert not item.output.exists()


@pytest.mark.parametrize("entry", ["image_link", "directory_link", "manifest_link", "root_link",
                                   "fifo", "manifest_fifo"])
def test_symlinks_and_special_inputs_are_rejected(exported, entry):
    item = exported
    outside = item.export.parent / "outside"
    if entry == "root_link":
        outside.symlink_to(item.export, target_is_directory=True)
        item.export = outside
    elif entry in {"fifo", "manifest_fifo"}:
        path = item.export / ("manifest.json" if entry == "manifest_fifo" else "unexpected.fifo")
        if path.exists():
            path.unlink()
        os.mkfifo(path)
    else:
        path = item.export / ({"image_link": "images/" + item.metadata[0]["file_name"],
                               "directory_link": "images", "manifest_link": "manifest.json"}[entry])
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=entry == "directory_link")
    with pytest.raises((ValueError, OSError)):
        generate(item)
    assert not item.output.exists()


@pytest.mark.parametrize("destination", ["existing", "inside", "parent_alias", "case_alias", "linked_parent", "missing_parent"])
def test_output_must_be_new_and_disjoint_from_every_input_alias(exported, destination):
    item = exported
    source = item.export
    before = snapshot(source)
    if destination == "existing":
        item.output.mkdir()
    elif destination == "inside":
        item.output = source / "new-replay"
    elif destination == "parent_alias":
        item.export = source / "images" / ".."
        item.output = source / "new-replay"
    elif destination == "case_alias":
        alias = source.with_name(source.name.upper())
        if not alias.exists() or not alias.samefile(source):
            pytest.skip("Filesystem has no case-insensitive directory alias")
        item.output = alias / "new-replay"
    elif destination == "linked_parent":
        link = source.parent / "linked"
        link.symlink_to(source.parent, target_is_directory=True)
        item.output = link / "new-replay"
    else:
        item.output = item.output / "missing" / "new-replay"
    with pytest.raises((ValueError, OSError)):
        generate(item)
    assert snapshot(source) == before
    if destination != "existing":
        assert not item.output.exists()


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "candidate", "photons", "label",
    "command", "selected", "raw_hash", "source_hash", "raw_path", "image_path"])
def test_metadata_must_match_complete_case_and_frame_evidence(exported, mutation):
    item = exported
    rows = item.metadata
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "missing":
        rows.pop()
    else:
        field, value = {
            "candidate": ("candidate_id", "random-large"), "photons": ("photons_per_exposure", 200),
            "label": ("evaluator_wfe_nm", 999.0), "command": ("applied_opd_nm", [9, 0, 0, 0, 0, 0]),
            "selected": ("selected_reference", False), "raw_hash": ("raw_array_sha256", "0" * 64),
            "source_hash": ("source_frame_sha256", "0" * 64), "raw_path": ("raw_array", "../outside.npy"),
            "image_path": ("file_name", "../outside.png")}[mutation]
        rows[0][field] = value
    write_lines(item.export / "images/metadata.jsonl", rows)
    repin(item)
    with pytest.raises((ValueError, OSError)):
        generate(item)
    assert not item.output.exists()


@pytest.mark.parametrize("mutation", ["duplicate_episode", "missing_episode", "final_indices", "final_has",
                                      "final_last", "final_command", "trajectory_index"])
def test_episode_inventory_and_final_observation_states_are_consistent(exported, mutation):
    item = exported
    if mutation == "duplicate_episode":
        item.episodes.append(item.episodes[0])
    elif mutation == "missing_episode":
        item.episodes.pop()
    else:
        result = item.episodes[0]["record"]["result"]
        if mutation == "final_indices":
            result["final_command_observation_indices"] = [99]
        elif mutation == "final_has":
            result["final_command_has_camera_observation"] = True
        elif mutation == "final_last":
            item.episodes[1]["record"]["result"]["final_command_is_last_observation"] = True
        elif mutation == "final_command":
            result["final_applied_opd_nm"] = [0, 0, 0, 0, 0, 0]
        else:
            result["trajectory"][0]["index"] = 8
    write_episodes(item)
    repin(item)
    with pytest.raises(ValueError):
        generate(item)
    assert not item.output.exists()


def test_episode_record_fingerprint_cannot_be_replaced_by_export_manifest(exported):
    exported.episodes[0]["record"]["result"]["final"]["residual_wfe_nm"] = 987.0
    write_lines(exported.export / "episodes.jsonl", exported.episodes)
    repin(exported)
    with pytest.raises(ValueError, match="Episode record fingerprint"):
        generate(exported)
    assert not exported.output.exists()


@pytest.mark.parametrize("filename,content", [
    ("images/metadata.jsonl", b'{"episode_id":"one","episode_id":"two"}\n'),
    ("episodes.jsonl", b'{"value":NaN}\n'), ("configuration.json", b"["),
])
def test_malformed_or_ambiguous_json_is_rejected(exported, filename, content):
    (exported.export / filename).write_bytes(content)
    repin(exported)
    with pytest.raises((ValueError, KeyError, TypeError)):
        generate(exported)
    assert not exported.output.exists()


@pytest.mark.parametrize("target", ["initial", "final", "trajectory"])
def test_extra_latent_fields_cannot_enter_public_measurement_objects(exported, target):
    result = exported.episodes[0]["record"]["result"]
    entry = result["trajectory"][0] if target == "trajectory" else result[target]
    entry["hidden_coefficients"] = [31, 0, 0, 0, 0, 0]
    write_episodes(exported)
    repin(exported)
    with pytest.raises(ValueError):
        generate(exported)
    assert not exported.output.exists()


@pytest.mark.parametrize("mutation", ["dtype", "shape", "nonfinite", "png_dimensions", "png_pixels"])
def test_display_and_signed_array_shapes_are_validated(exported, mutation):
    item = exported
    first = item.metadata[0]
    if mutation.startswith("png_"):
        Image.new("RGBA", (3, 4) if mutation == "png_dimensions" else (4, 4)).save(
            item.export / "images" / first["file_name"])
    else:
        raw = item.export / first["raw_array"]
        array = np.load(raw, allow_pickle=False)
        if mutation == "dtype":
            array = array.astype(np.float32)
        elif mutation == "shape":
            array = array.reshape(2, 8)
        else:
            array[0, 0] = np.nan
        np.save(raw, array, allow_pickle=False)
        first["raw_array_sha256"] = digest(raw.read_bytes())
        write_lines(item.export / "images/metadata.jsonl", item.metadata)
    repin(item)
    with pytest.raises(ValueError):
        generate(item)
    assert not item.output.exists()


def test_unbounded_raw_array_header_is_rejected_before_array_interpretation(exported, monkeypatch):
    item = exported
    first = item.metadata[0]
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": "<f8", "fortran_order": False, "shape": (1_000_000, 1_000_000)})
    raw = item.export / first["raw_array"]
    raw.write_bytes(header.getvalue())
    first["raw_array_sha256"] = digest(raw.read_bytes())
    write_lines(item.export / "images/metadata.jsonl", item.metadata)
    repin(item)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("An unbounded raw header reached array interpretation")

    monkeypatch.setattr(np, "frombuffer", forbidden_load)
    with pytest.raises(ValueError):
        generate(item)
    assert not item.output.exists()


def test_insufficient_storage_preserves_input_without_output(exported, monkeypatch):
    before = snapshot(exported.export)
    monkeypatch.setattr(replay.imagefolder.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=replay.imagefolder.RESERVE_BYTES - 1))
    with pytest.raises((ValueError, OSError)):
        generate(exported)
    assert not exported.output.exists()
    assert snapshot(exported.export) == before


def test_storage_reserve_is_checked_during_writing(exported, monkeypatch):
    def available(_):
        return SimpleNamespace(free=replay.imagefolder.RESERVE_BYTES - 1 if exported.output.exists()
                               else 100 * 1024**3)

    monkeypatch.setattr(replay.imagefolder.shutil, "disk_usage", available)
    with pytest.raises((ValueError, OSError)):
        generate(exported)
    assert exported.output.is_dir()
    assert not (exported.output / "manifest.json").exists()


def test_png_bytes_are_rehashed_when_consumed_after_admission(exported, monkeypatch):
    original = replay.imagefolder.check_space
    mutations = []
    image = exported.export / "images" / exported.metadata[-1]["file_name"]

    def mutate_after_admission(*args, **kwargs):
        original(*args, **kwargs)
        if not mutations:
            image.write_bytes(image.read_bytes() + b"changed after validation")
            mutations.append(True)

    monkeypatch.setattr(replay.imagefolder, "check_space", mutate_after_admission)
    with pytest.raises(ValueError):
        generate(exported)
    assert mutations == [True]
    assert not (exported.output / "manifest.json").exists()


def test_interrupted_image_copy_has_no_completion_manifest(exported, monkeypatch):
    original = replay.imagefolder.write_new
    copied = []

    def fail_second_image(path, *args, **kwargs):
        if Path(path).suffix == ".png":
            copied.append(path)
            if len(copied) == 2:
                raise OSError("synthetic image write failure")
        return original(path, *args, **kwargs)

    before = snapshot(exported.export)
    monkeypatch.setattr(replay.imagefolder, "write_new", fail_second_image)
    with pytest.raises(OSError, match="synthetic image write failure"):
        generate(exported)
    assert exported.output.is_dir()
    assert list((exported.output / "images").glob("*.png"))
    assert not (exported.output / "manifest.json").exists()
    assert snapshot(exported.export) == before
