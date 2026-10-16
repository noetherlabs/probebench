"""Publication boundaries must consume only the verified episode inventory."""
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import probebench.estimation as estimation
import probebench.sequential as sequential
from probebench.qualification import canonical_hash
from probebench.run import digest, verify_files, write_json


REPOSITORY = Path(__file__).parents[1]
RETAINED_RUN = REPOSITORY / "results/sequential-calibration-20261014"


@pytest.fixture(autouse=True)
def no_new_exposures_or_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Publication checks must not acquire or optimize")

    monkeypatch.setattr(sequential.Episode, "expose", forbidden)
    monkeypatch.setattr(estimation, "minimize", forbidden)


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY / "scripts"))
    modules = {}
    for name in ("export_imagefolder", "generate_sequential_replay"):
        spec = importlib.util.spec_from_file_location(
            f"export_security_{name}", REPOSITORY / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


@pytest.fixture
def retained_case(tmp_path):
    """Rebind one existing no-action exposure without generating camera data."""
    root = tmp_path.resolve() / "run"
    root.mkdir()
    shutil.copytree(RETAINED_RUN / "source", root / "source")
    study = json.loads((RETAINED_RUN / "study.json").read_text())
    study["configuration"].update(
        methods=["no_action_v2"], exposure_budgets=[1], replicates=1,
        truths_opd_nm=study["configuration"]["truths_opd_nm"][:1])
    write_json(root / "study.json", study)
    case = sequential.cases_for(study["configuration"])[0]
    record = json.loads((RETAINED_RUN / "cases/episode_0000.json").read_text())
    assert record["case"] == case

    frame_dir = root / "frames" / case["id"]
    frame_dir.mkdir(parents=True)
    with np.load(RETAINED_RUN / "frames/episode_0000/frame_0000.npz",
                 allow_pickle=False) as saved:
        pixels = saved["image_electrons"].copy()
        frame_record = json.loads(saved["record_json"].item())
    frame_record["case_binding"] = canonical_hash(
        {"study": canonical_hash(study), "case": case})
    np.savez_compressed(
        frame_dir / "frame_0000.npz", image_electrons=pixels,
        record_json=np.array(json.dumps(frame_record, sort_keys=True, allow_nan=False)),
        record_sha256=np.array(canonical_hash(frame_record)))
    record["study_binding"] = canonical_hash(study)
    record["frame_sha256"] = {"frame_0000.npz": digest(frame_dir / "frame_0000.npz")}
    record.pop("record_sha256")
    record["record_sha256"] = canonical_hash(record)
    (root / "cases").mkdir()
    write_json(root / "cases/episode_0000.json", record)
    rows = [sequential.summary_row(record)]
    write_json(root / "summary.json", rows)
    (root / "report.md").write_text(sequential.report_text(rows, study["configuration"]))
    manifest = {
        "schema": 1, "purpose": sequential.PURPOSE,
        "source_sha256": study["source_sha256"],
        "episode_count": 1, "exposure_count": 1,
        "files": {str(path.relative_to(root)): digest(path)
                  for path in sorted(root.rglob("*")) if path.is_file()},
    }
    write_json(root / "manifest.json", manifest)
    assert sequential.verify_run(root) == manifest
    return SimpleNamespace(root=root, record=record, pixels=pixels, manifest=manifest)


def invoke(entry, scripts, root, output, monkeypatch):
    if entry == "imagefolder":
        scripts["export_imagefolder"].export(root, output)
    else:
        monkeypatch.setattr(sys, "argv", [
            "generate_sequential_replay.py", str(root), "--output", str(output)])
        scripts["generate_sequential_replay"].main()


@pytest.mark.parametrize("entry", ["imagefolder", "replay"])
@pytest.mark.parametrize("identity", ["relative", "absolute"])
def test_extra_hashed_episode_cannot_escape_publication(
        retained_case, scripts, monkeypatch, entry, identity):
    root = retained_case.root
    external = root.parent / "external"
    external.mkdir()
    np.savez_compressed(external / "frame_0000.npz", image_electrons=retained_case.pixels)
    extra = copy.deepcopy(retained_case.record)
    extra["case"]["id"] = "../../external" if identity == "relative" else str(external)
    extra["record_sha256"] = "deliberately invalid inner record hash"
    extra_path = root / "cases/episode_extra.json"
    write_json(extra_path, extra)
    manifest = copy.deepcopy(retained_case.manifest)
    manifest["files"]["cases/episode_extra.json"] = digest(extra_path)
    (root / "manifest.json").write_text(json.dumps(manifest))

    # Reject extra records even when the outer manifest includes their hashes.
    with pytest.raises(ValueError, match="inventory"):
        sequential.verify_run(root)
    victims = [root.parent / f"external_frame_0000.{suffix}" for suffix in ("png", "npy")]
    for victim in victims:
        victim.write_bytes(b"preserve unrelated file")
    output = root.parent / "output"
    with pytest.raises(ValueError):
        invoke(entry, scripts, root, output, monkeypatch)
    assert not output.exists()
    assert all(victim.read_bytes() == b"preserve unrelated file" for victim in victims)


@pytest.mark.parametrize("entry", ["imagefolder", "replay"])
def test_symlinked_source_frames_are_rejected_before_output(
        retained_case, scripts, monkeypatch, entry):
    root = retained_case.root
    frames = root / "frames/episode_0000"
    external = root.parent / "external"
    frames.rename(external)
    frames.symlink_to(external, target_is_directory=True)
    preserved = (external / "frame_0000.npz").read_bytes()
    output = root.parent / "output"
    with pytest.raises(ValueError):
        invoke(entry, scripts, root, output, monkeypatch)
    assert not output.exists()
    assert (external / "frame_0000.npz").read_bytes() == preserved


@pytest.mark.parametrize("entry", ["imagefolder", "replay"])
@pytest.mark.parametrize("relative", [
    "cases/unexpected.json", "frames/episode_0000/frame_0001.npz",
    "frames/episode_extra/frame_0000.npz",
])
def test_extra_hashed_artifacts_are_rejected_before_output(
        retained_case, scripts, monkeypatch, entry, relative):
    root = retained_case.root
    extra = root / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"not part of the declared experiment")
    manifest = copy.deepcopy(retained_case.manifest)
    manifest["files"][relative] = digest(extra)
    (root / "manifest.json").write_text(json.dumps(manifest))
    output = root.parent / "output"
    with pytest.raises(ValueError, match="inventory"):
        invoke(entry, scripts, root, output, monkeypatch)
    assert not output.exists()


def test_valid_single_exposure_exports_and_replays_exact_saved_pixels(
        retained_case, scripts, monkeypatch):
    root = retained_case.root
    imagefolder, replay = root.parent / "imagefolder", root.parent / "replay"
    invoke("imagefolder", scripts, root, imagefolder, monkeypatch)
    invoke("replay", scripts, root, replay, monkeypatch)

    manifest = verify_files(imagefolder)
    assert manifest["episode_count"] == manifest["exposure_count"] == 1
    rows = [json.loads(line) for line in (imagefolder / "images/metadata.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (imagefolder / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == len(episodes) == 1
    assert rows[0]["episode_id"] == episodes[0]["episode_id"] == "episode_0000"
    raw = imagefolder / rows[0]["raw_array"]
    assert raw.resolve().is_relative_to(imagefolder.resolve())
    with raw.open("rb") as source:
        pixels = np.load(source, allow_pickle=False)
    assert pixels.dtype == np.dtype("float64")
    np.testing.assert_array_equal(pixels, retained_case.pixels)
    assert digest(raw) == rows[0]["raw_array_sha256"]
    assert rows[0]["source_frame_sha256"] == digest(root / "frames/episode_0000/frame_0000.npz")

    html = (replay / "index.html").read_text()
    embedded = json.loads(html.split('<script type="application/json" id="experiment">', 1)[1]
                         .split("</script>", 1)[0])
    provenance = json.loads((replay / "provenance.json").read_text())
    assert len(embedded["episodes"]) == provenance["episode_count"] == 1
    assert embedded["exposure_count"] == provenance["exposure_count"] == 1
    assert provenance["html_sha256"] == digest(replay / "index.html")
    frames = embedded["episodes"][0]["frames"]
    assert len(frames) == 1
    assert frames[0]["measurement"] == retained_case.record["result"]["trajectory"][0]
    assert frames[0]["png"].startswith("data:image/png;base64,")
    png = base64.b64decode(frames[0]["png"].split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(png)) as replay_image:
        with Image.open(imagefolder / "images" / rows[0]["file_name"]) as export_image:
            assert replay_image.size == export_image.size == retained_case.pixels.shape[::-1]
            np.testing.assert_array_equal(np.asarray(replay_image), np.asarray(export_image))
    assert sequential.verify_run(root) == retained_case.manifest
