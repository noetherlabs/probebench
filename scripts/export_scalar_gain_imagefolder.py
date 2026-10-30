"""Export the pinned scalar command-response release without executing its sources.

Only the released 288-episode, 2,016-exposure archive is accepted. Verification
establishes byte identity and export consistency, not a new optical replay.
Requested nominal commands never become legacy ``applied_opd_nm`` observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import math
import os
from pathlib import Path
import platform
import re
import zipfile

import export_selection_imagefolder as shared

import matplotlib.image
import numpy as np
from PIL import __version__ as pillow_version


release = shared.release
require, encoded = shared.require, shared.encoded
MANIFEST_SHA256 = "ee56441561fa50aeb80f1e33e516cf9151248445f75d997e0d46e9b5002c3322"
CONFIG_SHA256 = "47145e03b9c96bc75256cd56d27ed55400ee5131c525fe57410fd76eaad5c5e8"
SOURCE_REVISION = "99e5e883c9e064fdf778d8cbea7e439e7f09e91b"
NATIVE_PURPOSE = "scalar_gain_development_v1"
PURPOSE = "scalar_command_response_imagefolder_export_v1"
CONFIG_NAME, SPLIT = "scalar_command_response_v1", "development"
OBSERVATION_SCHEMA = "probebench.nominal-observation/1"
COHORT = "nominal_calibration_development_v1"
STRATA, MEMBERS, GAINS = ("low", "medium", "high"), ("primary", "conjugate"), (0.9, 1.0, 1.1)
CANDIDATES = {"fixed_probe_080": "fixed_diversity_mle_v1", "adaptive_probe_080": "phase_diversity_mle_v1"}
BASES_PER_STRATUM, EPISODES, EXPOSURES = 8, 288, 2016
FRAMES_PER_EPISODE, PHOTONS, SHAPE = 7, 2000, (64, 64)
CASE_JSON_BYTES, METADATA_ROW_BYTES = 65536, 4096
NOMINAL_CALIBRATION = {"response": "identity", "command_units": "nm_nominal_opd", "prediction_units": "nm_opd"}


def verify_archive(root, directory_fd):
    """Verify the complete immutable release, including non-executed source files."""
    manifest_bytes = shared.read_native(directory_fd, "manifest.json", MANIFEST_SHA256)
    manifest = release.read_json(manifest_bytes)
    require(manifest.get("schema") == 1 and manifest.get("purpose") == NATIVE_PURPOSE
            and manifest.get("episode_count") == EPISODES and manifest.get("exposure_count") == EXPOSURES,
            "Unexpected scalar release identity or counts")
    files = release.hash_inventory(manifest.get("files"))
    sizes = shared.native_inventory(root)
    require(set(sizes) == set(files) | {"manifest.json"}, "Incomplete, additional or pending native artifact")
    payload = {}
    retained = {"study.json", "cases.json", "preparation.json", "summary.json", "analysis.json"}
    for name, expected in sorted(files.items()):
        data = shared.read_native(directory_fd, name, expected)
        if name in retained:
            payload[name] = data
    require(set(payload) == retained, "Missing scalar release metadata")
    study, cases = (release.read_json(payload[name]) for name in ("study.json", "cases.json"))
    configuration = study["configuration"]
    protocol, admission = configuration["protocol"], configuration["admission"]
    require(release.canonical_hash(protocol) == CONFIG_SHA256
            and admission["configuration_sha256"] == CONFIG_SHA256
            and admission["source_revision"] == SOURCE_REVISION and admission["status"] == "FROZEN"
            and study["source_sha256"] == admission["source_sha256"] == manifest["source_sha256"]
            and study["runtime"] == admission["runtime"], "Frozen configuration/source/runtime binding changed")
    require(type(cases) is list and len(cases) == EPISODES, "Incomplete scalar case inventory")
    identities = set()
    for case in cases:
        require(type(case) is dict and type(case.get("id")) is str
                and re.fullmatch(r"case_[0-9a-f]{64}", case["id"]), "Invalid scalar case identity")
        require(case["cohort_id"] == COHORT and case["replicate"] == 0
                and case["base_wavefront_id"] == f"{COHORT}:{case['stratum_id']}:{case['base_index']:06d}"
                and CANDIDATES.get(case["candidate_id"]) == case["family"], "Scalar case labels disagree")
        identity = {key: case[key] for key in ("cohort_id", "base_index", "stratum_id", "member", "replicate",
                    "base_wavefront_id", "gain", "candidate_id", "family")}
        require(case["id"] == "case_" + release.canonical_hash(identity), "Scalar case ID does not bind its design")
        identities.add((case["stratum_id"], case["base_index"], case["member"], case["gain"], case["candidate_id"]))
    expected_grid = {(s, b, m, g, c) for s in STRATA for b in range(BASES_PER_STRATUM)
                     for m in MEMBERS for g in GAINS for c in CANDIDATES}
    require(identities == expected_grid and len(identities) == EPISODES
            and len({case["id"] for case in cases}) == EPISODES, "Scalar design grid is incomplete or duplicated")
    expected_files = {".lock", *retained, *("cases/" + case["id"] + ".npz" for case in cases)}
    for prefix, source_files in (("source/", study["source_sha256"]),
                                 ("reviewed-sources/", protocol["reviewed_fixture_sources"])):
        release.hash_inventory(source_files)
        for name, expected in source_files.items():
            require(files.get(prefix + name) == expected, "Retained source binding changed")
            expected_files.add(prefix + name)
    require(set(files) == expected_files, "Native inventory differs from the scalar protocol")
    return {"manifest": manifest, "manifest_bytes": manifest_bytes, "payload": payload,
            "study": study, "cases": cases, "sizes": sizes}


def decode_case(data):
    """Validate bounded NPY headers before loading verified in-memory NPZ bytes."""
    case_limit = FRAMES_PER_EPISODE * math.prod(SHAPE) * 8 + CASE_JSON_BYTES + 2048
    require(len(data) <= case_limit, "Oversized scalar case archive")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        require(len(members) == 2 and {item.filename for item in members} == {"images.npy", "metadata.npy"},
                "Unexpected scalar NPZ members")
        for item in members:
            require(item.compress_type == zipfile.ZIP_STORED and item.file_size <= case_limit,
                    "Compressed or oversized scalar array")
            with archive.open(item) as stream:
                require(np.lib.format.read_magic(stream) == (1, 0), "Unsupported NPY version")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                valid = (shape == (FRAMES_PER_EPISODE, *SHAPE) and dtype == np.dtype("<f8")) if item.filename == "images.npy" else (
                    len(shape) == 1 and 0 < shape[0] <= CASE_JSON_BYTES and dtype == np.dtype("uint8"))
                require(valid and not fortran and stream.tell() + math.prod(shape)*dtype.itemsize == item.file_size,
                        "Invalid scalar array layout")
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        record, images = release.read_json(saved["metadata"].tobytes()), saved["images"].copy()
    require(np.isfinite(images).all(), "Nonfinite detector values in the released completed cohort")
    return record, images


def check_record(study, case, record, images):
    require(record["schema"] == 1 and record["purpose"] == NATIVE_PURPOSE and record["case"] == case
            and record["case_binding"] == release.canonical_hash({"study_sha256": release.canonical_hash(study), "case": case})
            and record["images_sha256"] == release.digest(images.tobytes()), "Scalar case/image binding changed")
    observations, evaluation = record["observations"], record["evaluation"]
    physical = evaluation["physical"]
    require(record["failed_observation"] is None and evaluation["status"] == "submitted"
            and evaluation["schema"] == "probebench.scalar-gain-evaluation/1"
            and evaluation["command_gain"] == case["gain"]
            and evaluation["nominal_calibration"] == NOMINAL_CALIBRATION
            and evaluation["nominal_observations"] == observations,
            "Unsupported terminal record or nominal calibration")
    require(len(observations) == len(physical["trajectory"]) == FRAMES_PER_EPISODE
            and physical["exposures"] == FRAMES_PER_EPISODE
            and physical["expected_photons_used"] == FRAMES_PER_EPISODE*PHOTONS,
            "Incomplete exposure/photon inventory")
    fields = {"schema", "index", "commanded_opd_nm", "nominal_applied_opd_nm", "nominal_calibration",
              "expected_photons", "cumulative_expected_photons"}
    for index, (observation, measured) in enumerate(zip(observations, physical["trajectory"])):
        require(set(observation) == fields and observation["schema"] == OBSERVATION_SCHEMA
                and observation["nominal_calibration"] == NOMINAL_CALIBRATION
                and observation["commanded_opd_nm"] == observation["nominal_applied_opd_nm"],
                "Unsupported nominal-observation schema or command semantics")
        command = np.asarray(observation["commanded_opd_nm"])
        require(command.shape == (6,) and np.issubdtype(command.dtype, np.number)
                and not np.iscomplexobj(command) and np.isfinite(command).all()
                and np.max(np.abs(command)) <= 200, "Invalid nominal command")
        require(np.array_equal(measured["applied_opd_nm"], case["gain"]*command),
                "Physical evaluator command differs from nominal response")
        require(all(item["index"] == index and item["expected_photons"] == PHOTONS
                    and item["cumulative_expected_photons"] == (index+1)*PHOTONS
                    for item in (observation, measured)), "Exposure metadata disagrees")
        require(all(type(measured[key]) in (int, float) and math.isfinite(measured[key]) and measured[key] >= 0
                    for key in ("residual_wfe_nm", "strehl_peak_ratio")), "Invalid retained evaluator label")


def storage_estimate(block, metadata_bytes):
    pixels = math.prod(SHAPE)
    frame_bytes = EXPOSURES * (shared.allocation(pixels*8 + 256, block)
        + shared.allocation(pixels*4 + 1024, block) + METADATA_ROW_BYTES + 2*block)
    return frame_bytes + EPISODES*(2*CASE_JSON_BYTES + 512) + (2*EXPOSURES + 12)*256 + metadata_bytes + 16*1024**2


def dataset_card():
    return f'''---
language: [en]
license: apache-2.0
pretty_name: ProbeBench scalar command response v1
configs:
- config_name: {CONFIG_NAME}
  data_files:
  - split: {SPLIT}
    path: images/**
---

# ProbeBench scalar command response

This local export preserves {EPISODES} episodes and {EXPOSURES} acquired exposures
from the released descriptive development study. It has 24 independent base
wavefronts, both conjugates, command gains 0.9/1.0/1.1 and two frozen schedules.
It is not a held-out test set, hardware validation or a new scientific result.

`observation_schema` is `{OBSERVATION_SCHEMA}`. `commanded_opd_nm` is the requested
nominal OPD command; `nominal_applied_opd_nm` is the unit-calibration prediction,
not measured actuation. `nominal_calibration` records those units and assumptions.
There is deliberately no legacy `applied_opd_nm` observation column. Photon
counters and `exposure_index` describe the actual retained camera acquisition.

Columns prefixed `evaluator_` contain true gain, physical actuation, physical WFE
and Strehl labels. Those fields, cohort/stratum identities, and episode identities
must not be supplied as controller observations. `episodes.jsonl` preserves the
complete native records, including latent truth, seeds, final-command measurement
status and all fit diagnostics. It is outside the image data-file mapping.
`configuration.json`, `preparation.json`, `summary.json` and `analysis.json` are
also evaluator/reproduction material, not controller inputs.

`raw_array` is relative to this export root. Its NPY file preserves signed
little-endian float64 detector electrons. `raw_array_sha256` hashes the whole NPY
file; `image_electrons_sha256` hashes its C-order sample bytes. `source_case_sha256`
binds the native stacked-image NPZ. PNGs use log1p(max(electrons,0))/log1p(2000),
fixed limits 0–1, magma, and lower origin. Display clipping never changes the NPY.
No corrected image is invented for an unmeasured final command.

Load the local ImageFolder explicitly with split `development`, then verify the
NPY digest before using detector values. See the repository's scalar export-format
documentation for an observation-only projection. The `{CONFIG_NAME}` configuration
is separate from the existing classical-selection `default/public_tuning` release;
do not append these rows to that incompatible schema or move its release tag.

`source-manifest.json` retains the exact pinned native manifest. `manifest.json`
inventories this derived export. Verification checks bytes, schema and accounting;
it does not rerun optimization, bootstrap reductions, camera replay or archived
Python code. The raw release and its separate audit retain those claim boundaries.

Noether Labs. *Calibration mismatch changes the preferred measurement schedule.*
2026. https://github.com/noetherlabs/probebench/releases/tag/scalar-command-response-v1-20261028
'''.encode()


def export(archive, output):
    archive = release.directory(archive).resolve()
    output = shared.output_path(output, archive, archive)
    source_files = (Path(__file__), Path(shared.__file__), Path(release.__file__))
    export_sources = {path.name: release.digest(path.read_bytes()) for path in source_files}
    with shared.archive_lock(archive) as directory_fd:
        checked = verify_archive(archive, directory_fd)
        manifest, study, cases = checked["manifest"], checked["study"], checked["cases"]
        block = max(4096, os.statvfs(output.parent).f_frsize)
        estimate = storage_estimate(block, sum(map(len, checked["payload"].values())))
        shared.check_space(output.parent, estimate)
        output.mkdir(exist_ok=False)
        (output / "images").mkdir(); (output / "raw").mkdir()
        hashes, frame_count, negative_samples = {}, 0, 0
        metadata_hash, episode_hash = hashlib.sha256(), hashlib.sha256()
        with (output / "images/metadata.jsonl").open("xb", buffering=0) as metadata, (output / "episodes.jsonl").open("xb", buffering=0) as episodes:
            for case in cases:
                source_name = f"cases/{case['id']}.npz"
                source_hash = manifest["files"][source_name]
                record, images = decode_case(shared.read_native(directory_fd, source_name, source_hash))
                check_record(study, case, record, images)
                line = encoded({"episode_id": case["id"], "record": record})
                require(len(line) <= 2*CASE_JSON_BYTES + 512, "Oversized episode metadata")
                shared.check_space(output, shared.allocation(len(line), block) + block)
                require(episodes.write(line) == len(line), "Incomplete episode write")
                episode_hash.update(line)
                for index, (observation, measured, image) in enumerate(zip(record["observations"], record["evaluation"]["physical"]["trajectory"], images)):
                    basename = f"{case['id']}_frame_{index:04d}"
                    raw_buffer, png_buffer = io.BytesIO(), io.BytesIO()
                    np.save(raw_buffer, image, allow_pickle=False)
                    display = np.log1p(np.maximum(image, 0)) / np.log1p(PHOTONS)
                    matplotlib.image.imsave(png_buffer, display, format="png", origin="lower", cmap="magma", vmin=0, vmax=1)
                    raw_bytes, png_bytes = raw_buffer.getvalue(), png_buffer.getvalue()
                    require(len(raw_bytes) <= image.size*8 + 256 and len(png_bytes) <= image.size*4 + 1024,
                            "Rendered frame exceeds admitted storage")
                    require(np.load(io.BytesIO(raw_bytes), allow_pickle=False).tobytes() == image.tobytes(),
                            "Signed detector bytes changed during serialization")
                    raw_name = f"raw/{basename}.npy"
                    shared.write_new(output / raw_name, raw_bytes, output, block, hashes)
                    shared.write_new(output / f"images/{basename}.png", png_bytes, output, block, hashes)
                    entry = {"file_name": basename + ".png", "episode_id": case["id"], "exposure_index": index,
                        "split": SPLIT, "observation_schema": observation["schema"],
                        **{key: observation[key] for key in ("commanded_opd_nm", "nominal_applied_opd_nm", "nominal_calibration",
                                                            "expected_photons", "cumulative_expected_photons")},
                        **{key: case[key] for key in ("candidate_id", "family", "base_wavefront_id", "stratum_id", "member", "replicate")},
                        "exposure_budget": FRAMES_PER_EPISODE, "evaluator_command_gain": case["gain"],
                        "evaluator_applied_opd_nm": measured["applied_opd_nm"],
                        "evaluator_wfe_nm": measured["residual_wfe_nm"], "evaluator_strehl_peak_ratio": measured["strehl_peak_ratio"],
                        "raw_array": raw_name, "raw_array_sha256": release.digest(raw_bytes),
                        "image_electrons_sha256": release.digest(image.tobytes()), "source_case_sha256": source_hash}
                    line = encoded(entry)
                    require(len(line) <= METADATA_ROW_BYTES, "Oversized frame metadata")
                    shared.check_space(output, shared.allocation(len(line), block) + block)
                    require(metadata.write(line) == len(line), "Incomplete frame metadata write")
                    metadata_hash.update(line)
                    frame_count += 1
                    negative_samples += int(np.count_nonzero(image < 0))
        require(frame_count == EXPOSURES, "Export lost acquired exposures")
        hashes["images/metadata.jsonl"], hashes["episodes.jsonl"] = metadata_hash.hexdigest(), episode_hash.hexdigest()
        provenance = {"schema": 1, "purpose": PURPOSE, "input_manifest_sha256": MANIFEST_SHA256,
            "configuration_sha256": CONFIG_SHA256, "acquisition_source_revision": SOURCE_REVISION,
            "verification": "Pinned complete archive hashes, nominal schema, acquired-frame accounting and raw-byte preservation; no new scientific replay",
            "export_source_sha256": export_sources,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__, "pillow": pillow_version},
            "storage_admission": {"estimated_output_bytes": estimate, "reserve_bytes": shared.RESERVE_BYTES, "allocation_block_bytes": block},
            "display": {"transform": "log1p(max(electrons,0))/log1p(expected_photons)", "minimum": 0, "maximum": 1,
                        "colormap": "magma", "origin": "lower", "raw_dtype": "<f8", "raw_shape": list(SHAPE)},
            "negative_detector_samples_preserved": negative_samples}
        copies = {"source-manifest.json": checked["manifest_bytes"], "study.json": checked["payload"]["study.json"],
            **{name: checked["payload"][name] for name in ("preparation.json", "summary.json", "analysis.json")},
            "configuration.json": encoded(study["configuration"]), "provenance.json": encoded(provenance), "README.md": dataset_card(),
            **{name: release.regular_bytes(Path(__file__).parents[1] / name) for name in ("LICENSE", "NOTICE")}}
        for name, data in copies.items():
            shared.write_new(output / name, data, output, block, hashes)
        verify_archive(archive, directory_fd)
        require(set(shared.native_inventory(output)) == set(hashes), "Output inventory changed during export")
        require(all(release.digest(release.regular_bytes(output / name)) == expected for name, expected in hashes.items()),
                "Output bytes differ from completed writes")
        require(export_sources == {path.name: release.digest(path.read_bytes()) for path in source_files},
                "Exporter source changed during conversion")
        result = {"schema": 1, "purpose": PURPOSE, "config_name": CONFIG_NAME, "split": SPLIT,
            "input_manifest_sha256": MANIFEST_SHA256, "configuration_sha256": CONFIG_SHA256,
            "episode_count": EPISODES, "exposure_count": EXPOSURES, "expected_photons_consumed": EXPOSURES*PHOTONS,
            "independent_base_count": len(STRATA)*BASES_PER_STRATUM, "negative_detector_samples_preserved": negative_samples,
            "episodes_per_gain": dict(Counter(str(case["gain"]) for case in cases)), "files": hashes}
        pending = output / "manifest.json.pending"
        shared.write_new(pending, encoded(result), output, block, {})
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        shared.check_space(output)
        os.replace(pending, output / "manifest.json")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Extracted scalar-command-response-v1 directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; source remains unchanged")
    args = parser.parse_args()
    export(args.archive, args.output)
