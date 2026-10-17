"""Build a portable, offline viewer from a trusted classical-selection export.

The required manifest fingerprint must come from an independent trusted receipt.
All exported data are checked before output is created. The viewer copies retained
PNGs only; it neither acquires observations nor reruns the native optical model.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

import export_selection_imagefolder as imagefolder


release = imagefolder.release
require = imagefolder.require
TEMPLATE = Path(__file__).parent / "templates/selection-replay.html"
PURPOSE = "classical_selection_offline_replay_v1"
CASE_FIELDS = ("id", *imagefolder.IDENTITY_FIELDS)
METRICS = {"residual_wfe_nm", "strehl_peak_ratio"}
MEASUREMENT_FIELDS = METRICS | {"index", "applied_opd_nm", "expected_photons",
                              "cumulative_expected_photons", "realized_photoelectrons"}
STATIC_FILES = {"images/metadata.jsonl", "episodes.jsonl", "configuration.json",
                "selected-settings.json", "provenance.json", "README.md", "LICENSE", "NOTICE"}


def fingerprint(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def label(value):
    return type(value) is str and 0 < len(value) <= 256


def metrics(value):
    require(type(value) is dict and set(value) == METRICS
            and all(number(item) and item >= 0 for item in value.values()),
            "Invalid evaluator metrics")


def command(value):
    require(type(value) is list and len(value) == 6 and all(number(item) for item in value),
            "Invalid applied command")


def json_lines(data, count, maximum):
    lines = data.splitlines()
    require(len(lines) == count and all(0 < len(line) <= maximum for line in lines),
            "Incomplete or unbounded JSONL inventory")
    for line in lines:
        value = release.read_json(line)
        require(type(value) is dict, "JSONL records must be objects")
        yield value


def validate_array(data, shape):
    """Validate the NPY header and byte length before interpreting any array data."""
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    require(version in {(1, 0), (2, 0)}, "Unsupported NPY format")
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    actual_shape, fortran, dtype = reader(stream)
    require(actual_shape == shape and fortran is False and dtype == np.dtype("float64"),
            "Raw measurements must be bounded C-order float64 images")
    offset = stream.tell()
    require(len(data) - offset == math.prod(shape) * 8, "Invalid raw image byte length")
    array = np.frombuffer(data, dtype=dtype, offset=offset).reshape(shape)
    require(np.isfinite(array).all(), "Raw image contains nonfinite values")
    return array


def validate_png(data, raw, photons):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format == "PNG" and image.mode == "RGBA"
                and image.size == (raw.shape[1], raw.shape[0]), "Invalid display PNG")
        actual = np.asarray(image)
    display = np.log1p(np.maximum(raw, 0)) / np.log1p(photons)
    expected = imagefolder.matplotlib.colormaps["magma"](np.clip(display, 0, 1), bytes=True)[::-1]
    require(np.array_equal(actual, expected), "PNG does not display its retained raw measurement")


def checked_manifest(data, sizes):
    manifest = release.read_json(data)
    require(type(manifest) is dict and type(manifest.get("schema")) is int
            and manifest["schema"] == 1 and manifest.get("purpose") == imagefolder.PURPOSE
            and manifest.get("split") == "public_tuning"
            and manifest.get("input_manifest_sha256") == release.NATIVE_MANIFEST_SHA256
            and fingerprint(manifest.get("packet_manifest_sha256")), "Invalid selection export manifest")
    for field, maximum in (("episode_count", 4032), ("exposure_count", 30912),
                           ("candidate_count", 14), ("independent_base_count", 12)):
        require(integer(manifest.get(field), 1, maximum), "Invalid bounded inventory count: " + field)
    files = release.hash_inventory(manifest.get("files"))
    require("manifest.json" not in files and set(sizes) == set(files) | {"manifest.json"},
            "Export file inventory does not match the manifest")
    require(len(files) <= 2 * 30912 + len(STATIC_FILES)
            and all(size <= release.MAX_FILE_BYTES for size in sizes.values())
            and sum(sizes.values()) <= 2 * 1024**3, "Export exceeds the release resource bounds")
    return manifest


def build_data(read, manifest, manifest_sha256):
    """Check complete episode/frame semantics and retain only browser-safe fields."""
    config = release.read_json(read("configuration.json"))
    settings = release.read_json(read("selected-settings.json"))
    provenance = release.read_json(read("provenance.json"))
    receipt = provenance["verification"]
    require(type(provenance.get("schema")) is int and provenance["schema"] == 1
            and provenance.get("purpose") == imagefolder.PURPOSE
            and provenance.get("split") == "public_tuning"
            and receipt.get("full_camera_replay_verified") is True
            and receipt.get("archive_manifest_sha256") == release.NATIVE_MANIFEST_SHA256
            and receipt.get("source_inventory_sha256") == release.SOURCE_INVENTORY_SHA256
            and receipt.get("packet_manifest_sha256") == manifest["packet_manifest_sha256"]
            and receipt.get("episodes") == manifest["episode_count"]
            and receipt.get("exposures") == manifest["exposure_count"],
            "Export provenance does not bind a complete native replay")
    optics = config["cohort"]["specification"]["optics"]
    require(integer(optics["pupil_pixels"], 1, 64) and integer(optics["oversampling"], 1, 64),
            "Invalid optical image dimensions")
    side = optics["pupil_pixels"] * optics["oversampling"]
    require(side <= 64, "Image dimensions exceed the release bound")
    candidates = config["candidates"]
    require(type(candidates) is list and len(candidates) == manifest["candidate_count"],
            "Candidate configuration count disagrees with the manifest")
    by_id = {}
    for candidate in candidates:
        require(type(candidate) is dict and label(candidate.get("id"))
                and label(candidate.get("family")) and type(candidate.get("parameters")) is dict
                and candidate["id"] not in by_id, "Invalid or duplicate candidate configuration")
        by_id[candidate["id"]] = candidate
    require(type(config["common_policy"]) is dict and type(settings.get("schema")) is int
            and settings["schema"] == 1 and settings.get("purpose") == "classical_parameter_selection_v1",
            "Invalid selected settings")
    families = settings["families"]
    require(type(families) is dict and set(families) == {row["family"] for row in candidates},
            "Selected settings must cover every candidate family")
    for family, choice in families.items():
        candidate = by_id.get(choice["candidate_id"])
        require(candidate is not None and candidate["family"] == family
                and choice["policy"] == {**config["common_policy"], **candidate["parameters"]},
                "Selected settings do not match their configured candidate")
    rows = {}
    for row in json_lines(read("images/metadata.jsonl"), manifest["exposure_count"], imagefolder.METADATA_ROW_BYTES):
        require(label(row.get("episode_id")) and integer(row.get("exposure_index"), 0, 12),
                "Invalid exposure identity")
        key = (row["episode_id"], row["exposure_index"])
        require(key not in rows, "Duplicate exposure metadata")
        rows[key] = row
    episodes, identities, cells, bases, used_candidates = [], set(), set(), {}, set()
    expected_files = set(STATIC_FILES)
    for episode in json_lines(read("episodes.jsonl"), manifest["episode_count"], 2 * 1024**2):
        require(set(episode) == {"episode_id", "record"}, "Invalid complete episode record")
        record = episode["record"]
        require(type(record) is dict and fingerprint(record.get("record_sha256"))
                and release.canonical_hash({key: value for key, value in record.items()
                                            if key != "record_sha256"}) == record["record_sha256"],
                "Episode record fingerprint disagrees with its contents")
        case, result = record["case"], record["result"]
        identity = case["id"]
        require(type(identity) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity)
                and episode["episode_id"] == identity and identity not in identities,
                "Invalid or duplicate episode identity")
        identities.add(identity)
        require(all(label(case[key]) for key in ("candidate_id", "family", "base_wavefront_id", "stratum_id"))
                and case["member"] in ("primary", "conjugate")
                and integer(case["photons_per_exposure"], 1, 10**9)
                and integer(case["exposure_budget"], 1, 13) and integer(case["replicate"], 0, 1023),
                "Invalid case identity")
        candidate, photons, budget = by_id.get(case["candidate_id"]), case["photons_per_exposure"], case["exposure_budget"]
        require(candidate is not None and candidate["family"] == case["family"]
                and photons in config["photons_per_exposure"] and budget in config["exposure_budgets"]
                and integer(config["replicates"], 1, 1024) and case["replicate"] < config["replicates"],
                "Case disagrees with its configuration")
        semantic = tuple(case[key] for key in imagefolder.IDENTITY_FIELDS)
        require(semantic not in cells, "Duplicate semantic case")
        cells.add(semantic)
        base = case["base_wavefront_id"]
        require(base not in bases or bases[base] == case["stratum_id"], "Inconsistent base stratum")
        bases[base] = case["stratum_id"]
        used_candidates.add(case["candidate_id"])
        metrics(result["initial"])
        metrics(result["final"])
        command(result["final_applied_opd_nm"])
        trajectory = result["trajectory"]
        require(type(trajectory) is list and len(trajectory) == budget
                and type(result["exposures"]) is int and result["exposures"] == budget
                and type(result["expected_photons_used"]) is int
                and result["expected_photons_used"] == photons * budget, "Incomplete exposure accounting")
        frame_hashes = record["frame_sha256"]
        require(type(frame_hashes) is dict and set(frame_hashes) == {f"frame_{i:04d}.npz" for i in range(budget)}
                and all(fingerprint(value) for value in frame_hashes.values()), "Invalid native frame inventory")
        selected = families[case["family"]]["candidate_id"] == case["candidate_id"]
        frames = []
        for index, measured in enumerate(trajectory):
            require(type(measured) is dict and set(measured) == MEASUREMENT_FIELDS
                    and type(measured["index"]) is int and measured["index"] == index
                    and type(measured["expected_photons"]) is int and measured["expected_photons"] == photons
                    and type(measured["cumulative_expected_photons"]) is int
                    and measured["cumulative_expected_photons"] == (index + 1) * photons
                    and integer(measured["realized_photoelectrons"], 0, 10**12), "Invalid measurement accounting")
            command(measured["applied_opd_nm"])
            metrics({key: measured[key] for key in METRICS})
            basename = f"{identity}_frame_{index:04d}"
            raw_name, image_name = f"raw/{basename}.npy", f"images/{basename}.png"
            expected = {"episode_id": identity, **{key: case[key] for key in imagefolder.IDENTITY_FIELDS},
                "split": "public_tuning", "exposure_index": index, "file_name": basename + ".png",
                "raw_array": raw_name, "raw_array_sha256": manifest["files"].get(raw_name),
                "source_frame_sha256": frame_hashes[f"frame_{index:04d}.npz"], "selected_reference": selected,
                **{key: measured[key] for key in ("applied_opd_nm", "expected_photons", "cumulative_expected_photons")},
                "evaluator_wfe_nm": measured["residual_wfe_nm"],
                "evaluator_strehl_peak_ratio": measured["strehl_peak_ratio"]}
            require(imagefolder.encoded(rows.pop((identity, index), None)) == imagefolder.encoded(expected),
                    "Image metadata disagrees with the complete episode")
            raw = validate_array(read(raw_name), (side, side))
            validate_png(read(image_name), raw, photons)
            expected_files.update((raw_name, image_name))
            frames.append({"image": image_name, "measurement": measured})
        indices = result["final_command_observation_indices"]
        actual_indices = [index for index, measured in enumerate(trajectory)
                          if measured["applied_opd_nm"] == result["final_applied_opd_nm"]]
        require(type(indices) is list and all(type(index) is int for index in indices)
                and indices == actual_indices
                and result["final_command_has_camera_observation"] is bool(indices)
                and result["final_command_is_last_observation"] is (budget - 1 in indices),
                "Final-command observation state disagrees with measured commands")
        require(all(result["final"] == {key: trajectory[index][key] for key in METRICS} for index in indices),
                "Final evaluator metrics disagree with the measured final command")
        fits = result["fit_records"]
        require(type(fits) is list and len(fits) <= 13
                and all(type(fit) is dict and fit.get("fit_status") in ("finite", "numerical_failure")
                        and type(fit.get("optimizer_success")) is bool
                        and (fit["fit_status"] != "numerical_failure" or not fit["optimizer_success"])
                        for fit in fits), "Invalid fit diagnostics")
        episodes.append({"case": {key: case[key] for key in CASE_FIELDS}, "selected_reference": selected,
            "frames": frames, "final": result["final"], "final_command": result["final_applied_opd_nm"],
            "measured_indices": indices, "initial": result["initial"], "fit_count": len(fits),
            "failed_fits": sum(fit["fit_status"] == "numerical_failure" for fit in fits),
            "unconverged_fits": sum(fit["fit_status"] == "finite" and not fit["optimizer_success"] for fit in fits)})
    require(not rows and used_candidates == set(by_id) and len(bases) == manifest["independent_base_count"]
            and expected_files == set(manifest["files"]), "Incomplete episode, candidate, base or file inventory")
    return {"schema": 1, "split": "public_tuning", "episodes": episodes,
            "manifest_sha256": manifest_sha256, "native_manifest_sha256": release.NATIVE_MANIFEST_SHA256,
            **{key: manifest[key] for key in ("exposure_count", "independent_base_count", "candidate_count")}}


def render(data, template):
    require(template.count("%%DATA%%") == 1 and template.count("%%SCRIPT_HASH%%") == 1,
            "Replay template must contain each substitution exactly once")
    scripts = re.findall(r'<script id="replay-script">(.*?)</script>', template, flags=re.S)
    require(len(scripts) == 1 and "%%DATA%%" not in scripts[0] and "%%SCRIPT_HASH%%" not in scripts[0],
            "Replay template must contain one static script")
    script_hash = base64.b64encode(hashlib.sha256(scripts[0].encode()).digest()).decode()
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return template.replace("%%SCRIPT_HASH%%", script_hash).replace("%%DATA%%", encoded).encode(), script_hash


def generate(export, manifest_sha256, output):
    require(fingerprint(manifest_sha256), "An independently trusted SHA256 manifest fingerprint is required")
    root = release.directory(export).resolve()
    output = imagefolder.output_path(output, root, root)
    sizes = imagefolder.native_inventory(root)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        manifest = checked_manifest(imagefolder.read_native(directory_fd, "manifest.json", manifest_sha256), sizes)

        def read(name):
            require(name in manifest["files"], "Missing manifested input: " + name)
            return imagefolder.read_native(directory_fd, name, manifest["files"][name])

        for name in sorted(manifest["files"]):
            read(name)
        data = build_data(read, manifest, manifest_sha256)
        template_bytes = release.regular_bytes(TEMPLATE)
        html, script_hash = render(data, template_bytes.decode())
        provenance = {"schema": 1, "purpose": PURPOSE, "split": "public_tuning",
            "input_manifest_sha256": manifest_sha256,
            "native_manifest_sha256": release.NATIVE_MANIFEST_SHA256,
            "packet_manifest_sha256": manifest["packet_manifest_sha256"],
            "generator_sha256": release.digest(release.regular_bytes(Path(__file__))),
            "imagefolder_helper_sha256": release.digest(release.regular_bytes(Path(imagefolder.__file__))),
            "verifier_helper_sha256": release.digest(release.regular_bytes(Path(release.__file__))),
            "template_sha256": release.digest(template_bytes), "script_sha256_base64": script_hash,
            "verification_scope": "Complete derived-export inventory, hashes and data consistency; native physics and optimization are not rerun.",
            "images": "Unchanged recorded PNGs; no synthetic final image and no raw-array copies.",
            "display": "log1p(max(electrons,0))/log1p(photons_per_exposure), fixed 0–1 magma, origin lower",
            "reserve_bytes": imagefolder.RESERVE_BYTES}
        assets = {"index.html": html, "provenance.json": imagefolder.encoded(provenance),
                  **{name: read(name) for name in ("LICENSE", "NOTICE")}}
        images = sorted(name for name in manifest["files"] if name.startswith("images/") and name.endswith(".png"))
        block = max(4096, os.statvfs(output.parent).f_frsize)
        estimate = (sum(imagefolder.allocation(sizes[name], block) + block for name in images)
                    + sum(imagefolder.allocation(len(value), block) + block for value in assets.values())
                    + (len(images) + len(assets)) * 256 + 16 * 1024**2)
        imagefolder.check_space(output.parent, estimate)
        output.mkdir(exist_ok=False)
        (output / "images").mkdir()
        hashes = {}
        for name in images:
            imagefolder.write_new(output / name, read(name), output, block, hashes)
        for name, value in assets.items():
            imagefolder.write_new(output / name, value, output, block, hashes)
        completed = {"schema": 1, "purpose": PURPOSE, "split": "public_tuning",
            "input_manifest_sha256": manifest_sha256, "native_manifest_sha256": release.NATIVE_MANIFEST_SHA256,
            **{key: manifest[key] for key in ("episode_count", "exposure_count", "candidate_count", "independent_base_count")},
            "files": hashes}
        manifest_bytes = imagefolder.encoded(completed)
        require(len(manifest_bytes) <= (len(images) + len(assets)) * 256, "Replay manifest exceeds its allocation bound")
        imagefolder.write_new(output / ".manifest.pending", manifest_bytes, output, block, {})
        os.replace(output / ".manifest.pending", output / "manifest.json")
        return completed
    finally:
        os.close(directory_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = generate(args.export, args.manifest_sha256, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Selection replay refused: {error}\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
