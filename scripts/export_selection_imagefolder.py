"""Export the pinned classical-selection v1 archive to a local ImageFolder dataset.

Full replay uses the independently verified source snapshot. Exported PNGs are
display views of retained measurements; signed electron arrays remain separate.
The archive is shared-locked throughout verification and export. Partial outputs
have no completed manifest and cannot be resumed or overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat

import matplotlib
matplotlib.use("Agg")
import matplotlib.image
import numpy as np
from PIL import __version__ as pillow_version

import verify_selection_release as release


RESERVE_BYTES = 1_342_177_280
METADATA_ROW_BYTES = 2048
PURPOSE = "classical_selection_imagefolder_export_v1"
IDENTITY_FIELDS = ("candidate_id", "family", "photons_per_exposure", "exposure_budget",
                   "base_wavefront_id", "member", "stratum_id", "replicate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def check_space(path, required_bytes=0):
    if shutil.disk_usage(path).free < RESERVE_BYTES + required_bytes:
        raise ValueError("Insufficient storage: export must preserve the operating reserve")


def allocation(size, block):
    return ((size + block - 1) // block) * block


def output_path(value, packet, archive):
    path = Path(value).absolute()
    require(".." not in path.parts, "Output must not contain parent traversal")
    require(not path.exists() and not path.is_symlink(), "Output must be a new directory")
    release.directory(path.parent)
    for source in (packet, archive):
        require(not path.is_relative_to(source) and not source.is_relative_to(path),
                "Output must be disjoint from the packet and native archive")
        require(not any(source.samefile(parent) for parent in (path.parent, *path.parent.parents)),
                "Output must be disjoint from the packet and native archive")
    return path


def native_inventory(root):
    files = {}
    for path in root.rglob("*"):
        info = path.lstat()
        require(stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode),
                "Archive entries must be regular files or directories")
        if stat.S_ISREG(info.st_mode):
            files[release.safe_name(path.relative_to(root).as_posix())] = info.st_size
    return files


@contextmanager
def archive_lock(root):
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        lock_fd = os.open(".lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        try:
            require(stat.S_ISREG(os.fstat(lock_fd).st_mode), "Archive lock must be a regular file")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Archive is busy; export requires an inactive writer") from error
            yield directory_fd
        finally:
            os.close(lock_fd)
    finally:
        os.close(directory_fd)


def read_native(directory_fd, name, expected):
    """Read and hash exactly the bytes consumed, without following linked components."""
    parts = Path(release.safe_name(name)).parts
    parent = os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_size <= release.MAX_FILE_BYTES,
                    "Native input must be a bounded regular file")
            data = stream.read(release.MAX_FILE_BYTES + 1)
            require(len(data) <= release.MAX_FILE_BYTES, "Native input exceeds the size bound")
        require(release.digest(data) == expected, "Native input changed after verification: " + name)
        return data
    finally:
        os.close(parent)


def storage_estimate(rows, native_sizes, shape, block):
    """Bound raw arrays, uncompressed RGBA PNGs, JSONL, hashes and allocation overhead."""
    exposures = sum(row["exposure_budget"] for row in rows)
    pixels = math.prod(shape)
    raw_bound, png_bound = pixels * 8 + 256, pixels * 4 + 1024
    frames = exposures * (allocation(raw_bound, block) + allocation(png_bound, block)
                          + METADATA_ROW_BYTES + 2 * block)
    episodes = sum(2 * native_sizes[f"cases/{row['id']}.json"] + 512 for row in rows)
    manifest = (2 * exposures + 8) * 256
    return frames + episodes + manifest + 16 * 1024**2


def write_new(path, data, root, block, hashes):
    release.directory(path.parent)
    check_space(root, allocation(len(data), block) + block)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    hashes[path.relative_to(root).as_posix()] = release.digest(data)


def dataset_card(episodes, exposures, candidates, bases):
    return f'''---
language:
- en
license: apache-2.0
pretty_name: ProbeBench classical reference selection v1
tags:
- optics
- wavefront-sensing
- scientific-computing
configs:
- config_name: default
  data_files:
  - split: public_tuning
    path: images/**
---

# ProbeBench classical reference selection v1

This export retains {episodes} episodes and {exposures} acquired images across all
{candidates} candidates and {bases} independent base wavefronts. The split is
`public_tuning`: these are public parameter-selection data, not training/test splits,
held-out performance estimates, telescope measurements or language-model results.

`images/metadata.jsonl` preserves candidate, family, photon/exposure cell, base,
conjugate member, RMS stratum and detector-replicate identity. `selected_reference`
identifies the single global choice per family; non-winning candidates remain included.
The command and expected-photon fields describe the actual exposure. Columns prefixed
`evaluator_` are latent simulator labels and must not be supplied as controller inputs.
Cohort identity and selection flags are analysis labels, not additional observations.

`raw/` stores the corresponding signed float64 electron arrays. `raw_array` paths are
relative to this export root, while `file_name` is relative to `images/`. PNGs use
log1p(max(electrons, 0))/log1p(photons per exposure), fixed limits 0–1, the magma map,
and image origin lower. Display clipping does not alter the raw arrays. No final image
is invented for an unmeasured command; every image corresponds to a retained exposure.

`episodes.jsonl` preserves complete native case records under `record`, including
latent wavefronts, seeds, all optimizer diagnostics, and final-command observation
indices. A final command measured earlier is distinct from the last exposure and from
an unmeasured submission. Episode records and `configuration.json` contain evaluator
information and are not observation inputs. `selected-settings.json` preserves the
six globally selected configurations. `provenance.json` binds the source fingerprints,
full replay receipt, generator and rendering dependencies. `manifest.json` inventories
this export separately from the original archive.

## Load locally without a second image cache

```python
from pathlib import Path
from datasets import load_dataset
import numpy as np

root = Path("work/selection-imagefolder")
frames = load_dataset("imagefolder",
    data_files={{"public_tuning": str(root / "images" / "**")}},
    split="public_tuning", streaming=True)
first = next(iter(frames))
electrons = np.load(root / first["raw_array"], allow_pickle=False)
```

The layout follows the official [ImageFolder metadata format](https://huggingface.co/docs/datasets/image_dataset)
and [local loading configuration](https://huggingface.co/docs/datasets/loading).

The model is static monochromatic scalar Fourier optics with six controlled modes,
an exactly matched estimator, and Poisson–Gaussian detector noise. Conjugates,
replicates and shared budget prefixes are correlated; they do not add independent
wavefronts. Keep all numerical failures and non-winning results in analyses. Recorded
decision times do not establish comparable algorithm latency or real-time guarantees.
Hashes establish consistency with the pinned release, not authorship. Shared file
locking excludes cooperating writers; keep the input tree unchanged during export.
An interrupted export has no completed manifest and must not be treated as a dataset.
'''.encode()


def export(packet, archive, output):
    packet, archive = release.directory(packet).resolve(), release.directory(archive).resolve()
    output = output_path(output, packet, archive)
    with archive_lock(archive) as directory_fd:
        sizes = native_inventory(archive)
        verified = release.verify(packet, archive=archive)
        checked = release.validate_packet(packet)
        native, payload, study = checked["native"], checked["bytes"], checked["study"]
        require(verified.get("full_camera_replay_verified") is True
                and verified.get("archive_manifest_sha256") == release.NATIVE_MANIFEST_SHA256
                and verified.get("source_inventory_sha256") == release.SOURCE_INVENTORY_SHA256
                and verified.get("packet_manifest_sha256") == checked["packet_manifest_sha256"]
                and verified.get("episodes") == native["episode_count"]
                and verified.get("exposures") == native["exposure_count"],
                "Export requires a matching complete native replay receipt")
        require(set(sizes) == set(native["files"]) | {"manifest.json"},
                "Native archive inventory changed after verification")
        read_native(directory_fd, "manifest.json", release.NATIVE_MANIFEST_SHA256)
        rows = release.read_json(payload["summary.json"])
        settings = release.read_json(payload["selected-settings.json"])
        require(type(rows) is list and len(rows) == native["episode_count"] and rows,
                "Complete selection summary is required")
        ids = [row["id"] for row in rows]
        require(len(set(ids)) == len(ids) and all(type(i) is str and re.fullmatch(r"[A-Za-z0-9_-]+", i)
                                                  for i in ids), "Unsafe or repeated episode identity")
        require(all(type(row["exposure_budget"]) is int and row["exposure_budget"] > 0 for row in rows)
                and sum(row["exposure_budget"] for row in rows) == native["exposure_count"],
                "Exposure inventory disagrees with the native archive")
        expected = {f"cases/{row['id']}.json" for row in rows}
        expected.update(f"frames/{row['id']}/frame_{index:04d}.npz"
                        for row in rows for index in range(row["exposure_budget"]))
        require(expected == {name for name in native["files"] if name.startswith(("cases/", "frames/"))},
                "Case/frame inventory disagrees with the complete summary")
        optics = study["configuration"]["cohort"]["specification"]["optics"]
        side = optics["pupil_pixels"] * optics["oversampling"]
        require(type(side) is int and 1 <= side <= 4096, "Invalid bounded image dimensions")
        shape = (side, side)
        block = max(4096, os.statvfs(output.parent).f_frsize)
        estimate = storage_estimate(rows, sizes, shape, block)
        check_space(output.parent, estimate)
        output.mkdir(exist_ok=False)
        (output / "images").mkdir()
        (output / "raw").mkdir()
        hashes, frame_count = {}, 0
        metadata_hash, episodes_hash = hashlib.sha256(), hashlib.sha256()
        with (output / "images/metadata.jsonl").open("xb", buffering=0) as metadata, (output / "episodes.jsonl").open("xb", buffering=0) as episodes:
            for row in rows:
                identity = row["id"]
                case_name = f"cases/{identity}.json"
                case_bytes = read_native(directory_fd, case_name, native["files"][case_name])
                record = release.read_json(case_bytes)
                case, result = record["case"], record["result"]
                require(case["id"] == identity and all(case[key] == row[key] for key in IDENTITY_FIELDS),
                        "Case identity disagrees with the complete summary")
                require(len(result["trajectory"]) == case["exposure_budget"]
                        and result["exposures"] == case["exposure_budget"], "Incomplete episode trajectory")
                line = encoded({"episode_id": identity, "record": record})
                require(len(line) <= 2 * len(case_bytes) + 512, "Episode exceeds admitted metadata bound")
                check_space(output, allocation(len(line), block) + block)
                require(episodes.write(line) == len(line), "Incomplete episode metadata write")
                episodes_hash.update(line)
                for index, measured in enumerate(result["trajectory"]):
                    source_name = f"frames/{identity}/frame_{index:04d}.npz"
                    frame_bytes = read_native(directory_fd, source_name, native["files"][source_name])
                    with np.load(io.BytesIO(frame_bytes), allow_pickle=False) as saved:
                        require(set(saved.files) == {"image_electrons", "record_json", "record_sha256"},
                                "Unexpected native frame schema")
                        image = saved["image_electrons"].copy()
                    require(image.dtype == np.dtype("float64") and image.shape == shape
                            and np.isfinite(image).all(), "Invalid signed detector array")
                    photons = case["photons_per_exposure"]
                    require(type(photons) is int and photons > 0 and measured["index"] == index
                            and measured["expected_photons"] == photons
                            and measured["cumulative_expected_photons"] == (index + 1) * photons,
                            "Exposure accounting changed")
                    basename = f"{identity}_frame_{index:04d}"
                    raw_buffer, png_buffer = io.BytesIO(), io.BytesIO()
                    np.save(raw_buffer, image, allow_pickle=False)
                    display = np.log1p(np.maximum(image, 0)) / np.log1p(photons)
                    matplotlib.image.imsave(png_buffer, display, format="png", origin="lower",
                                           cmap="magma", vmin=0, vmax=1)
                    raw_bytes, png_bytes = raw_buffer.getvalue(), png_buffer.getvalue()
                    require(len(raw_bytes) <= image.size * 8 + 256
                            and len(png_bytes) <= image.size * 4 + 1024, "Rendered image exceeds admission bound")
                    raw_name = f"raw/{basename}.npy"
                    write_new(output / raw_name, raw_bytes, output, block, hashes)
                    write_new(output / f"images/{basename}.png", png_bytes, output, block, hashes)
                    entry = {"episode_id": identity, **{key: case[key] for key in IDENTITY_FIELDS},
                        "split": "public_tuning", "exposure_index": index, "file_name": basename + ".png",
                        "raw_array": raw_name, "raw_array_sha256": release.digest(raw_bytes),
                        "source_frame_sha256": native["files"][source_name],
                        "selected_reference": settings["families"][case["family"]]["candidate_id"] == case["candidate_id"],
                        **{key: measured[key] for key in ("applied_opd_nm", "expected_photons", "cumulative_expected_photons")},
                        "evaluator_wfe_nm": measured["residual_wfe_nm"],
                        "evaluator_strehl_peak_ratio": measured["strehl_peak_ratio"]}
                    line = encoded(entry)
                    require(len(line) <= METADATA_ROW_BYTES, "Image metadata exceeds admission bound")
                    check_space(output, allocation(len(line), block) + block)
                    require(metadata.write(line) == len(line), "Incomplete image metadata write")
                    metadata_hash.update(line)
                    frame_count += 1
        require(frame_count == native["exposure_count"], "Export did not retain every exposure")
        hashes["images/metadata.jsonl"], hashes["episodes.jsonl"] = metadata_hash.hexdigest(), episodes_hash.hexdigest()
        provenance = {"schema": 1, "purpose": PURPOSE, "split": "public_tuning",
            "verification": verified, "generator_sha256": release.digest(Path(__file__).read_bytes()),
            "verifier_sha256": release.digest(Path(release.__file__).read_bytes()),
            "display": {"transform": "log1p(max(electrons,0))/log1p(photons_per_exposure)",
                        "minimum": 0, "maximum": 1, "colormap": "magma", "origin": "lower",
                        "raw_dtype": "float64", "raw_units": "electrons", "raw_shape": list(shape)},
            "rendering_runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                  "matplotlib": matplotlib.__version__, "pillow": pillow_version},
            "storage_admission": {"estimated_output_bytes": estimate, "reserve_bytes": RESERVE_BYTES}}
        candidates = len({row["candidate_id"] for row in rows})
        bases = len({row["base_wavefront_id"] for row in rows})
        for name, data in {"configuration.json": encoded(study["configuration"]),
                           "selected-settings.json": payload["selected-settings.json"],
                           "provenance.json": encoded(provenance),
                           "README.md": dataset_card(len(rows), frame_count, candidates, bases),
                           **{name: release.regular_bytes(Path(__file__).parents[1] / name)
                              for name in ("LICENSE", "NOTICE")}}.items():
            write_new(output / name, data, output, block, hashes)
        manifest = {"schema": 1, "purpose": PURPOSE, "split": "public_tuning",
            "input_manifest_sha256": release.NATIVE_MANIFEST_SHA256,
            "packet_manifest_sha256": checked["packet_manifest_sha256"],
            "episode_count": len(rows), "exposure_count": frame_count,
            "candidate_count": candidates, "independent_base_count": bases, "files": hashes}
        manifest_bytes = encoded(manifest)
        require(len(manifest_bytes) <= (2 * frame_count + 8) * 256, "Export manifest exceeds admission bound")
        write_new(output / ".manifest.pending", manifest_bytes, output, block, {})
        os.replace(output / ".manifest.pending", output / "manifest.json")
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export(args.packet, args.archive, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Selection export refused: {error}\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
