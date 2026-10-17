"""Verify the frozen v1 reference packet, optionally replaying its full archive.

Packet verification checks the published native-manifest fingerprint, original
metadata/source bytes and complete selection reduction. It does not verify camera
measurements unless --archive supplies the complete native archive. No environment
is installed and no source is imported before its independent fingerprint passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile


NATIVE_MANIFEST_SHA256 = "95130d4d9bf5d534aa3bfc88bdaef5934b5862769061df2bb26c03cf72806ef0"
SOURCE_INVENTORY_SHA256 = "9dc4c5901e54b73e129f93e94b977df7e02bf8e32cddecd47014d51c09e8aff0"
PACKET_PURPOSE = "classical_parameter_selection_reference_packet_v1"
RELEASE_PURPOSE = "classical_parameter_selection_release_v1"
NATIVE_PURPOSE = "classical_parameter_selection_v1"
ORIGINAL_JSON = ("study.json", "preparation.json", "summary.json", "selection.json",
                 "selected-settings.json")
MAX_FILE_BYTES = 64 * 1024**2
MAX_PACKET_BYTES = 128 * 1024**2
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode())


def read_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def finite(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=finite)


def safe_name(name):
    if (type(name) is not str or not name or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name or name == "."):
        raise ValueError("Unsafe relative artifact path")
    return name


def hash_inventory(value):
    if type(value) is not dict or not value:
        raise ValueError("A nonempty file-hash inventory is required")
    for name, sha256 in value.items():
        safe_name(name)
        if type(sha256) is not str or SHA256.fullmatch(sha256) is None:
            raise ValueError("Invalid SHA-256 inventory value")
    return value


def directory(path):
    path = Path(path).absolute()
    if (not path.is_dir()
            or any(parent.is_symlink() for parent in (path, *path.parents))):
        raise ValueError("Artifact directory must not use symbolic links")
    return path


def regular_bytes(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("Artifact must be a bounded regular file")
        data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("Artifact exceeds the reference-packet size bound")
        return data


def validate_packet(packet):
    """Return checked bytes, without importing or executing any packet source."""
    root = directory(packet)
    manifest_name = "packet-manifest.json"
    manifest_bytes = regular_bytes(root / manifest_name)
    manifest = read_json(manifest_bytes)
    if (type(manifest) is not dict or type(manifest.get("schema")) is not int
            or manifest["schema"] != 1 or manifest.get("purpose") != PACKET_PURPOSE):
        raise ValueError("Unexpected reference-packet manifest schema or purpose")
    files = hash_inventory(manifest.get("files"))
    if manifest_name in files:
        raise ValueError("Packet manifest must not include itself")
    actual = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            raise ValueError("Packet entries must be regular files or directories")
        if stat.S_ISREG(mode):
            actual.add(safe_name(path.relative_to(root).as_posix()))
    if actual != set(files) | {manifest_name}:
        raise ValueError("Packet inventory has missing or unlisted files")
    required = {*ORIGINAL_JSON, "archive-manifest.json", "release.json"}
    if not required <= set(files):
        raise ValueError("Packet is missing required selection evidence")
    payload, total = {}, len(manifest_bytes)
    for name, expected in sorted(files.items()):
        data = regular_bytes(root / name)
        total += len(data)
        if total > MAX_PACKET_BYTES:
            raise ValueError("Reference packet exceeds the total size bound")
        if digest(data) != expected:
            raise ValueError(f"Packet file hash mismatch: {name}")
        payload[name] = data
    if digest(payload["archive-manifest.json"]) != NATIVE_MANIFEST_SHA256:
        raise ValueError("Native manifest differs from the independent release fingerprint")
    native = read_json(payload["archive-manifest.json"])
    if (type(native) is not dict or type(native.get("schema")) is not int
            or native["schema"] != 1 or native.get("purpose") != NATIVE_PURPOSE
            or type(native.get("episode_count")) is not int or native["episode_count"] != 4032
            or type(native.get("exposure_count")) is not int or native["exposure_count"] != 30912):
        raise ValueError("Unexpected native study schema or counts")
    native_files = hash_inventory(native.get("files"))
    sources = hash_inventory(native.get("source_sha256"))
    if canonical_hash(sources) != SOURCE_INVENTORY_SHA256:
        raise ValueError("Source inventory differs from the independent source fingerprint")
    source_names = {"source/" + name for name in sources}
    if {name for name in files if name.startswith("source/")} != source_names:
        raise ValueError("Packet frozen-source inventory must match exactly")
    for name in (*ORIGINAL_JSON, *sorted(source_names)):
        if digest(payload[name]) != native_files.get(name):
            raise ValueError(f"Original archive file changed: {name}")
    if any(native_files.get("source/" + name) != sha256 for name, sha256 in sources.items()):
        raise ValueError("Native source hashes disagree")
    study = read_json(payload["study.json"])
    if type(study) is not dict or study.get("source_sha256") != sources:
        raise ValueError("Study/source binding mismatch")
    release = read_json(payload["release.json"])
    if (type(release) is not dict or type(release.get("schema")) is not int
            or release["schema"] != 1 or release.get("purpose") != RELEASE_PURPOSE
            or release.get("archive_manifest_sha256") != NATIVE_MANIFEST_SHA256
            or release.get("source_inventory_sha256") != SOURCE_INVENTORY_SHA256):
        raise ValueError("Release identity does not match the pinned native study")
    return {"bytes": payload, "native": native, "study": study,
            "packet_manifest_sha256": digest(manifest_bytes)}


def validate_archive_identity(archive):
    root = directory(archive)
    if digest(regular_bytes(root / "manifest.json")) != NATIVE_MANIFEST_SHA256:
        raise ValueError("Full archive manifest differs from the pinned native study")
    return root


def isolated_check(packet, archive=None):
    """Recheck bytes in the worker and import only their verified private copy."""
    checked = validate_packet(packet)
    payload, native, study = checked["bytes"], checked["native"], checked["study"]
    full_root = validate_archive_identity(archive) if archive is not None else None
    if any(name == "probebench" or name.startswith("probebench.") for name in sys.modules):
        raise ValueError("Selection verification requires a fresh isolated process")
    sys.dont_write_bytecode = True
    with tempfile.TemporaryDirectory(prefix="probebench-selection-verifier-") as temporary:
        source = Path(temporary).resolve()
        for name in native["source_sha256"]:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload["source/" + name])
        sys.path.insert(0, str(source / "src"))
        from probebench import selection

        for name, module in tuple(sys.modules.items()):
            if name == "probebench" or name.startswith("probebench."):
                if not Path(module.__file__).resolve().is_relative_to(source):
                    raise ValueError("Imported source did not come from the verified snapshot")
        preparation = read_json(payload["preparation.json"])
        cases = selection.cases_for(study["configuration"], preparation)
        if (len(cases) != native["episode_count"]
                or preparation["expected_episodes"] != len(cases)
                or sum(case["exposure_budget"] for case in cases) != native["exposure_count"]):
            raise ValueError("Generated selection inventory disagrees with native counts")
        expected_files = {".lock", *ORIGINAL_JSON, *("source/" + name for name in native["source_sha256"])}
        for case in cases:
            expected_files.add(f"cases/{case['id']}.json")
            expected_files.update(f"frames/{case['id']}/frame_{index:04d}.npz"
                                  for index in range(case["exposure_budget"]))
        if expected_files != set(native["files"]):
            raise ValueError("Native manifest disagrees with the complete expected inventory")
        rows = read_json(payload["summary.json"])
        if len(rows) != len(cases):
            raise ValueError("Incomplete selection summary")
        for row, case in zip(rows, cases, strict=True):
            if any(row[key] != case[key] for key in selection.CASE_FIELDS):
                raise ValueError("Summary case identities disagree with the frozen design")
        expected = [{key: case[key] for key in selection.CASE_FIELDS} for case in cases]
        outcomes = [{"case_id": row["id"], **{key: row[key] for key in
                    ("residual_wfe_nm", "outcome_status", "failure_reason")}} for row in rows]
        reduced = selection.select_candidates(expected, outcomes)
        if reduced != read_json(payload["selection.json"]):
            raise ValueError("Selection reduction does not reproduce exactly")
        candidates = {candidate["id"]: candidate for candidate in study["configuration"]["candidates"]}
        settings = {"schema": 1, "purpose": NATIVE_PURPOSE,
            "study_sha256": selection.canonical_hash(study),
            "selection_sha256": selection.canonical_hash(reduced),
            "families": {family: {"candidate_id": identity,
                "policy": selection.policy_configuration(study["configuration"], candidates[identity])}
                for family, identity in reduced["selected_candidates"].items()}}
        if (len(candidates) != 14 or len(settings["families"]) != 6
                or settings != read_json(payload["selected-settings.json"])):
            raise ValueError("Selected settings or complete candidate inventory disagree")
        if full_root is not None:
            verified = selection.verify_run(full_root)
            if verified != native:
                raise ValueError("Full replay returned a different archive manifest")
        return {"schema": 1, "packet_verified": True, "selection_reduction_verified": True,
            "full_camera_replay_verified": full_root is not None,
            "archive_manifest_sha256": NATIVE_MANIFEST_SHA256,
            "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "packet_manifest_sha256": checked["packet_manifest_sha256"],
            "episodes": len(cases), "exposures": native["exposure_count"],
            "candidates": len(candidates), "selected_families": len(settings["families"]),
            "scope": ("Complete native archive replay using saved fits; no optimizer refitting."
                      if full_root is not None else
                      "Compact packet consistency and selection reduction only; camera data not checked."),
            "limitations": "Hashes do not authenticate authorship. Auxiliary figures and prose are hash-checked, "
                           "not independently derived. Tuning losses are not held-out performance."}


def verify(packet, archive=None):
    """Run scientific verification with isolated imports and bytecode disabled."""
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
               str(Path(packet).absolute()), "--_worker"]
    if archive is not None:
        command += ["--archive", str(Path(archive).absolute())]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Isolated selection verification failed")
    return read_json(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--archive", type=Path, help="Complete native archive for camera/policy replay")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        result = isolated_check(args.packet, args.archive) if args._worker else verify(args.packet, args.archive)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Selection release verification failed: {error}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
