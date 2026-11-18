"""Versioned full-precision prior records for future preparation archives.

This standard-library-only format never regenerates a wavefront. Trusted outer
study fingerprints must authenticate the declaration and completed manifest.
An intact archive proves custody of its declared records, not complete history,
physical correctness, statistical independence, or execution admission.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import PurePosixPath
import re

from .comparison_contract import _digest, _keys, canonical_bytes, parse_json

SCHEMA = "probebench.preparation-archive/2"
DECLARATION_SCHEMA = "probebench.prior-record-declaration/1"
PURPOSE = "full_precision_prior_retention"
ROLES = ("required_prior", "additional_prior")
CHUNK_RECORDS = 256
MAX_PRIORS = 32
MAX_RECORDS_PER_PRIOR = 4096
MAX_TOTAL_RECORDS = 65536
MAX_SOURCES = 32
MAX_FILES = 400
MAX_FILE_BYTES = 1024**2
MAX_ARCHIVE_BYTES = 64 * 1024**2
DECLARATION_FIELDS = {"role", "index", "source", "configuration_pointer",
                      "clusters_sha256_pointer", "configuration_sha256",
                      "clusters_sha256", "record_count"}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _identifier(value):
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
        raise ValueError("Invalid prior-record identifier")
    return value


def _payload(value):
    if type(value) is not bytes or not 1 <= len(value) <= MAX_FILE_BYTES:
        raise ValueError("Prior-record file exceeds its byte bound")
    return value


def _pointer(document, pointer):
    if type(pointer) is not str or not pointer.startswith("/") or len(pointer) > 1024:
        raise ValueError("Invalid prior-record source pointer")
    value = document
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            raise ValueError("Invalid JSON pointer escape")
        key = raw.replace("~1", "/").replace("~0", "~")
        if type(value) is dict and key in value:
            value = value[key]
        elif type(value) is list and re.fullmatch(r"0|[1-9][0-9]*", key):
            index = int(key)
            if index >= len(value):
                raise ValueError("Prior-record source pointer is absent")
            value = value[index]
        else:
            raise ValueError("Prior-record source pointer is absent")
    return value


def _declaration(payload, expected_sha256, sources):
    _digest(expected_sha256, "prior declaration")
    _payload(payload)
    if _sha(payload) != expected_sha256:
        raise ValueError("Prior declaration raw fingerprint differs")
    value = parse_json(payload)
    _keys(value, {"schema", "sources", "priors"}, "prior declaration")
    if value["schema"] != DECLARATION_SCHEMA:
        raise ValueError("Unsupported prior declaration schema")
    hashes, entries = value["sources"], value["priors"]
    if (type(hashes) is not dict or not 1 <= len(hashes) <= MAX_SOURCES
            or type(sources) is not dict or set(sources) != set(hashes)):
        raise ValueError("Prior source inventory differs")
    documents = {}
    for name, expected in hashes.items():
        _identifier(name)
        _digest(expected, "prior source")
        if _sha(_payload(sources[name])) != expected:
            raise ValueError("Prior source raw fingerprint differs")
        documents[name] = parse_json(sources[name])
    if type(entries) is not list or not 1 <= len(entries) <= MAX_PRIORS:
        raise ValueError("Prior declaration count exceeds its bound")
    next_index, total, previous_role, configurations = dict.fromkeys(ROLES, 0), 0, 0, []
    for entry in entries:
        _keys(entry, DECLARATION_FIELDS, "prior declaration entry")
        role, index, count = entry["role"], entry["index"], entry["record_count"]
        if (type(role) is not str or role not in ROLES or type(index) is not int
                or index != next_index[role] or ROLES.index(role) < previous_role):
            raise ValueError("Prior role/index order differs")
        previous_role = ROLES.index(role)
        next_index[role] += 1
        if type(count) is not int or not 1 <= count <= MAX_RECORDS_PER_PRIOR:
            raise ValueError("Prior record count exceeds its bound")
        total += count
        if total > MAX_TOTAL_RECORDS:
            raise ValueError("Total prior record count exceeds its bound")
        _identifier(entry["source"])
        if entry["source"] not in documents:
            raise ValueError("Prior source is absent")
        for name in ("configuration_sha256", "clusters_sha256"):
            _digest(entry[name], name)
        document = documents[entry["source"]]
        config = _pointer(document, entry["configuration_pointer"])
        fingerprint = _pointer(document, entry["clusters_sha256_pointer"])
        if (_sha(canonical_bytes(config)) != entry["configuration_sha256"]
                or fingerprint != entry["clusters_sha256"]):
            raise ValueError("Prior source/configuration association differs")
        _record_identities(config, count)
        configurations.append(config)
    return value, configurations


def _record_identities(configuration, count):
    _keys(configuration, {"specification", "seed_hex", "exclusions", "operating_reserve_bytes"},
          "prior configuration")
    spec = configuration["specification"]
    if (type(spec) is not dict or type(spec.get("split")) is not str
            or spec["split"] not in {"development", "selection"}):
        raise ValueError("Only already-exposed development or selection priors are supported")
    cohort = _identifier(spec.get("cohort_id"))
    strata = spec.get("strata")
    if type(strata) is not list or not 1 <= len(strata) <= count:
        raise ValueError("Prior stratum inventory exceeds its bound")
    counts = {}
    for row in strata:
        _keys(row, {"id", "rms_min_nm", "rms_max_nm", "base_count"}, "prior stratum")
        name, size = _identifier(row["id"]), row["base_count"]
        if name in counts or type(size) is not int or not 1 <= size <= count:
            raise ValueError("Prior stratum identity/count differs")
        counts[name] = size
    if sum(counts.values()) != count:
        raise ValueError("Prior configuration count differs")
    return [(f"{cohort}:{name}:{index:06d}", name, index)
            for name, size in sorted(counts.items()) for index in range(size)]


def _finite(value):
    return (type(value) is float and math.isfinite(value)
            or type(value) is int and value.bit_length() <= 63)


def _record(record, identity):
    _keys(record, {"base_wavefront_id", "stratum_id", "base_index", "sampled_rms_nm",
                   "physical_rms_nm", "members"}, "full prior record")
    if ((record["base_wavefront_id"], record["stratum_id"], record["base_index"]) != identity
            or type(record["base_index"]) is not int):
        raise ValueError("Prior record identity/order differs from configuration")
    for name in ("sampled_rms_nm", "physical_rms_nm"):
        value = record[name]
        if not _finite(value) or value < 0:
            raise ValueError("Prior RMS metadata must be finite and nonnegative")
    members = record["members"]
    if type(members) is not list or len(members) not in (1, 2):
        raise ValueError("Complete prior members are required")
    for member, expected in zip(members, ("primary", "conjugate")):
        _keys(member, {"member", "truth_opd_nm"}, "prior member")
        vector = member["truth_opd_nm"]
        if (member["member"] != expected or type(vector) is not list or len(vector) != 6
                or any(not _finite(v) for v in vector)):
            raise ValueError("Complete finite prior coefficient vectors are required")


def _prefix(entry):
    return f"priors/{entry['role']}/{entry['index']:04d}"


def _input_files(files):
    if type(files) is not dict or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("Preparation file count exceeds its bound")
    total = 0
    for name, payload in files.items():
        if (type(name) is not str or len(name) > 256 or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or str(PurePosixPath(name)) != name):
            raise ValueError("Invalid preparation file name")
        total += len(_payload(payload))
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError("Preparation archive exceeds its total byte bound")
    return dict(files)


def build_archive(preparation_bytes, *, declaration_bytes, expected_declaration_sha256,
                  sources, records, chunk_records=CHUNK_RECORDS):
    """Capture full prior records while a future preparation still owns them.

    ``records`` maps each ``(role, index)`` to its original ordered list. The
    declaration's source pointers bind independently frozen configurations and
    cluster fingerprints. Returned bytes are detached; no path is opened, no
    cohort is generated, and no existing v1 artifact is changed.
    """
    _payload(preparation_bytes)
    parse_json(preparation_bytes)
    declaration, configs = _declaration(declaration_bytes, expected_declaration_sha256, sources)
    entries = declaration["priors"]
    if type(chunk_records) is not int or not 1 <= chunk_records <= CHUNK_RECORDS:
        raise ValueError("Prior chunk size exceeds its bound")
    planned_files = 2 + len(sources) + len(entries) + sum(
        (e["record_count"] + chunk_records - 1) // chunk_records for e in entries)
    if planned_files > MAX_FILES:
        raise ValueError("Preparation file count exceeds its bound")
    expected_keys = {(e["role"], e["index"]) for e in entries}
    if type(records) is not dict or set(records) != expected_keys:
        raise ValueError("Complete declared prior record inventory is required")
    files = {"preparation.json": preparation_bytes, "prior-declaration.json": declaration_bytes,
             **{f"sources/{name}.json": payload for name, payload in sources.items()}}
    files = _input_files(files)
    total_bytes = sum(map(len, files.values()))
    def retain(path, payload):
        nonlocal total_bytes
        _payload(payload)
        if path in files:
            raise ValueError("Duplicate preparation file")
        total_bytes += len(payload)
        if total_bytes > MAX_ARCHIVE_BYTES:
            raise ValueError("Preparation archive exceeds its total byte bound")
        files[path] = payload
    retained = []
    for entry, config in zip(entries, configs, strict=True):
        rows = records[(entry["role"], entry["index"])]
        count = entry["record_count"]
        if type(rows) is not list or len(rows) != count:
            raise ValueError("Complete declared prior records are required")
        identities = _record_identities(config, count)
        prefix = _prefix(entry)
        config_path = prefix + "/configuration.json"
        retain(config_path, canonical_bytes(config))
        chunks, full_hash = [], hashlib.sha256(b"[")
        for start in range(0, count, chunk_records):
            chunk = rows[start:start + chunk_records]
            for offset, row in enumerate(chunk):
                _record(row, identities[start + offset])
                if start + offset:
                    full_hash.update(b",")
                full_hash.update(canonical_bytes(row))
            path = f"{prefix}/chunks/{start // chunk_records:04d}.json"
            payload = canonical_bytes(chunk)
            retain(path, payload)
            chunks.append({"path": path, "first_record": start, "record_count": len(chunk),
                           "sha256": _sha(payload)})
        full_hash.update(b"]")
        if full_hash.hexdigest() != entry["clusters_sha256"]:
            raise ValueError("Full prior cluster fingerprint differs")
        retained.append({**entry, "configuration_path": config_path, "chunks": chunks})
    files = _input_files(files)
    manifest = {"schema": SCHEMA, "purpose": PURPOSE,
                "declaration_sha256": expected_declaration_sha256,
                "preparation_sha256": _sha(preparation_bytes), "prior_count": len(entries),
                "chunk_records": chunk_records,
                "total_records": sum(e["record_count"] for e in entries), "priors": retained,
                "files": {name: {"sha256": _sha(payload), "bytes": len(payload)}
                          for name, payload in files.items()},
                "historical_global_closure_established": False, "execution_admitted": False}
    manifest_bytes = canonical_bytes(manifest)
    if total_bytes + len(manifest_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("Preparation archive and manifest exceed the total byte bound")
    result = {"manifest_bytes": manifest_bytes, "manifest_sha256": _sha(manifest_bytes), "files": files}
    read_archive(manifest_bytes, files=files, expected_manifest_sha256=result["manifest_sha256"],
                 expected_declaration_sha256=expected_declaration_sha256)
    return result


def read_archive(manifest_bytes, *, files, expected_manifest_sha256, expected_declaration_sha256):
    """Verify a v2 archive without replay; legacy summaries cannot satisfy v2."""
    _digest(expected_manifest_sha256, "preparation manifest")
    _payload(manifest_bytes)
    if _sha(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("Preparation manifest raw fingerprint differs")
    manifest = parse_json(manifest_bytes)
    _keys(manifest, {"schema", "purpose", "declaration_sha256", "preparation_sha256",
                     "prior_count", "total_records", "priors", "files", "chunk_records",
                     "historical_global_closure_established", "execution_admitted"}, "preparation manifest")
    if (manifest["schema"] != SCHEMA or manifest["purpose"] != PURPOSE
            or manifest["historical_global_closure_established"] is not False
            or manifest["execution_admitted"] is not False):
        raise ValueError("Only the versioned prior-retention format is supported")
    snapshot = _input_files(files)
    if sum(map(len, snapshot.values())) + len(manifest_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("Preparation archive and manifest exceed the total byte bound")
    chunk_records = manifest["chunk_records"]
    if type(chunk_records) is not int or not 1 <= chunk_records <= CHUNK_RECORDS:
        raise ValueError("Prior chunk size exceeds its bound")
    inventory = manifest["files"]
    if type(inventory) is not dict or set(inventory) != set(snapshot):
        raise ValueError("Preparation file inventory differs")
    for name, row in inventory.items():
        _keys(row, {"sha256", "bytes"}, "preparation file identity")
        if (type(row["bytes"]) is not int or row["bytes"] != len(snapshot[name])
                or row["sha256"] != _sha(snapshot[name])):
            raise ValueError("Preparation file raw identity differs")
    try:
        declaration_bytes = snapshot["prior-declaration.json"]
        declaration_value = parse_json(declaration_bytes)
        sources = {name: snapshot[f"sources/{name}.json"] for name in declaration_value["sources"]}
        declaration, configs = _declaration(declaration_bytes, expected_declaration_sha256, sources)
        preparation = snapshot["preparation.json"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Required preparation source is absent") from exc
    parse_json(preparation)
    entries = declaration["priors"]
    if (manifest["declaration_sha256"] != expected_declaration_sha256
            or manifest["preparation_sha256"] != _sha(preparation)
            or type(manifest["prior_count"]) is not int or manifest["prior_count"] != len(entries)
            or type(manifest["total_records"]) is not int
            or manifest["total_records"] != sum(e["record_count"] for e in entries)
            or type(manifest["priors"]) is not list or len(manifest["priors"]) != len(entries)):
        raise ValueError("Preparation declaration binding differs")
    required = {"preparation.json", "prior-declaration.json", *[f"sources/{n}.json" for n in sources]}
    retained = {}
    for entry, config, saved in zip(entries, configs, manifest["priors"], strict=True):
        _keys(saved, DECLARATION_FIELDS | {"configuration_path", "chunks"}, "retained prior")
        if canonical_bytes({name: saved[name] for name in DECLARATION_FIELDS}) != canonical_bytes(entry):
            raise ValueError("Retained prior source/role/index mapping differs")
        prefix, count = _prefix(entry), entry["record_count"]
        config_path = prefix + "/configuration.json"
        if saved["configuration_path"] != config_path or snapshot.get(config_path) != canonical_bytes(config):
            raise ValueError("Retained prior configuration differs")
        required.add(config_path)
        chunks = saved["chunks"]
        if type(chunks) is not list or len(chunks) != (count + chunk_records - 1) // chunk_records:
            raise ValueError("Complete prior chunk inventory is required")
        identities, full_hash, kept = _record_identities(config, count), hashlib.sha256(b"["), []
        for index, chunk in enumerate(chunks):
            _keys(chunk, {"path", "first_record", "record_count", "sha256"}, "prior chunk")
            path, start = f"{prefix}/chunks/{index:04d}.json", index * chunk_records
            expected_count = min(chunk_records, count - start)
            if (chunk["path"] != path or type(chunk["first_record"]) is not int
                    or chunk["first_record"] != start or type(chunk["record_count"]) is not int
                    or chunk["record_count"] != expected_count or path not in snapshot
                    or chunk["sha256"] != _sha(snapshot[path])):
                raise ValueError("Prior chunk identity/order differs")
            required.add(path)
            rows = parse_json(snapshot[path])
            if type(rows) is not list or len(rows) != expected_count:
                raise ValueError("Prior chunk record count differs")
            for offset, row in enumerate(rows):
                _record(row, identities[start + offset])
                if start + offset:
                    full_hash.update(b",")
                full_hash.update(canonical_bytes(row))
            kept.append(snapshot[path])
        full_hash.update(b"]")
        if full_hash.hexdigest() != entry["clusters_sha256"]:
            raise ValueError("Full retained prior cluster fingerprint differs")
        retained[(entry["role"], entry["index"])] = tuple(kept)
    if required != set(snapshot):
        raise ValueError("Unexpected preparation files")
    return {"schema": SCHEMA, "preparation_bytes": preparation, "prior_chunks": retained,
            "manifest_sha256": expected_manifest_sha256,
            "declaration_sha256": expected_declaration_sha256,
            "prior_count": len(entries), "total_records": manifest["total_records"],
            "declared_records_complete": True, "historical_global_closure_established": False,
            "execution_admitted": False}
