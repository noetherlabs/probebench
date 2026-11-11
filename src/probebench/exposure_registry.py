"""Bounded, source-bound batches of already exposed wavefronts.

An occurrence is a source location, not a unique base identity. These helpers
cannot establish that the caller has supplied every historical exposure.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .cohorts import orbit_distances, pupil_gram
from .comparison_contract import canonical_bytes, parse_json
from .optics import OpticalModel, OpticsConfig
from .scalar_gain_study import map_pupil_coefficients

SCHEMA = "probebench.exposure-batch/1"
MAX_ROWS = 256
MAX_SOURCES = 64
MAX_SOURCE_BYTES = 1_048_576
MAX_TOTAL_BYTES = 16_777_216


def _keys(value, expected):
    if type(value) is not dict or len(value) != len(expected) or set(value) != set(expected):
        raise ValueError("Unexpected exposure inventory fields")


def _label(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError("Invalid exposure identity")


def _optics(value):
    _keys(value, {"pupil_pixels", "oversampling", "wavelength_nm"})
    n, scale, wavelength = (value[k] for k in ("pupil_pixels", "oversampling", "wavelength_nm"))
    if (type(n) is not int or not 12 <= n <= 256 or type(scale) is not int
            or not 1 <= scale <= 1024 // n or type(wavelength) not in (int, float)
            or not 0 < wavelength <= 1_000_000_000):
        raise ValueError("Invalid bounded optical geometry")
    return value


def _vector(value):
    if (type(value) is not list or len(value) != 6
            or any(type(x) not in (int, float) or not -1_000_000 <= x <= 1_000_000 for x in value)):
        raise ValueError("Expected six finite OPD coefficients within the registry bound")
    return value


def _pointer(document, pointer):
    if type(pointer) is not str or len(pointer) > 512 or not pointer.startswith("/"):
        raise ValueError("Expected a bounded non-root JSON pointer")
    value = document
    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("Invalid JSON pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if type(value) is dict:
            value = value[token]
        elif type(value) is list and re.fullmatch(r"0|[1-9][0-9]{0,8}", token):
            value = value[int(token)]
        else:
            raise ValueError("JSON pointer does not identify a retained value")
    return value


def compile_exposure_batch(manifest_bytes, *, source_files):
    """Resolve full-precision public values; preserve missing or invalid sources.

Each bounded batch is checked independently. Unknown source content produces an
unresolved row, never a fabricated zero vector or an absent-overlap conclusion.
Source hashes bind supplied bytes; they do not authenticate their provenance.
"""
    manifest = parse_json(manifest_bytes)
    _keys(manifest, {"schema", "purpose", "sources", "occurrences"})
    if (manifest["schema"] != SCHEMA
            or manifest["purpose"] != "already_exposed_public_material"):
        raise ValueError("Only already exposed public material is accepted")
    declared, rows = manifest["sources"], manifest["occurrences"]
    if type(declared) is not dict or not 1 <= len(declared) <= MAX_SOURCES:
        raise ValueError("Exposure source count exceeds its bound")
    if type(rows) is not list or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Exposure occurrence count exceeds its bound")
    if type(source_files) is not dict or len(source_files) > len(declared):
        raise ValueError("Unexpected exposure source inventory")
    if not set(source_files) <= set(declared):
        raise ValueError("Unexpected exposure source names")
    # Validate structure before inspecting any supplied source contents.
    occurrences = set()
    for row in rows:
        _keys(row, {"occurrence_id", "base_id", "vector_source", "vector_pointer",
                    "optics_source", "optics_pointer"})
        for key in ("occurrence_id", "base_id"):
            _label(row[key])
        if row["occurrence_id"] in occurrences:
            raise ValueError("Duplicate exposure occurrence")
        occurrences.add(row["occurrence_id"])
        for role in ("vector", "optics"):
            if type(row[f"{role}_source"]) is not str or row[f"{role}_source"] not in declared:
                raise ValueError("Undeclared exposure source")
            pointer = row[f"{role}_pointer"]
            if type(pointer) is not str or len(pointer) > 512 or not pointer.startswith("/"):
                raise ValueError("Invalid exposure source pointer")
    documents, bindings, total = {}, {}, 0
    for name, expected in declared.items():
        _label(name)
        if type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("Invalid exposure source digest")
        payload = source_files.get(name)
        binding = {"expected_sha256": expected, "status": "missing"}
        if payload is not None:
            if type(payload) is not bytes or len(payload) > MAX_SOURCE_BYTES:
                raise ValueError("Exposure source exceeds its byte allowance")
            total += len(payload)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Exposure source inventory exceeds its byte allowance")
            actual = hashlib.sha256(payload).hexdigest()
            binding.update(sha256=actual, bytes=len(payload), status="digest_mismatch")
            if actual == expected:
                try:
                    documents[name] = parse_json(payload)
                except (ValueError, TypeError):
                    binding["status"] = "unparseable_or_out_of_bounds"
                else:
                    binding["status"] = "resolved"
        bindings[name] = binding
    resolved, unresolved = [], []
    for row in rows:
        try:
            vector = _vector(_pointer(documents[row["vector_source"]], row["vector_pointer"]))
            optics = _optics(_pointer(documents[row["optics_source"]], row["optics_pointer"]))
        except (KeyError, IndexError, ValueError, TypeError):
            unresolved.append({**row, "reason": "source_or_pointer_or_physical_value_unresolved"})
        else:
            resolved.append({**row, "truth_opd_nm": vector, "source_optics": optics})
    # Unused broken declarations still prevent a completeness claim.
    complete = not unresolved and all(b["status"] == "resolved" for b in bindings.values())
    return {"schema": SCHEMA, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "sources": bindings, "resolved": resolved, "unresolved": unresolved,
            "batch_resolved": complete, "historical_closure_established": False}


def check_exposure(base_id, truth_opd_nm, *, target_optics, manifest_bytes,
                   source_files, tolerance_nm=1e-6):
    """Check a single supplied candidate against one revalidated public batch.

Identity reuse and physical/conjugate overlap are separate rejection reasons.
No sampling, resampling, optical fitting or detector acquisition occurs here.
"""
    _label(base_id)
    vector, optics = _vector(truth_opd_nm), _optics(target_optics)
    if type(tolerance_nm) not in (int, float) or not 1e-6 <= tolerance_nm <= 1_000_000:
        raise ValueError("Overlap tolerance must be between 1e-6 and 1e6 nm")
    registry = compile_exposure_batch(manifest_bytes, source_files=source_files)
    result = {"schema": "probebench.exposure-check/1", "registry": registry,
              "base_id": base_id, "target_optics": optics, "truth_opd_nm": vector,
              "tolerance_nm": tolerance_nm, "identity_collisions": [],
              "physical_overlaps": [], "check_complete": False,
              "clear_of_supplied_batch": False, "historical_closure_established": False}
    # Do not turn a partial inventory into an apparently clean physical check.
    if not registry["batch_resolved"]:
        return result
    model = OpticalModel(OpticsConfig(**optics))
    gram = pupil_gram(model)
    converted = [row["truth_opd_nm"] if row["source_optics"] == optics else
                 map_pupil_coefficients(row["truth_opd_nm"], row["source_optics"], optics)
                 for row in registry["resolved"]]
    distances = orbit_distances(vector, converted, gram)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite physical exposure distance")
    for row, distance in zip(registry["resolved"], distances):
        if row["base_id"] == base_id:
            result["identity_collisions"].append(row["occurrence_id"])
        if distance <= tolerance_nm:
            result["physical_overlaps"].append({"occurrence_id": row["occurrence_id"],
                                                "distance_nm": float(distance)})
    result["check_complete"] = True
    result["clear_of_supplied_batch"] = not (result["identity_collisions"] or result["physical_overlaps"])
    # Return detached plain values, without exposing mutable source objects.
    return parse_json(canonical_bytes(result))
