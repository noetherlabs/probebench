"""Join bounded public exposure batches without claiming global history closure.

Reconstruction checks a retained public cohort fingerprint. Physical exclusion
checks the complete caller-declared history. Neither establishes that the caller
has disclosed every past experiment or admits prospective execution.
"""
from __future__ import annotations

import hashlib

from . import exposure_registry as registry
from .cohorts import MAX_BASES, generate_cohort
from .comparison_contract import _digest, canonical_bytes, parse_json
from .qualification import canonical_hash

SCHEMA = "probebench.exposure-history/1"
PURPOSE = "declared_already_exposed_public_material"
MAX_BATCHES = 64
MAX_HISTORY_BYTES = 64 * 1024**2
CHUNK_ROWS = registry.MAX_ROWS


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _tolerance(value):
    if type(value) not in (int, float) or not 1e-6 <= value <= 1_000_000:
        raise ValueError("History overlap tolerance must be between 1e-6 and 1e6 nm")
    return value


def reconstruct_public_cohort(configuration_bytes, *, history_id,
                              expected_configuration_sha256, expected_clusters_sha256):
    """Replay an already-public configuration into full-precision exposure batches.

    Expected digests must come from independently reviewed retained provenance;
    caller-supplied hashes alone do not prove that a cohort was previously public.
    Configuration fingerprints use the existing canonical cohort convention.
    This has numerical cost and requires external resource admission. No keys,
    files, detector frames, prospective cohorts or execution tokens are created.
    """
    registry._label(history_id)
    if len(history_id) > 64:
        raise ValueError("Historical source label exceeds its bound")
    for value in (expected_configuration_sha256, expected_clusters_sha256):
        _digest(value, "historical fingerprint")
    config = parse_json(configuration_bytes)
    registry._keys(config, {"specification", "seed_hex", "exclusions", "operating_reserve_bytes"})
    spec = config["specification"]
    if (type(spec) is not dict or type(spec.get("split")) is not str
            or spec["split"] not in {"development", "selection"}):
        raise ValueError("Only previously exposed development or selection cohorts can be reconstructed")
    if canonical_hash(config) != expected_configuration_sha256:
        raise ValueError("Historical configuration fingerprint differs")
    tolerance = _tolerance(spec.get("overlap_tolerance_nm"))
    cohort = generate_cohort(config)
    clusters = cohort["clusters"]
    if (type(clusters) is not list or not 1 <= len(clusters) <= MAX_BASES
            or cohort["base_count"] != len(clusters)
            or cohort["configuration_sha256"] != expected_configuration_sha256
            or canonical_hash(clusters) != expected_clusters_sha256):
        raise ValueError("Historical cluster fingerprint or count differs")
    # Retain every occurrence, including repeated IDs in other historical sources.
    # Checking each primary against both conjugate orbits also excludes its partner.
    entries, batches = [], {}
    for start in range(0, len(clusters), CHUNK_ROWS):
        batch_id = f"{history_id}:{start // CHUNK_ROWS:04d}"
        source = canonical_bytes({"clusters": clusters[start:start + CHUNK_ROWS]})
        rows = [{"occurrence_id": f"{history_id}:{start + index:06d}",
                 "base_id": cluster["base_wavefront_id"],
                 "vector_source": "clusters", "vector_pointer": f"/clusters/{index}/members/0/truth_opd_nm",
                 "optics_source": "configuration", "optics_pointer": "/specification/optics"}
                for index, cluster in enumerate(clusters[start:start + CHUNK_ROWS])]
        manifest = canonical_bytes({"schema": registry.SCHEMA,
            "purpose": "already_exposed_public_material",
            "sources": {"configuration": _sha(configuration_bytes), "clusters": _sha(source)},
            "occurrences": rows})
        batches[batch_id] = {"manifest_bytes": manifest,
                            "source_files": {"configuration": configuration_bytes, "clusters": source}}
        entries.append({"batch_id": batch_id, "manifest_sha256": _sha(manifest),
                        "occurrence_count": len(rows), "overlap_tolerance_nm": tolerance})
    inventory = canonical_bytes({"schema": SCHEMA, "purpose": PURPOSE, "batches": entries})
    checked = compile_exposure_history(inventory, batches=batches)
    if not checked["declared_history_resolved"]:
        raise ValueError("Reconstructed public values could not be resolved")
    return {"configuration_sha256": expected_configuration_sha256,
            "configuration_bytes_sha256": _sha(configuration_bytes),
            "clusters_sha256": expected_clusters_sha256, "base_occurrences": len(clusters),
            "inventory_bytes": inventory, "batches": batches,
            "historical_closure_established": False, "execution_admitted": False}


def _inputs(inventory_bytes, batches):
    inventory = parse_json(inventory_bytes)
    registry._keys(inventory, {"schema", "purpose", "batches"})
    if inventory["schema"] != SCHEMA or inventory["purpose"] != PURPOSE:
        raise ValueError("Only declared already-exposed histories are supported")
    entries = inventory["batches"]
    if type(entries) is not list or not 1 <= len(entries) <= MAX_BATCHES:
        raise ValueError("Historical batch inventory exceeds its bound")
    declared = {}
    for entry in entries:
        registry._keys(entry, {"batch_id", "manifest_sha256", "occurrence_count", "overlap_tolerance_nm"})
        name = entry["batch_id"]
        registry._label(name)
        _digest(entry["manifest_sha256"], "exposure batch")
        if name in declared:
            raise ValueError("Duplicate historical batch identity")
        count = entry["occurrence_count"]
        if type(count) is not int or not 1 <= count <= registry.MAX_ROWS:
            raise ValueError("Historical batch occurrence count exceeds its bound")
        _tolerance(entry["overlap_tolerance_nm"])
        declared[name] = entry
    if type(batches) is not dict or len(batches) > len(declared) or not set(batches) <= set(declared):
        raise ValueError("Unexpected historical batches")
    # Bound the whole input before hashing/parsing any source or constructing optics.
    total, snapshot = len(inventory_bytes), {}
    for name, batch in batches.items():
        registry._keys(batch, {"manifest_bytes", "source_files"})
        manifest, sources = batch["manifest_bytes"], batch["source_files"]
        if type(manifest) is not bytes or len(manifest) > registry.MAX_SOURCE_BYTES:
            raise ValueError("Historical batch manifest exceeds its byte allowance")
        if type(sources) is not dict or len(sources) > registry.MAX_SOURCES:
            raise ValueError("Historical batch sources exceed their count allowance")
        total += len(manifest)
        for source_name, payload in sources.items():
            registry._label(source_name)
            if type(payload) is not bytes or len(payload) > registry.MAX_SOURCE_BYTES:
                raise ValueError("Historical source exceeds its byte allowance")
            total += len(payload)
            if total > MAX_HISTORY_BYTES:
                raise ValueError("Historical sources exceed their total byte allowance")
        if total > MAX_HISTORY_BYTES:
            raise ValueError("Historical sources exceed their total byte allowance")
        snapshot[name] = {"manifest_bytes": manifest, "source_files": dict(sources)}
    return declared, snapshot


def _compile(inventory_bytes, batches):
    declared, snapshot = _inputs(inventory_bytes, batches)
    summaries, occurrence_ids, locations = {}, set(), set()
    for name, entry in declared.items():
        summary = {"expected_manifest_sha256": entry["manifest_sha256"],
                   "declared_occurrences": entry["occurrence_count"],
                   "overlap_tolerance_nm": entry["overlap_tolerance_nm"],
                   "resolved_occurrences": 0, "status": "missing"}
        if name in snapshot:
            batch = snapshot[name]
            actual = _sha(batch["manifest_bytes"])
            summary.update(manifest_sha256=actual, status="manifest_digest_mismatch")
            if actual == entry["manifest_sha256"]:
                compiled = registry.compile_exposure_batch(**batch)
                rows = compiled["resolved"] + compiled["unresolved"]
                if len(rows) != entry["occurrence_count"]:
                    raise ValueError("Historical occurrence count differs from its declared inventory")
                for row in rows:
                    occurrence = row["occurrence_id"]
                    location = (row["base_id"],
                        compiled["sources"][row["vector_source"]]["expected_sha256"], row["vector_pointer"],
                        compiled["sources"][row["optics_source"]]["expected_sha256"], row["optics_pointer"])
                    if occurrence in occurrence_ids or location in locations:
                        raise ValueError("Duplicate historical occurrence or same-identity source location")
                    occurrence_ids.add(occurrence)
                    locations.add(location)
                summary.update(status="resolved" if compiled["batch_resolved"] else "unresolved",
                    resolved_occurrences=len(compiled["resolved"]), sources=compiled["sources"],
                    unresolved_occurrence_ids=[row["occurrence_id"] for row in compiled["unresolved"]])
        summaries[name] = summary
    result = {"schema": SCHEMA, "inventory_sha256": _sha(inventory_bytes),
              "batches": summaries,
              "declared_occurrences": sum(row["occurrence_count"] for row in declared.values()),
              "resolved_occurrences": sum(row["resolved_occurrences"] for row in summaries.values()),
              "declared_history_resolved": all(row["status"] == "resolved" for row in summaries.values()),
              "historical_closure_established": False, "execution_admitted": False}
    return result, snapshot


def compile_exposure_history(inventory_bytes, *, batches):
    """Resolve every independently declared batch; missing history stays unresolved.

    Batches contain immutable manifest bytes and named JSON source bytes. No files
    are opened. Failed/partial runs and aliases must be included by the owner of
    the independent inventory. This function cannot discover omitted history.
    """
    return _compile(inventory_bytes, batches)[0]


def check_exposure_history(base_id, truth_opd_nm, *, target_optics, inventory_bytes,
                           batches, tolerance_nm=1e-6):
    """Check a supplied candidate only after the complete declared history resolves.

    Reuse the per-batch physical oracle at the maximum applicable tolerance.
    No sampling, replacement, policy invocation or future execution is admitted.
    """
    registry._label(base_id)
    vector = list(registry._vector(truth_opd_nm))
    optics = dict(registry._optics(target_optics))
    _tolerance(tolerance_nm)
    history, snapshot = _compile(inventory_bytes, batches)
    result = {"schema": "probebench.exposure-history-check/1", "history": history,
              "base_id": base_id, "truth_opd_nm": vector, "target_optics": optics,
              "requested_tolerance_nm": tolerance_nm, "checked_batches": [],
              "identity_collisions": [], "physical_overlaps": [],
              "check_complete": False, "clear_of_declared_history": False,
              "historical_closure_established": False, "execution_admitted": False}
    if not history["declared_history_resolved"]:
        return result
    for name, summary in history["batches"].items():
        applied = max(tolerance_nm, summary["overlap_tolerance_nm"])
        checked = registry.check_exposure(base_id, vector, target_optics=optics,
                                         tolerance_nm=applied, **snapshot[name])
        if not checked["check_complete"]:
            raise ValueError("Resolved historical batch did not complete its physical check")
        result["checked_batches"].append({"batch_id": name, "tolerance_nm": applied})
        result["identity_collisions"].extend({"batch_id": name, "occurrence_id": occurrence}
                                             for occurrence in checked["identity_collisions"])
        result["physical_overlaps"].extend({"batch_id": name, **row} for row in checked["physical_overlaps"])
    result["check_complete"] = True
    result["clear_of_declared_history"] = not (result["identity_collisions"] or result["physical_overlaps"])
    return result
