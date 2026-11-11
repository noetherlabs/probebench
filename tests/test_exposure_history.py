"""Historical reconstruction uses published configurations and exposed fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from probebench import exposure_history as history
from probebench import exposure_registry as registry
from probebench.run import FIXTURES

ROOT = Path(__file__).resolve().parents[1]
OPTICS = {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0}
PUBLIC = dict(FIXTURES)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def batch(name, *, vector=None, base_id="mixed_a", optics=None, tolerance=1e-6):
    source = encode({"optics": optics or OPTICS,
                     "wavefronts": [PUBLIC["mixed_a"] if vector is None else vector]})
    manifest = {"schema": registry.SCHEMA, "purpose": "already_exposed_public_material",
                "sources": {"public": digest(source)},
                "occurrences": [{"occurrence_id": name + ":0", "base_id": base_id,
                    "vector_source": "public", "vector_pointer": "/wavefronts/0",
                    "optics_source": "public", "optics_pointer": "/optics"}]}
    payload = encode(manifest)
    return ({"batch_id": name, "manifest_sha256": digest(payload), "occurrence_count": 1,
             "overlap_tolerance_nm": tolerance},
            {"manifest_bytes": payload, "source_files": {"public": source}})


def inventory(*specifications):
    entries, batches = [], {}
    for entry, content in specifications:
        entries.append(entry)
        batches[entry["batch_id"]] = content
    return encode({"schema": history.SCHEMA, "purpose": history.PURPOSE, "batches": entries}), batches


def check(payload, batches, base_id="candidate", vector=None, **kwargs):
    return history.check_exposure_history(base_id, PUBLIC["mixed_b"] if vector is None else vector,
        target_optics=OPTICS, inventory_bytes=payload, batches=batches, **kwargs)


def test_later_batch_physical_collision_defeats_clear_first_batch():
    first = batch("first")
    last = batch("last", vector=PUBLIC["mixed_b"], base_id="another_identity")
    payload, batches = inventory(first, last)
    assert registry.check_exposure("candidate", PUBLIC["mixed_b"], target_optics=OPTICS,
                                   **first[1])["clear_of_supplied_batch"]
    result = check(payload, batches)
    assert result["check_complete"] and not result["clear_of_declared_history"]
    assert result["physical_overlaps"] == [{"batch_id": "last", "occurrence_id": "last:0", "distance_nm": 0.0}]
    assert result["history"]["declared_occurrences"] == result["history"]["resolved_occurrences"] == 2
    assert not result["historical_closure_established"] and not result["execution_admitted"]


@pytest.mark.parametrize("failure", ["missing_batch", "wrong_manifest", "missing_source", "bad_pointer"])
def test_incomplete_declared_history_never_reaches_physical_check(monkeypatch, failure):
    payload, batches = inventory(batch("first"), batch("last", vector=PUBLIC["mixed_b"]))
    if failure == "missing_batch":
        del batches["last"]
    elif failure == "wrong_manifest":
        batches["last"]["manifest_bytes"] += b" "
    elif failure == "missing_source":
        batches["last"]["source_files"] = {}
    else:
        manifest = json.loads(batches["last"]["manifest_bytes"])
        manifest["occurrences"][0]["vector_pointer"] = "/wavefronts/99"
        batches["last"]["manifest_bytes"] = encode(manifest)
        declared = json.loads(payload)
        declared["batches"][1]["manifest_sha256"] = digest(batches["last"]["manifest_bytes"])
        payload = encode(declared)
    monkeypatch.setattr(registry, "check_exposure", lambda *a, **k: pytest.fail("Incomplete history reached optics"))
    result = check(payload, batches)
    assert not result["check_complete"] and not result["clear_of_declared_history"]
    assert not result["history"]["declared_history_resolved"]
    assert result["history"]["declared_occurrences"] == 2
    assert result["history"]["resolved_occurrences"] == 1
    assert result["history"]["batches"]["last"]["status"] != "resolved"


def test_reused_identity_and_relabelled_vector_keep_every_occurrence():
    payload, batches = inventory(batch("one"), batch("two", vector=PUBLIC["mixed_b"]),
                                batch("alias", base_id="another_identity"))
    result = check(payload, batches, base_id="mixed_a", vector=PUBLIC["positive_defocus"])
    assert result["identity_collisions"] == [{"batch_id": "one", "occurrence_id": "one:0"},
                                              {"batch_id": "two", "occurrence_id": "two:0"}]
    assert result["physical_overlaps"] == []
    assert result["history"]["resolved_occurrences"] == 3
    alias = check(payload, batches, vector=PUBLIC["mixed_a"])
    assert {row["batch_id"] for row in alias["physical_overlaps"]} == {"one", "alias"}
    assert not alias["clear_of_declared_history"]


@pytest.mark.parametrize("kind", ["occurrence", "source_location", "count", "batch", "unexpected"])
def test_inconsistent_history_inventory_is_refused(kind):
    first, last = batch("one"), batch("two", vector=PUBLIC["mixed_b"])
    if kind == "source_location":
        last = batch("two")  # same source/pointer/base ID, despite another occurrence label
    if kind == "occurrence":
        manifest = json.loads(last[1]["manifest_bytes"])
        manifest["occurrences"][0]["occurrence_id"] = "one:0"
        last[1]["manifest_bytes"] = encode(manifest)
        last[0]["manifest_sha256"] = digest(last[1]["manifest_bytes"])
    if kind == "count":
        last[0]["occurrence_count"] = 2
    if kind == "batch":
        last[0]["batch_id"] = "one"
    payload, batches = inventory(first, last)
    if kind == "unexpected":
        batches["undeclared"] = batches["one"]
    with pytest.raises(ValueError, match="Duplicate|count differs|Unexpected"):
        history.compile_exposure_history(payload, batches=batches)


def test_history_uses_maximum_applicable_tolerance_and_detached_inputs(monkeypatch):
    payload, batches = inventory(batch("one", tolerance=0.01),
                                batch("two", vector=PUBLIC["mixed_b"], tolerance=0.1))
    calls = []
    def physical(base_id, vector, **kwargs):
        calls.append(kwargs["tolerance_nm"])
        return {"check_complete": True, "identity_collisions": [], "physical_overlaps": []}
    monkeypatch.setattr(registry, "check_exposure", physical)
    result = check(payload, batches, tolerance_nm=0.05)
    assert calls == [0.05, 0.1]
    assert result["check_complete"] and result["clear_of_declared_history"]
    batches.clear()
    assert len(result["history"]["batches"]) == 2
    assert not result["execution_admitted"]


def test_aggregate_byte_bounds_precede_source_hashes_and_optics(monkeypatch):
    payload, batches = inventory(batch("one"), batch("two", vector=PUBLIC["mixed_b"]))
    total = len(payload) + sum(len(b["manifest_bytes"]) + sum(map(len, b["source_files"].values()))
                               for b in batches.values())
    monkeypatch.setattr(history, "MAX_HISTORY_BYTES", total - 1)
    monkeypatch.setattr(history, "_sha", lambda *_: pytest.fail("Oversized history was hashed"))
    monkeypatch.setattr(registry, "compile_exposure_batch", lambda *a, **k: pytest.fail("Oversized history was compiled"))
    with pytest.raises(ValueError, match="total byte allowance"):
        history.compile_exposure_history(payload, batches=batches)


@pytest.mark.parametrize("field,value", [("occurrence_count", True), ("occurrence_count", 257),
                                        ("overlap_tolerance_nm", 0), ("overlap_tolerance_nm", True)])
def test_invalid_history_count_and_tolerance(field, value):
    entry, content = batch("one")
    entry[field] = value
    payload, batches = inventory((entry, content))
    with pytest.raises(ValueError):
        history.compile_exposure_history(payload, batches=batches)


@pytest.fixture
def public_configuration():
    study = json.loads((ROOT / "results/scalar-command-response-v1-20261028/study.json").read_text())
    row = study["configuration"]["protocol"]["historical_selection_config"]["excluded_cohorts"][0]
    return row["configuration"], row["clusters_sha256"]


def reconstruct(config, cluster_hash, **kwargs):
    payload = encode(config)
    return history.reconstruct_public_cohort(payload, history_id="public_original",
        expected_configuration_sha256=digest(payload), expected_clusters_sha256=cluster_hash, **kwargs)


def test_public_cohort_reconstruction_matches_retained_fingerprint_and_chunks(public_configuration, monkeypatch):
    config, cluster_hash = public_configuration
    assert cluster_hash == "769bb018ea2f3eb7f7d8ea650f84e6764bb103d5467779bca7d13754051863f2"
    assert digest(encode(config)) == "839fdb57bbc251fd50e2388589869f08af86ae9c9c2bc1866a91b82aee28a2d7"
    monkeypatch.setattr(history, "CHUNK_ROWS", 3)
    result = reconstruct(config, cluster_hash)
    assert result["base_occurrences"] == 12 and len(result["batches"]) == 4
    resolved = history.compile_exposure_history(result["inventory_bytes"], batches=result["batches"])
    assert resolved["declared_history_resolved"] and resolved["resolved_occurrences"] == 12
    assert not resolved["historical_closure_established"] and not result["execution_admitted"]
    original_clusters = []
    for content in result["batches"].values():
        original_clusters.extend(json.loads(content["source_files"]["clusters"])["clusters"])
        compiled = registry.compile_exposure_batch(**content)
        assert all(row["source_optics"] == OPTICS for row in compiled["resolved"])
    assert digest(encode(original_clusters)) == cluster_hash
    # Bad retained fingerprint cannot produce a usable batch result.
    replayed = {"clusters": original_clusters, "base_count": 12,
                "configuration_sha256": digest(encode(config))}
    monkeypatch.setattr(history, "generate_cohort", lambda _: deepcopy(replayed))
    with pytest.raises(ValueError, match="cluster fingerprint"):
        reconstruct(config, "0" * 64)
    config["specification"]["optics"]["pupil_pixels"] = 12
    assert all(registry.compile_exposure_batch(**content)["resolved"][0]["source_optics"] == OPTICS
               for content in result["batches"].values())


@pytest.mark.parametrize("kind", ["reserved", "bad_split", "configuration_digest", "oversize"])
def test_bad_historical_admission_refuses_before_sampler(public_configuration, monkeypatch, kind):
    config, clusters = public_configuration
    if kind in {"reserved", "bad_split"}:
        config["specification"]["split"] = "reserved" if kind == "reserved" else []
    payload = encode(config)
    fingerprint = digest(payload)
    if kind == "configuration_digest":
        fingerprint = "0" * 64
    if kind == "oversize":
        payload = b" " * 1_048_577
    monkeypatch.setattr(history, "generate_cohort", lambda _: pytest.fail("Unadmitted history reached sampler"))
    with pytest.raises(ValueError):
        history.reconstruct_public_cohort(payload, history_id="public_original",
            expected_configuration_sha256=fingerprint, expected_clusters_sha256=clusters)
