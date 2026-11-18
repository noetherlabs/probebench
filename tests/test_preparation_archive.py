"""Archive custody checks use existing public records; no sampler is invoked."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from probebench import preparation_archive as archive

ROOT = Path(__file__).resolve().parents[1]
SELECTION = "classical-selection-v1-20261016"
SCALAR = "scalar-command-response-v1-20261028"
CLUSTER_HASHES = (
    "a4bbd6ba53379d308b88a2fda890fa8617725a0c71e06c6564a513c5c6929647",
    "1f99ba437da7b774ab18c2c184587235564e7d9be03796c4738984ccb4d321df",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def public(folder, filename):
    return (ROOT / "results" / folder / filename).read_bytes()


@pytest.fixture
def inputs():
    selection = json.loads(public(SELECTION, "study.json"))["configuration"]["cohort"]
    scalar = json.loads(public(SCALAR, "study.json"))["configuration"]["protocol"]["cohort"]
    configs = [selection, scalar]
    rows = [json.loads(public(name, "preparation.json"))["cohort"]["clusters"]
            for name in (SELECTION, SCALAR)]
    assert [sha(encoded(value)) for value in rows] == list(CLUSTER_HASHES)
    source = encoded({"priors": [{"configuration": config, "clusters_sha256": expected}
                                 for config, expected in zip(configs, CLUSTER_HASHES, strict=True)]})
    entries = [{"role": role, "index": 0, "source": "protocol",
                "configuration_pointer": f"/priors/{index}/configuration",
                "clusters_sha256_pointer": f"/priors/{index}/clusters_sha256",
                "configuration_sha256": sha(encoded(configs[index])),
                "clusters_sha256": CLUSTER_HASHES[index], "record_count": len(rows[index])}
               for index, role in enumerate(archive.ROLES)]
    declaration = encoded({"schema": archive.DECLARATION_SCHEMA,
                           "sources": {"protocol": sha(source)}, "priors": entries})
    return {"preparation_bytes": public(SCALAR, "preparation.json"),
            "declaration_bytes": declaration, "expected_declaration_sha256": sha(declaration),
            "sources": {"protocol": source},
            "records": {(role, 0): row for role, row in zip(archive.ROLES, rows, strict=True)}}


def read(packet, inputs):
    return archive.read_archive(packet["manifest_bytes"], files=packet["files"],
                                expected_manifest_sha256=packet["manifest_sha256"],
                                expected_declaration_sha256=inputs["expected_declaration_sha256"])


def rewrite_manifest(packet, value):
    packet["manifest_bytes"] = encoded(value)
    packet["manifest_sha256"] = sha(packet["manifest_bytes"])


def rebind_file(packet, manifest, path, payload):
    packet["files"][path] = payload
    manifest["files"][path] = {"bytes": len(payload), "sha256": sha(payload)}
    for prior in manifest["priors"]:
        for chunk in prior["chunks"]:
            if chunk["path"] == path:
                chunk["sha256"] = sha(payload)
    rewrite_manifest(packet, manifest)


@pytest.mark.parametrize("chunk_records", [1, 3, 7, 256])
def test_existing_full_precision_priors_roundtrip_across_chunk_boundaries(inputs, chunk_records):
    packet = archive.build_archive(**inputs, chunk_records=chunk_records)
    result = read(packet, inputs)
    assert result["preparation_bytes"] == inputs["preparation_bytes"]
    assert result["total_records"] == 36 and result["prior_count"] == 2
    for index, key in enumerate(inputs["records"]):
        restored = [row for payload in result["prior_chunks"][key] for row in json.loads(payload)]
        assert sha(encoded(restored)) == CLUSTER_HASHES[index]
        for expected, actual in zip(inputs["records"][key], restored, strict=True):
            for left, right in zip(expected["members"], actual["members"], strict=True):
                assert [float(v).hex() for v in left["truth_opd_nm"]] == [
                    float(v).hex() for v in right["truth_opd_nm"]]
    assert result["declared_records_complete"] is True
    assert result["historical_global_closure_established"] is False
    assert result["execution_admitted"] is False


def test_old_hash_only_preparation_is_not_a_v2_record_archive(inputs):
    legacy = public(SCALAR, "preparation.json")
    old = json.loads(legacy)
    assert old["prior_cohorts"][1]["base_count"] == 4096
    assert "clusters" not in old["prior_cohorts"][1]
    with pytest.raises(ValueError, match="preparation manifest"):
        archive.read_archive(legacy, files={}, expected_manifest_sha256=sha(legacy),
                             expected_declaration_sha256=inputs["expected_declaration_sha256"])


def test_writer_refuses_missing_prior_before_creating_an_archive(inputs):
    inputs["records"].pop(("additional_prior", 0))
    with pytest.raises(ValueError, match="Complete declared prior record inventory"):
        archive.build_archive(**inputs)


def test_reader_refuses_missing_chunk_even_when_outer_manifest_is_rehashed(inputs):
    packet = archive.build_archive(**inputs, chunk_records=3)
    manifest = json.loads(packet["manifest_bytes"])
    path = manifest["priors"][0]["chunks"][1]["path"]
    packet["files"].pop(path)
    manifest["files"].pop(path)
    rewrite_manifest(packet, manifest)
    with pytest.raises(ValueError, match="Prior chunk identity/order"):
        read(packet, inputs)


def test_raw_file_and_manifest_fingerprints_are_mandatory(inputs):
    packet = archive.build_archive(**inputs)
    altered = deepcopy(packet)
    altered["manifest_bytes"] += b" "
    with pytest.raises(ValueError, match="manifest raw fingerprint"):
        read(altered, inputs)
    packet["files"]["preparation.json"] += b" "
    with pytest.raises(ValueError, match="file raw identity"):
        read(packet, inputs)


def test_one_ulp_corruption_fails_complete_cluster_hash_after_chunk_rehash(inputs):
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    path = manifest["priors"][0]["chunks"][0]["path"]
    rows = json.loads(packet["files"][path])
    vector = rows[0]["members"][0]["truth_opd_nm"]
    vector[0] = math.nextafter(vector[0], math.inf)
    rebind_file(packet, manifest, path, encoded(rows))
    with pytest.raises(ValueError, match="Full retained prior cluster fingerprint"):
        read(packet, inputs)


def test_authentic_prior_payloads_cannot_be_cross_paired_between_roles(inputs):
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    manifest["priors"].reverse()
    rewrite_manifest(packet, manifest)
    with pytest.raises(ValueError, match="source/role/index mapping"):
        read(packet, inputs)


@pytest.mark.parametrize("field,value", [("index", False), ("record_count", 12.0)])
def test_prior_mapping_requires_exact_json_types_after_outer_rehash(inputs, field, value):
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    manifest["priors"][0][field] = value
    rewrite_manifest(packet, manifest)
    with pytest.raises(ValueError, match="source/role/index mapping"):
        read(packet, inputs)


def test_authentic_configurations_cannot_be_cross_paired_between_priors(inputs):
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    left, right = [row["configuration_path"] for row in manifest["priors"]]
    a, b = packet["files"][left], packet["files"][right]
    rebind_file(packet, manifest, left, b)
    rebind_file(packet, manifest, right, a)
    with pytest.raises(ValueError, match="prior configuration differs"):
        read(packet, inputs)


@pytest.mark.parametrize("field,value", [("role", "additional_prior"), ("index", 1),
                                        ("configuration_pointer", "/priors/1/configuration"),
                                        ("clusters_sha256_pointer", "/priors/1/clusters_sha256")])
def test_caller_cannot_relabel_or_redirect_a_frozen_declaration(inputs, field, value):
    declaration = json.loads(inputs["declaration_bytes"])
    declaration["priors"][0][field] = value
    inputs["declaration_bytes"] = encoded(declaration)
    with pytest.raises(ValueError, match="declaration raw fingerprint"):
        archive.build_archive(**inputs)


def test_redeclared_source_pointer_must_still_match_its_configuration_and_cluster_hashes(inputs):
    declaration = json.loads(inputs["declaration_bytes"])
    declaration["priors"][0]["configuration_pointer"] = "/priors/1/configuration"
    inputs["declaration_bytes"] = encoded(declaration)
    inputs["expected_declaration_sha256"] = sha(inputs["declaration_bytes"])
    with pytest.raises(ValueError, match="source/configuration association"):
        archive.build_archive(**inputs)


@pytest.mark.parametrize("field,value", [("first_record", 1), ("record_count", 2), ("first_record", False)])
def test_chunk_position_or_count_cannot_change_after_outer_rehash(inputs, field, value):
    packet = archive.build_archive(**inputs, chunk_records=3)
    manifest = json.loads(packet["manifest_bytes"])
    manifest["priors"][0]["chunks"][0][field] = value
    rewrite_manifest(packet, manifest)
    with pytest.raises(ValueError, match="Prior chunk identity/order"):
        read(packet, inputs)


@pytest.mark.parametrize("mutate", ["base_id", "base_index", "member", "truncated_vector", "missing_member"])
def test_complete_records_cannot_silently_degrade_to_vectors_or_other_identities(inputs, mutate):
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    path = manifest["priors"][0]["chunks"][0]["path"]
    rows = json.loads(packet["files"][path])
    row = rows[0]
    if mutate == "base_id":
        row["base_wavefront_id"] = "another:high:000000"
    elif mutate == "base_index":
        row["base_index"] = False
    elif mutate == "member":
        row["members"][0]["member"] = "conjugate"
    elif mutate == "truncated_vector":
        row["members"][0]["truth_opd_nm"].pop()
    else:
        row["members"].pop()
    rebind_file(packet, manifest, path, encoded(rows))
    with pytest.raises(ValueError, match="identity/order|coefficient vectors|cluster fingerprint"):
        read(packet, inputs)


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, 10**1000])
def test_writer_rejects_nonfinite_boolean_or_unbounded_coefficients(inputs, bad):
    inputs["records"][("required_prior", 0)][0]["members"][0]["truth_opd_nm"][0] = bad
    with pytest.raises(ValueError, match="finite prior coefficient"):
        archive.build_archive(**inputs)


@pytest.mark.parametrize("chunk_records", [0, 257, True])
def test_chunk_allowance_is_enforced_before_packing(inputs, chunk_records):
    with pytest.raises(ValueError, match="chunk size"):
        archive.build_archive(**inputs, chunk_records=chunk_records)


def test_total_byte_allowance_is_enforced(inputs, monkeypatch):
    monkeypatch.setattr(archive, "MAX_ARCHIVE_BYTES", 16)
    with pytest.raises(ValueError, match="total byte bound"):
        archive.build_archive(**inputs)


def test_unlisted_extra_files_and_raw_source_changes_are_refused(inputs):
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    rebind_file(packet, manifest, "unlisted.json", b"{}")
    with pytest.raises(ValueError, match="Unexpected preparation files"):
        read(packet, inputs)
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    path = "sources/protocol.json"
    rebind_file(packet, manifest, path, packet["files"][path] + b" ")
    with pytest.raises(ValueError, match="source raw fingerprint"):
        read(packet, inputs)


def test_returned_bytes_are_detached_from_mutable_input_objects(inputs):
    packet = archive.build_archive(**inputs)
    expected = deepcopy(packet)
    inputs["records"][("required_prior", 0)][0]["members"][0]["truth_opd_nm"][0] = 0
    inputs["sources"]["protocol"] = b"{}"
    assert packet == expected
    assert read(packet, inputs)["declared_records_complete"]


@pytest.mark.parametrize("flag", ["historical_global_closure_established", "execution_admitted"])
def test_archive_cannot_promote_custody_to_history_closure_or_admission(inputs, flag):
    packet = archive.build_archive(**inputs)
    manifest = json.loads(packet["manifest_bytes"])
    manifest[flag] = True
    rewrite_manifest(packet, manifest)
    with pytest.raises(ValueError, match="versioned prior-retention"):
        read(packet, inputs)
