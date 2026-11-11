"""Exposure accounting uses existing public fixtures, not a new incident cohort."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

import probebench.exposure_registry as registry
from probebench.optics import OpticalModel, OpticsConfig
from probebench.run import FIXTURES

OPTICS = {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0}
PUBLIC = dict(FIXTURES)


def packet(rows=None, optics=None):
    source = {"optics": optics or OPTICS, "wavefronts": rows or [PUBLIC["mixed_a"]]}
    payload = json.dumps(source).encode()
    manifest = {"schema": registry.SCHEMA, "purpose": "already_exposed_public_material",
                "sources": {"fixture": hashlib.sha256(payload).hexdigest()},
                "occurrences": [{"occurrence_id": f"fixture-{i}", "base_id": "mixed_a",
                    "vector_source": "fixture", "vector_pointer": f"/wavefronts/{i}",
                    "optics_source": "fixture", "optics_pointer": "/optics"}
                    for i in range(len(source["wavefronts"]))]}
    return manifest, {"fixture": payload}


def compile_packet(manifest, sources):
    return registry.compile_exposure_batch(json.dumps(manifest).encode(), source_files=sources)


def check(manifest, sources, base_id, vector, **kwargs):
    return registry.check_exposure(base_id, vector, manifest_bytes=json.dumps(manifest).encode(),
                                   source_files=sources, target_optics=OPTICS, **kwargs)


def test_reused_base_id_retains_each_full_precision_occurrence():
    manifest, sources = packet([PUBLIC["mixed_a"], PUBLIC["mixed_b"]])
    result = compile_packet(manifest, sources)
    assert result["batch_resolved"] and not result["historical_closure_established"]
    assert [row["truth_opd_nm"] for row in result["resolved"]] == [PUBLIC["mixed_a"], PUBLIC["mixed_b"]]
    same_identity = check(manifest, sources, "mixed_a", PUBLIC["positive_defocus"])
    assert same_identity["identity_collisions"] == ["fixture-0", "fixture-1"]
    assert same_identity["physical_overlaps"] == []
    assert not same_identity["clear_of_supplied_batch"]


def test_relabelled_conjugate_and_exact_vector_are_physical_overlap():
    manifest, sources = packet()
    for vector in (PUBLIC["mixed_a"], [-30, -12, 10, 8, -6, -15]):
        result = check(manifest, sources, "a_different_identity", vector)
        assert result["identity_collisions"] == []
        assert result["physical_overlaps"] == [{"occurrence_id": "fixture-0", "distance_nm": 0.0}]
        assert not result["clear_of_supplied_batch"]


def test_independent_pupil_oracle_checks_cross_grid_mapping():
    small = {**OPTICS, "pupil_pixels": 12}
    target_model = OpticalModel(OpticsConfig(**OPTICS))
    def raw_scales(n):
        x = np.linspace(-1 + 1/n, 1 - 1/n, n)
        y, x = np.meshgrid(x, x, indexing="ij")
        r2 = x*x + y*y
        mask = r2 <= 1
        values = [np.sqrt(3)*(2*r2-1), np.sqrt(6)*(x*x-y*y), np.sqrt(6)*2*x*y,
                  np.sqrt(8)*(3*r2-2)*x, np.sqrt(8)*(3*r2-2)*y,
                  np.sqrt(5)*(6*r2*r2-6*r2+1)]
        return np.array([np.std(value[mask]) for value in values])
    converted = np.asarray(PUBLIC["mixed_a"])*raw_scales(32)/raw_scales(12)
    # Direct finite-pupil RMS is the oracle, not coefficient Euclidean distance.
    candidate = np.asarray(PUBLIC["mixed_b"])
    direct = target_model.wavefront_nm(candidate - converted)[target_model.pupil]
    opposite = target_model.wavefront_nm(candidate - converted*np.array([-1,-1,-1,1,1,-1]))[target_model.pupil]
    distance = min(np.std(direct), np.std(opposite))
    manifest, sources = packet(optics=small)
    overlap = check(manifest, sources, "renamed", converted.tolist())
    assert overlap["physical_overlaps"][0]["distance_nm"] < 1e-12
    apart = check(manifest, sources, "other", candidate.tolist(), tolerance_nm=float(distance)+1e-8)
    assert apart["physical_overlaps"][0]["distance_nm"] == pytest.approx(distance)


@pytest.mark.parametrize("kind", ["missing", "wrong_hash", "not_json", "bad_pointer", "boolean", "nonfinite", "wrong_optics"])
def test_unknown_history_is_unresolved_and_never_clear(monkeypatch, kind):
    manifest, sources = packet()
    if kind == "missing":
        sources = {}
    elif kind == "wrong_hash":
        sources["fixture"] += b" "
    elif kind == "not_json":
        sources["fixture"] = b"unparseable retained source"
        manifest["sources"]["fixture"] = hashlib.sha256(sources["fixture"]).hexdigest()
    elif kind == "bad_pointer":
        manifest["occurrences"][0]["vector_pointer"] = "/wavefronts/9"
    else:
        source = json.loads(sources["fixture"])
        if kind == "boolean":
            source["wavefronts"][0][0] = True
        elif kind == "nonfinite":
            source["wavefronts"][0][0] = float("nan")
        else:
            source["optics"]["pupil_pixels"] = 10**9
        sources["fixture"] = json.dumps(source).encode()
        manifest["sources"]["fixture"] = hashlib.sha256(sources["fixture"]).hexdigest()
    monkeypatch.setattr(registry, "OpticalModel", lambda *_: pytest.fail("Unknown history reached optics"))
    result = check(manifest, sources, "new_label", PUBLIC["mixed_b"])
    assert result["registry"]["unresolved"]
    assert not result["check_complete"] and not result["clear_of_supplied_batch"]


def test_duplicate_occurrence_and_reserved_material_refused():
    manifest, sources = packet()
    duplicate = deepcopy(manifest)
    duplicate["occurrences"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        compile_packet(duplicate, sources)
    manifest["purpose"] = "reserved"
    with pytest.raises(ValueError, match="already exposed"):
        compile_packet(manifest, sources)


def test_no_rounding_or_approximate_replacement():
    manifest, sources = packet()
    source = json.loads(sources["fixture"])
    # Synthetic bookkeeping perturbation of an already-public point; no draw.
    source["wavefronts"][0][0] += 0.0000007
    sources["fixture"] = json.dumps(source).encode()
    manifest["sources"]["fixture"] = hashlib.sha256(sources["fixture"]).hexdigest()
    result = compile_packet(manifest, sources)
    assert result["resolved"][0]["truth_opd_nm"][0] == 30.0000007
    assert check(manifest, sources, "renamed", PUBLIC["mixed_a"])["physical_overlaps"]


def test_bounded_source_inventory_and_duplicate_json_keys():
    manifest, sources = packet()
    with pytest.raises(ValueError, match="inventory"):
        compile_packet(manifest, {**sources, "unexpected": b"{}"})
    with pytest.raises(ValueError, match="byte allowance"):
        compile_packet(manifest, {"fixture": b"x"*(registry.MAX_SOURCE_BYTES+1)})
    with pytest.raises(ValueError, match="Duplicate JSON"):
        registry.compile_exposure_batch(b'{"schema": 1, "schema": 2}', source_files={})
