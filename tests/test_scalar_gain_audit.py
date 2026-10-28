"""Independent-auditor regressions using synthetic, non-study fixtures only."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "audit_scalar_gain", Path(__file__).resolve().parents[1] / "scripts/audit_scalar_gain.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("raw", [b'{"schema":1,"schema":2}', b'{"wfe":NaN}', b'{"wfe":Infinity}'])
def test_ambiguous_or_nonfinite_json_is_rejected(raw):
    with pytest.raises(ValueError):
        audit.parse(raw)


def test_incomplete_or_different_cohort_is_rejected_before_outcome_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "read_json", lambda *_: pytest.fail("Read outcomes before cohort admission"))
    with pytest.raises(ValueError, match="Complete manifest is absent"):
        audit.audit(tmp_path)
    (tmp_path / "manifest.json").write_text('{"episode_count":287}')
    with pytest.raises(ValueError, match="released complete cohort"):
        audit.audit(tmp_path)


def test_changed_configuration_is_rejected_before_random_draws(monkeypatch):
    monkeypatch.setattr(audit.np.random, "Generator", lambda *_: pytest.fail("Unadmitted random draw"))
    with pytest.raises(ValueError, match="Configuration differs"):
        audit.check_design({"cohort": "different"}, {}, [], None)


def test_receipt_never_overwrites_or_mutates_the_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(audit, "audit", lambda *_: pytest.fail("Read evidence before output validation"))
    with pytest.raises(ValueError, match="outside the retained run"):
        audit.main(["--run", str(run), "--out", str(run / "receipt.json")])
    existing = tmp_path / "receipt.json"
    existing.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        audit.main(["--run", str(run), "--out", str(existing)])
    assert existing.read_text() == "preserve"


@pytest.fixture
def inventory(tmp_path):
    case = {"id": "case_" + "0" * 64}
    study = {"source_sha256": {}, "configuration": {"protocol": {"reviewed_fixture_sources": {}}}}
    names = [".lock", "study.json", "preparation.json", "cases.json", "summary.json", "analysis.json",
             "cases/" + case["id"] + ".npz"]
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"fixture")
    manifest = {"schema": 1, "purpose": audit.PURPOSE, "source_sha256": {},
                "files": {name: hashlib.sha256(b"fixture").hexdigest() for name in names}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest, study, [case]


@pytest.mark.parametrize("mutation", ["content", "extra", "missing", "symlink", "source_digest"])
def test_inventory_detects_tampering(inventory, mutation):
    root, manifest, study, cases = inventory
    audit.check_inventory(root, manifest, study, cases)
    target = root / "summary.json"
    if mutation == "content":
        target.write_bytes(b"changed")  # Same size, different content.
    elif mutation == "extra":
        (root / "case.npz.pending").write_bytes(b"unfinished")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(root / "analysis.json")
    else:
        manifest["source_sha256"] = {"source.py": "0" * 64}
    with pytest.raises(ValueError):
        audit.check_inventory(root, manifest, study, cases)


def test_case_rejects_compression_and_unbounded_headers_before_loading(tmp_path, monkeypatch):
    target = tmp_path / "case.npz"
    np.savez_compressed(target, images=np.zeros((1, 64, 64)), metadata=np.array([123, 125], dtype=np.uint8))
    monkeypatch.setattr(audit.np, "load", lambda *_a, **_k: pytest.fail("Allocated unvalidated NPZ"))
    with pytest.raises(ValueError, match="Compressed"):
        audit.load_case(target)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False,
                                                "shape": (10**9, 64, 64)})
    metadata = io.BytesIO()
    np.save(metadata, np.array([123, 125], dtype=np.uint8))
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("images.npy", header.getvalue())
        archive.writestr("metadata.npy", metadata.getvalue())
    with pytest.raises(ValueError, match="bounded NPY"):
        audit.load_case(target)


@pytest.fixture
def physical_record():
    """Known 13-nm defocus fixture with synthetic images, not camera evidence."""
    study = {"fixture": "physical-metadata-unit-check"}
    case = {"id": "unit_fixture", "gain": 0.9, "family": audit.ARMS[0],
            "truth_opd_nm": [13., 0., 0., 0., 0., 0.]}
    images = np.zeros((7, 64, 64), dtype="<f8")
    calibration = {"response": "identity", "command_units": "nm_nominal_opd", "prediction_units": "nm_opd"}
    observations, trajectory = [], []
    for i in range(7):
        command = [(0., 80., -80.)[i % 3], 0., 0., 0., 0., 0.]
        applied = [0.9*v for v in command]
        observations.append({"schema": "probebench.nominal-observation/1", "index": i,
            "commanded_opd_nm": command, "nominal_applied_opd_nm": command,
            "nominal_calibration": calibration, "expected_photons": 2000,
            "cumulative_expected_photons": 2000*(i+1)})
        trajectory.append({"index": i, "applied_opd_nm": applied, "residual_wfe_nm": abs(13.-applied[0]),
            "expected_photons": 2000, "cumulative_expected_photons": 2000*(i+1)})
    final = [10., 0., 0., 0., 0., 0.]
    physical = {"hidden_opd_nm": case["truth_opd_nm"], "exposures": 7, "expected_photons_used": 14000,
        "trajectory": trajectory, "initial": {"residual_wfe_nm": 13.}, "final": {"residual_wfe_nm": 4.},
        "final_applied_opd_nm": [9., 0., 0., 0., 0., 0.],
        "final_mirror_surface_coefficients_nm": [4.5, 0., 0., 0., 0., 0.],
        "final_command_observation_indices": [], "final_command_has_camera_observation": False,
        "final_command_is_last_observation": False, "peak_probe_wfe_increase_nm": 72.}
    evaluation = {"schema": "probebench.scalar-gain-evaluation/1", "status": "submitted",
        "command_gain": 0.9, "nominal_command_limit_nm": 200., "nominal_observations": observations,
        "nominal_calibration": calibration, "physical": physical, "final_commanded_opd_nm": final}
    fit = {"observations_used": 7, "estimated_opd_nm": final, "fit_status": "finite", "optimizer_success": True,
           "visible_history_sha256": audit.public_history_hash(observations, images)}
    record = {"schema": 1, "purpose": audit.PURPOSE, "case": case,
        "case_binding": audit.digest_value({"study_sha256": audit.digest_value(study), "case": case}),
        "images_sha256": hashlib.sha256(images.tobytes()).hexdigest(), "evaluation": evaluation,
        "observations": observations, "fit_records": [fit], "elapsed_seconds": 1., "failed_observation": None}
    gram, _ = audit.independent_gram()
    return study, case, record, images, gram


@pytest.mark.parametrize("mutation", ["wfe", "physical_command", "nominal_command", "fit_prefix", "raw_pixel", "surface"])
def test_raw_and_physical_semantics_are_not_accepted_from_summary_alone(physical_record, mutation):
    study, case, record, images, gram = physical_record
    assert audit.check_record(study, case, record, images, gram) < 1e-10
    changed = deepcopy(record)
    if mutation == "wfe":
        changed["evaluation"]["physical"]["final"]["residual_wfe_nm"] = 3.
    elif mutation == "physical_command":
        changed["evaluation"]["physical"]["final_applied_opd_nm"][0] = 10.
    elif mutation == "nominal_command":
        changed["observations"][0]["nominal_applied_opd_nm"][0] = 1.
    elif mutation == "fit_prefix":
        changed["fit_records"][0]["visible_history_sha256"] = "0" * 64
    elif mutation == "raw_pixel":
        images[0, 0, 0] = 1.
    else:
        changed["evaluation"]["physical"]["final_mirror_surface_coefficients_nm"][0] = 9.
    with pytest.raises(ValueError):
        audit.check_record(study, case, changed, images, gram)


def synthetic_rows():
    rows = []
    for s, (stratum, _, _) in enumerate(audit.STRATA):
        for b in range(8):
            for m, member in enumerate(audit.MEMBERS):
                for g, gain in enumerate(audit.GAINS):
                    for a, arm in enumerate(audit.ARMS):
                        rows.append({"case_id": f"unit_{s}_{b}_{m}_{g}_{a}", "base_wavefront_id": f"unit:{stratum}:{b:06d}",
                            "stratum_id": stratum, "member": member, "gain": gain, "family": arm,
                            "status": "submitted", "residual_wfe_nm": 100. + 10*s + 5*b + 6*m + 20*g + a*(-2., .25, -3.)[g]})
    return rows


def fixture_config():
    # An explicitly synthetic bootstrap key; no physical cohort is generated.
    return {"bootstrap_seed_hex": "00"*32, "seed_commitments": {"bootstrap": "unit_fixture"}}


def test_bootstrap_preserves_conjugates_and_paired_gain_arm_curves():
    rows = synthetic_rows()
    result = audit.independent_analysis(fixture_config(), [], rows)
    assert result["independent_bases"] == 24
    assert result["base_curves"]["low"]["gain_arm_wfe_nm"][0][0] == [103., 101.]
    assert result["equal_stratum"][0]["fixed_wfe"]["mean_nm"] == 130.5
    for g, delta in enumerate((-2., .25, -3.)):
        contrast = result["equal_stratum"][g]["adaptive_minus_fixed"]
        assert contrast["mean_nm"] == pytest.approx(delta, abs=1e-12)
        assert contrast["interval_nm"] == pytest.approx([delta, delta], abs=1e-12)
        did = result["equal_stratum"][g]["difference_in_differences"]
        assert did["mean_nm"] == pytest.approx(delta-.25, abs=1e-12)
        assert did["interval_nm"] == pytest.approx([delta-.25, delta-.25], abs=1e-12)


def test_missing_conjugate_refuses_instead_of_changing_statistical_unit():
    rows = synthetic_rows()
    rows.pop()
    with pytest.raises(ValueError, match="Conjugate pair"):
        audit.independent_analysis(fixture_config(), [], rows)


def test_terminal_failure_withholds_aggregates_without_dropping_case():
    rows = synthetic_rows()
    rows[0].update(status="numerical_failure", residual_wfe_nm=None)
    result = audit.independent_analysis(fixture_config(), [], rows)
    assert result["status"] == "withheld_missing_finite_outcomes"
    assert result["failed_case_ids"] == [rows[0]["case_id"]]
    assert "equal_stratum" not in result and "strata" not in result


def test_changed_report_interval_is_rejected():
    reference = {"interval_nm": [-2.5, -1.5], "mean_nm": -2.}
    with pytest.raises(ValueError, match="Report numerical mismatch"):
        audit.compare_report({"interval_nm": [-2.5, -1.4], "mean_nm": -2.}, reference)
