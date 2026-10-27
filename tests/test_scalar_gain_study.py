import hashlib
import json
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probebench import scalar_gain_study as gain
from probebench import study as storage
from probebench.budget_policies import PolicyConfig
from probebench.cohorts import CONJUGATE_SIGNS, stream_seed
from probebench.optics import OpticalModel, OpticsConfig
from probebench.qualification import canonical_hash
from probebench.run import digest, source_inventory

ROOT = Path(__file__).resolve().parents[1]
TRUTH = [30., 12., -10., 8., -6., 15.]


@pytest.fixture
def protocol():
    historical = json.loads((ROOT / "configs/classical_selection_v1.json").read_text())
    cohort = deepcopy(historical["cohort"])
    cohort["specification"].update(cohort_id=gain.COHORT_ID, split="development")
    for stratum in cohort["specification"]["strata"]:
        stratum["base_count"] = 8
    cohort["seed_hex"] = "a1" * 32
    for value in (13., 31.):
        cohort["exclusions"]["wavefronts"].append({"id": f"export_{int(value)}",
            "truth_opd_nm": [value, 0., 0., 0., 0., 0.], "sources": ["tests/test_selection_export.py"]})
    result = {"schema": 1, "purpose": gain.PURPOSE, "cohort": cohort,
        "gains": list(gain.GAINS), "candidates": [
            {"id": name, "family": family, "policy": asdict(PolicyConfig())}
            for name, family in zip(("fixed_probe_080", "adaptive_probe_080"), gain.ARMS, strict=True)],
        "exposure_budget": 7, "photons_per_exposure": 2000, "read_noise_electrons": 1.,
        "historical_selection_config": historical, "selected_settings_sha256": gain.SELECTED_SETTINGS,
        "additional_prior_cohorts": [], "cross_grid_fixtures": [],
        "reviewed_fixture_sources": {"tests/test_nominal_calibration.py": digest(ROOT / "tests/test_nominal_calibration.py")},
        "detector_seed_hex": "b2" * 32, "policy_seed_hex": "c3" * 32,
        "bootstrap_seed_hex": "d4" * 32, "operating_reserve_bytes": gain.RESERVE}
    result["seed_commitments"] = gain.seed_commitments(result)
    return result


def admission(config):
    return {"schema": 1, "purpose": gain.PURPOSE, "status": "FROZEN",
        "configuration_sha256": canonical_hash(config), "source_revision": "0" * 40,
        "source_sha256": source_inventory()[1], "runtime": gain.runtime(),
        "seed_commitments": gain.seed_commitments(config),
        "exclusion_inventory_sha256": canonical_hash(gain.exclusion_inventory(config))}


@pytest.fixture
def tiny(protocol, monkeypatch):
    """Existing known fixtures only; never call the new-cohort sampler in tests."""
    config = deepcopy(protocol)
    config["cohort"]["specification"]["optics"] = {"pupil_pixels": 12, "oversampling": 1, "wavelength_nm": 550.}
    config.update(exposure_budget=3, photons_per_exposure=100)
    for candidate in config["candidates"]:
        candidate["policy"]["max_mle_iterations"] = 2
    clusters = []
    for index, vector in enumerate((TRUTH, [-25., -8., 14., -10., 9., -12.])):
        clusters.append({"base_wavefront_id": f"known_fixture:low:{index:06d}", "stratum_id": "low",
            "base_index": index, "members": [{"member": "primary", "truth_opd_nm": vector},
            {"member": "conjugate", "truth_opd_nm": (np.asarray(vector) * CONJUGATE_SIGNS).tolist()}]})
    prepared = {"cohort": {"clusters": clusters}, "exclusion_inventory_sha256": "fixture_only"}
    monkeypatch.setattr(gain, "validate_configuration", lambda config: None)
    monkeypatch.setattr(gain, "validate_admission", lambda config, receipt:
        gain._admission_bindings(config, receipt, source_inventory()[1], gain.runtime()))
    monkeypatch.setattr(gain, "_prepare", lambda config: deepcopy(prepared))
    monkeypatch.setattr(gain, "storage_estimate", lambda config: {"retained_bytes_for_admission": 1024**2})
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * gain.RESERVE))
    return config, prepared


def bound(config):
    return {"configuration": {"protocol": config, "admission": admission(config),
                              "operating_reserve_bytes": config["operating_reserve_bytes"]},
            "source_sha256": source_inventory()[1], "runtime": gain.runtime()}


def test_inspection_and_storage_estimate_never_sample_new_keys(protocol, monkeypatch):
    monkeypatch.setattr(gain, "generate_cohort", lambda *args: pytest.fail("Inspection sampled a cohort"))
    gain.validate_configuration(protocol)
    estimate = gain.storage_estimate(protocol)
    assert estimate["episodes"] == 288 and estimate["maximum_exposures"] == 2016
    assert estimate["raw_image_bytes"] == 66_060_288
    assert estimate["required_free_bytes"] > 1_345_323_008


@pytest.mark.parametrize("mutation", ["seed", "settings", "history", "reserve", "fixture", "cohort_id", "path"])
def test_changed_protocol_or_seed_fails_before_any_generation(protocol, monkeypatch, mutation):
    monkeypatch.setattr(gain, "generate_cohort", lambda *args: pytest.fail("Invalid configuration sampled"))
    if mutation == "seed":
        protocol["policy_seed_hex"] = protocol["detector_seed_hex"]
    elif mutation == "settings":
        protocol["candidates"][0]["policy"]["diversity_probe_nm"] = 79.
    elif mutation == "history":
        protocol["historical_selection_config"]["replicates"] = 1
    elif mutation == "reserve":
        protocol["operating_reserve_bytes"] -= 1
    elif mutation == "fixture":
        protocol["cohort"]["exclusions"]["wavefronts"].pop(0)
    elif mutation == "path":
        protocol["reviewed_fixture_sources"] = {"../private.py": "0" * 64}
    else:
        protocol["cohort"]["specification"]["cohort_id"] = "renamed_cohort"
    with pytest.raises(ValueError):
        gain.validate_configuration(protocol)


def test_unfrozen_admission_refuses_before_preparation(protocol, tmp_path, monkeypatch):
    monkeypatch.setattr(gain, "committed_source", lambda: ("0" * 40, source_inventory()[1]))
    monkeypatch.setattr(gain, "_prepare", lambda *args: pytest.fail("Unadmitted wavefront generation"))
    receipt = admission(protocol)
    receipt["status"] = "DRAFT"
    with pytest.raises(ValueError, match="frozen"):
        gain.run(tmp_path / "unadmitted", protocol, receipt)
    assert not (tmp_path / "unadmitted").exists()


def test_changed_prior_fingerprint_refuses_before_new_cohort_draw(protocol, monkeypatch):
    sampled = []
    def old_only(config):
        sampled.append(config["seed_hex"])
        return {"clusters": []}
    monkeypatch.setattr(gain, "generate_cohort", old_only)
    with pytest.raises(ValueError, match="Prior cohort cluster fingerprint"):
        gain._prepare(protocol)
    assert sampled == [protocol["historical_selection_config"]["excluded_cohorts"][0]["configuration"]["seed_hex"]]
    assert protocol["cohort"]["seed_hex"] not in sampled


def test_gate_fails_before_prepare_even_with_frozen_receipt(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=gain.RESERVE - 1))
    monkeypatch.setattr(gain, "_prepare", lambda *args: pytest.fail("Resource refusal sampled"))
    with pytest.raises(OSError, match="Insufficient disk"):
        gain.run(tmp_path / "unadmitted", config, admission(config))


def test_source_drift_between_admission_and_snapshot_refuses_before_preparation(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    original = gain.study_directory
    @contextmanager
    def changed(*args, **kwargs):
        with original(*args, **kwargs) as study:
            study["source_sha256"]["src/probebench/scalar_gain_study.py"] = "0" * 64
            yield study
    monkeypatch.setattr(gain, "study_directory", changed)
    monkeypatch.setattr(gain, "_prepare", lambda *args: pytest.fail("Changed source sampled"))
    with pytest.raises(ValueError, match="Admission"):
        gain.run(tmp_path / "study", config, admission(config))


def test_new_fixture_snapshot_is_verified_before_preparation(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    original = gain.shutil.copyfile
    def changed(source, destination):
        result = original(source, destination)
        if "reviewed-sources" in str(destination):
            with Path(destination).open("ab") as stream:
                stream.write(b"changed")
        return result
    monkeypatch.setattr(gain.shutil, "copyfile", changed)
    monkeypatch.setattr(gain, "_prepare", lambda *args: pytest.fail("Changed fixture snapshot sampled"))
    with pytest.raises(ValueError, match="Newly copied"):
        gain.run(tmp_path / "study", config, admission(config))


def test_case_ids_include_gain_and_arm_but_streams_preserve_frozen_identity(tiny):
    config, prepared = tiny
    cases = gain.cases_for(config, prepared)
    assert len(cases) == len({case["id"] for case in cases}) == 24
    first = cases[:6]
    assert len({(case["detector_seed"], case["policy_seed"]) for case in first}) == 1
    identity = {"cohort_id": gain.COHORT_ID, "stratum_id": "low", "base_index": 0,
                "member": "primary", "replicate": 0}
    assert first[0]["detector_seed"] == stream_seed(config["detector_seed_hex"], "detector", identity)
    config["candidates"].reverse()
    prepared["cohort"]["clusters"].reverse()
    for base in prepared["cohort"]["clusters"]:
        base["members"].reverse()
    assert gain.cases_for(config, prepared) == cases


def test_cross_grid_mapping_preserves_continuous_polynomial_modulo_piston():
    source = OpticalModel(OpticsConfig(pupil_pixels=12, oversampling=1))
    target = OpticalModel(OpticsConfig(pupil_pixels=32, oversampling=2))
    def monomials(n):
        x = 2 * (np.arange(n) - (n-1)/2) / n
        xx, yy = np.meshgrid(x, x)
        return np.stack([xx**i * yy**j for i in range(5) for j in range(5-i)], axis=-1)
    polynomial = np.linalg.lstsq(monomials(12)[source.pupil], source.wavefront_nm(TRUTH)[source.pupil], rcond=None)[0]
    expected = monomials(32)[target.pupil] @ polynomial
    expected -= expected.mean()
    mapped = gain.map_pupil_coefficients(TRUTH, asdict(source.config), asdict(target.config))
    actual = target.wavefront_nm(mapped)[target.pupil]
    actual -= actual.mean()
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    assert np.max(np.abs(target.wavefront_nm(TRUTH)[target.pupil] - expected)) > 1


@pytest.mark.parametrize("family", gain.ARMS)
def test_atomic_case_roundtrip_replays_without_refitting(tiny, tmp_path, monkeypatch, family):
    config, prepared = tiny
    study = bound(config)
    case = next(c for c in gain.cases_for(config, prepared) if c["family"] == family)
    record, images = gain._run_case(study, case)
    path = tmp_path / "case.npz"
    gain._save_case(path, config, record, images)
    saved, raw = gain._load_case(path, config)
    assert raw.dtype == np.dtype("float64")
    assert raw.tobytes() == images.tobytes() and np.any(raw < 0)
    import probebench.budget_policies as policies
    monkeypatch.setattr(policies, "estimate_phase_mle", lambda *a, **k: pytest.fail("Verification refit"))
    gain.verify_case(study, case, saved, raw)
    with pytest.raises(FileExistsError):
        gain._save_case(path, config, record, images)


def test_rehashed_raw_frame_and_command_tampering_are_rejected(tiny):
    config, prepared = tiny
    study = bound(config)
    case = gain.cases_for(config, prepared)[0]
    record, images = gain._run_case(study, case)
    images[0, 0, 0] += 1
    record["images_sha256"] = hashlib.sha256(images.tobytes()).hexdigest()
    with pytest.raises(ValueError, match="camera replay"):
        gain.verify_case(study, case, record, images)


def test_terminal_nonfinite_raw_is_retained_and_reconstructs(tiny, tmp_path):
    config, prepared = tiny
    config["read_noise_electrons"] = np.finfo(float).max
    study = bound(config)
    case = gain.cases_for(config, prepared)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        record, images = gain._run_case(study, case)
        assert record["evaluation"]["status"] == "numerical_failure"
        assert len(images) == 1 and not np.isfinite(images).all()
        gain._save_case(tmp_path / "failed.npz", config, record, images)
        saved, raw = gain._load_case(tmp_path / "failed.npz", config)
        gain.verify_case(study, case, saved, raw)
    assert record["evaluation"]["physical"]["expected_photons_used"] == 100
    assert record["evaluation"]["failure"]["produced_observation_count"] == 1


def test_metadata_cap_is_enforced_without_truncation(tiny, tmp_path):
    config, prepared = tiny
    record, images = gain._run_case(bound(config), gain.cases_for(config, prepared)[0])
    record["oversized"] = "x" * gain.CASE_JSON_LIMIT
    with pytest.raises(ValueError, match="metadata allowance"):
        gain._save_case(tmp_path / "oversized.npz", config, record, images)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("nonfinite", [False, True])
def test_complete_pending_case_is_promoted_without_reacquisition(tiny, tmp_path, monkeypatch, nonfinite):
    config, _ = tiny
    if nonfinite:
        config["read_noise_electrons"] = np.finfo(float).max
    output = tmp_path / "study"
    original = gain.os.replace
    def interrupt(source, destination):
        if str(destination).endswith(".npz"):
            raise RuntimeError("interrupt after fsync")
        return original(source, destination)
    monkeypatch.setattr(gain.os, "replace", interrupt)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(RuntimeError, match="after fsync"):
        gain.run(output, config, admission(config), limit_episodes=1)
    pending = next((output / "cases").glob("*.pending"))
    before = pending.read_bytes()
    monkeypatch.setattr(gain.os, "replace", original)
    real_run = gain._run_case
    retained_id = gain._load_case(pending, config)[0]["case"]["id"]
    calls = []
    def acquire(study, case):
        assert case["id"] != retained_id
        calls.append(case["id"])
        return real_run(study, case)
    monkeypatch.setattr(gain, "_run_case", acquire)
    with np.errstate(over="ignore", invalid="ignore"):
        result = gain.run(output, config, admission(config), resume=True, limit_episodes=1)
    assert result["completed_episodes"] == 2 and len(calls) == 1
    assert pending.with_suffix("").read_bytes() == before
    if nonfinite:
        assert gain._load_case(pending.with_suffix(""), config)[0]["evaluation"]["status"] == "numerical_failure"


def test_corrupt_pending_case_is_preserved_and_refuses(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    output = tmp_path / "study"
    gain.run(output, config, admission(config), limit_episodes=1)
    cases = gain._read_json(output / "cases.json")
    pending = gain._case_path(output, cases[1]).with_suffix(".npz.pending")
    pending.write_bytes(b"interrupted archive")
    monkeypatch.setattr(gain, "_run_case", lambda *args: pytest.fail("Corrupt pending evidence rerun"))
    with pytest.raises(zipfile.BadZipFile):
        gain.run(output, config, admission(config), resume=True)
    assert pending.read_bytes() == b"interrupted archive"


def test_unknown_partial_artifact_and_missing_source_are_not_repaired(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    output = tmp_path / "study"
    gain.run(output, config, admission(config), limit_episodes=1)
    unexpected = output / "unlisted.txt"
    unexpected.write_text("unexpected evidence")
    monkeypatch.setattr(gain, "_run_case", lambda *args: pytest.fail("Invalid inventory acquired a case"))
    with pytest.raises(ValueError, match="Unexpected or missing"):
        gain.run(output, config, admission(config), resume=True)
    unexpected.unlink()
    source = output / "source" / "src/probebench/scalar_gain_study.py"
    source.unlink()
    with pytest.raises(ValueError, match="Immutable source snapshot"):
        gain.run(output, config, admission(config), resume=True)
    assert not source.exists()


def test_completed_study_and_terminal_failures_are_never_retried(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    config["read_noise_electrons"] = np.finfo(float).max
    output = tmp_path / "study"
    with np.errstate(over="ignore", invalid="ignore"):
        gain.run(output, config, admission(config))
    before = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(gain, "_run_case", lambda *args: pytest.fail("Terminal case reacquired"))
    with np.errstate(over="ignore", invalid="ignore"):
        gain.run(output, config, admission(config), resume=True)
    assert before == {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    report = gain._read_json(output / "analysis.json")
    assert report["status"] == "withheld_missing_finite_outcomes"
    assert len(report["failed_case_ids"]) == 24
    assert "equal_stratum" not in report


def test_changed_saved_admission_refuses_before_any_cohort_regeneration(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    config["read_noise_electrons"] = np.finfo(float).max
    output = tmp_path / "study"
    with np.errstate(over="ignore", invalid="ignore"):
        gain.run(output, config, admission(config))
    study = gain._read_json(output / "study.json")
    study["configuration"]["admission"]["configuration_sha256"] = "0" * 64
    (output / "study.json").write_text(json.dumps(study))
    monkeypatch.setattr(gain, "_prepare", lambda *args: pytest.fail("Unauthenticated regeneration"))
    with pytest.raises(ValueError, match="Admission"):
        gain.verify_run(output)


def test_raw_hash_failure_refuses_before_any_cohort_regeneration(tiny, tmp_path, monkeypatch):
    config, _ = tiny
    config["read_noise_electrons"] = np.finfo(float).max
    output = tmp_path / "study"
    with np.errstate(over="ignore", invalid="ignore"):
        gain.run(output, config, admission(config))
    path = next((output / "cases").glob("*.npz"))
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    monkeypatch.setattr(gain, "_prepare", lambda *args: pytest.fail("Integrity failure sampled a cohort"))
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        gain.verify_run(output)


@pytest.mark.parametrize("damage", ["compressed", "dtype", "extra_member"])
def test_npz_reader_rejects_unsupported_containers_before_replay(tiny, tmp_path, damage):
    config, prepared = tiny
    record, images = gain._run_case(bound(config), gain.cases_for(config, prepared)[0])
    if damage == "dtype":
        images = images.astype(np.float32)
    path = tmp_path / "bad.npz"
    values = {"images": images, "metadata": np.frombuffer(gain._json_bytes(record), dtype=np.uint8)}
    if damage == "extra_member":
        values["hidden"] = np.array([1.])
    writer = np.savez_compressed if damage == "compressed" else np.savez
    writer(path, **values)
    with pytest.raises(ValueError):
        gain._load_case(path, config)


def test_total_global_budget_is_checked(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(gain, "GLOBAL_ALLOWANCE", 20)
    (tmp_path / "one").write_bytes(b"x" * 12)
    (tmp_path / "two").write_bytes(b"x" * 12)
    with pytest.raises(ValueError, match="total retention"):
        gain._global_budget(tmp_path)


def test_base_cluster_analysis_preserves_known_curve_and_fallbacks(tiny):
    config, prepared = tiny
    cases = gain.cases_for(config, prepared)
    records = []
    for case in cases:
        base = case["base_index"]
        member = 2 if case["member"] == "conjugate" else 0
        arm = 3 + 10 * (case["gain"] - 1) if case["family"] == gain.ARMS[1] else 0
        records.append({"case": case, "evaluation": {"status": "submitted",
            "physical": {"final": {"residual_wfe_nm": 10 + base + member + arm},
                         "exposures": 3, "expected_photons_used": 300}},
            "fit_records": [{"fit_status": "numerical_failure", "optimizer_success": False}],
            "elapsed_seconds": 0.1})
    result = gain.analysis(config, cases, records)
    assert result["independent_bases"] == 2
    assert result["bootstrap_seed"] == stream_seed(config["bootstrap_seed_hex"], "bootstrap",
        {"cohort_id": gain.COHORT_ID, "analysis": "paired_base_curve_v1", "replicates": 10000})
    for row, expected in zip(result["equal_stratum"], (2., 3., 4.), strict=True):
        assert row["adaptive_minus_fixed"]["mean_nm"] == pytest.approx(expected)
        np.testing.assert_allclose(row["adaptive_minus_fixed"]["interval_nm"], expected)
    reversed_result = gain.analysis(config, cases, list(reversed(records)))
    assert reversed_result["equal_stratum"] == result["equal_stratum"]
    with pytest.raises(ValueError, match="every declared"):
        gain.analysis(config, cases, records[:-1])
