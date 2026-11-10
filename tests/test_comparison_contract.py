"""Synthetic contract checks; no reserved cohort, model or camera is constructed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from probebench import comparison_contract as contract_api


ROOT = Path(__file__).resolve().parents[1]


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def inputs():
    selected = (ROOT / "results/classical-selection-v1-20261016/selected-settings.json").read_bytes()
    source_paths = [*sorted((ROOT / "src/probebench").glob("*.py")),
                    ROOT / "pyproject.toml", ROOT / "uv.lock"]
    source_files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in source_paths}
    exclusions = b'{"status":"synthetic_fixture","note":"No physical cohort or exclusion claim"}\n'
    value = {
        "schema": "probebench.classical-comparison/1",
        "purpose": "classical_heldout_comparison_v1", "status": "synthetic_fixture",
        "cohort_id": "synthetic_contract_only",
        "selection": {"artifact_sha256": digest(selected), **json.loads(selected)},
        "population": {
            "sampler": "finite_pupil_isotropic_uniform_rms_pcg64_v1", "split": "reserved",
            "independent_base_count": 96, "stratum_weighting": "equal",
            "strata": [{"id": "low", "rms_min_nm": 10.0, "rms_max_nm": 25.0, "base_count": 32},
                       {"id": "medium", "rms_min_nm": 25.0, "rms_max_nm": 50.0, "base_count": 32},
                       {"id": "high", "rms_min_nm": 50.0, "rms_max_nm": 90.0, "base_count": 32}],
            "optics": {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0},
            "mode_order": ["defocus", "astigmatism_cos", "astigmatism_sin", "coma_x", "coma_y", "spherical"],
            "command_limit_nm": 200.0, "overlap_tolerance_nm": 1e-6,
            "conjugate_signs": [-1, -1, -1, 1, 1, -1],
            "singleton_rule": "one_member_same_base_weight", "overlap_action": "stop_without_replacement",
            "sample_size_rationale": "fixed_descriptive_precision_budget_not_powered",
        },
        "experiment": {
            "track": "classical", "photons_per_exposure": [2000, 20000], "exposure_budgets": [3, 7, 13],
            "detector_replicates": 2, "read_noise_electrons": 1.0,
            "actual_command_gain": 1.0, "nominal_command_gain": 1.0, "incident_wavefront": "static",
            "initial_zero_command_exposure_charged": True, "final_submission_exposure": False,
            "paired_stream_identity": ["protocol", "base_wavefront_id", "member", "replicate", "photons_per_exposure"],
            "stream_domains": ["wavefront", "detector", "policy", "bootstrap"],
            "policy_inputs": "public_optics_instrument_policy_randomness_observations_only",
            "budget_semantics": "equal_exposures_and_expected_photons_not_equal_compute",
        },
        "analysis": {
            "outcome": "final_residual_wfe_nm", "contrast": "target_minus_reference",
            "hierarchy": ["replicate", "member", "base", "stratum"],
            "cell_order": [[2000, 3], [2000, 7], [2000, 13], [20000, 3], [20000, 7], [20000, 13]],
            "primary": {"reference": "fixed_diversity_mle_v1", "target": "phase_diversity_mle_v1",
                        "family_size": 6, "family_alpha": 0.05, "marginal_confidence": 0.9916666666666667,
                        "resamples": 50000, "multiplicity": "bonferroni_six_cells",
                        "coverage": "nominal_conditional_on_marginal_bootstrap_coverage"},
            "secondary": {"reference": "fixed_diversity_mle_v1",
                          "targets": ["coordinate_search_v2", "no_action_v2", "random_probe_v2", "unprobed_mle_v1"],
                          "confidence": 0.95, "resamples": 10000,
                          "scope": "descriptive_pointwise_no_confirmatory_claim"},
            "resampling": "whole_base_curves_within_stratum", "quantile": "linear",
            "reference_reselection": False, "interim_outcome_inspection": False,
            "optional_stopping": False, "equivalence_claim": False,
            "cell_advantage": "primary_upper_endpoint_strictly_below_zero",
            "curve_advantage": "all_six_primary_cells_advantage",
            "degenerate_interval": "retain_warning_no_certainty_or_equivalence",
            "secondary_metrics": ["absolute_wfe", "strehl", "maximum_acquired_probe_wfe_increase",
                                  "fit_statuses", "exposures", "expected_photons", "operational_costs"],
        },
        "failures": {
            "finite_nonconvergence": "score_and_retain_all_start_statuses",
            "numerical_fit_failure": "existing_best_measured_camera_score_fallback",
            "unexpected_policy_error": "retain_failure_withhold_complete_contrasts",
            "nonfinite_evaluator": "retain_failure_withhold_complete_contrasts",
            "invalid_or_missing_evidence": "stop_preserve_no_replacement",
            "resource_interruption": "preserve_prefix_no_outcome_conditioned_exclusion",
            "unknown_dispatch_outcome": "stop_reconcile_no_invisible_retry",
            "completed_case": "immutable_no_reexecution",
            "completion_gate": "all_declared_terminal_identities_and_raw_evidence_verified",
            "accounting": ["logical_exposures", "expected_photons", "simulator_replay_calls", "fit_calls",
                           "optimizer_starts", "optimizer_iterations", "interrupted_attempts", "wall_time"],
        },
        "sources": {"revision": "0" * 40, "roles": dict(contract_api.SOURCE_ROLES),
                    "files": {path: digest(data) for path, data in source_files.items()}},
        "runtime": {"python": "3.11.15", "numpy": "2.4.6", "scipy": "1.17.1", "matplotlib": "3.11.2",
                    "platform": "synthetic-platform", "machine": "synthetic-machine",
                    "interpreter_sha256": "0" * 64},
        "exclusions": {"inventory_sha256": digest(exclusions),
                       "verification": "pending_full_precision_identity_and_orbit_closure"},
        "custody_schema": "probebench.comparison-custody/1",
    }
    return value, {"selected_settings_bytes": selected, "source_files": source_files,
                   "exclusion_inventory_bytes": exclusions}


def validate(inputs):
    value, evidence = inputs
    return contract_api.validate_comparison_contract(encode(value), **evidence)


def test_synthetic_contract_is_detached_and_does_not_authorize_execution(inputs, monkeypatch):
    value, evidence = inputs
    before = deepcopy(value)
    # All file reads happened in fixture setup. Validation receives bytes only.
    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("Unexpected file access"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Unexpected path access"))
    result = validate(inputs)
    assert value == before
    assert result.sha256 == digest(encode(value))
    assert result.status == "synthetic_fixture"
    assert json.loads(result.canonical_json)["analysis"]["primary"]["family_size"] == 6
    value["cohort_id"] = "synthetic_later"
    evidence["source_files"].clear()
    assert result.canonical_json == encode(before)


@pytest.mark.parametrize("path,replacement", [
    (("status",), "registered"), (("cohort_id",), "apparently_real_reserve"),
    (("population", "independent_base_count"), 12),
    (("population", "split"), "selection"),
    (("population", "strata", 0, "base_count"), True),
    (("population", "conjugate_signs"), [-1] * 6),
    (("population", "stratum_weighting"), "observed_prevalence"),
    (("population", "overlap_action"), "resample"),
    (("experiment", "detector_replicates"), 1),
    (("experiment", "photons_per_exposure"), [2000, 2000]),
    (("experiment", "exposure_budgets"), [3, 7]),
    (("experiment", "initial_zero_command_exposure_charged"), False),
    (("experiment", "final_submission_exposure"), 0),
    (("experiment", "paired_stream_identity"), ["method", "budget"]),
    (("experiment", "policy_inputs"), "full_evaluator"),
    (("analysis", "primary", "reference"), "no_action_v2"),
    (("analysis", "primary", "family_size"), 1),
    (("analysis", "primary", "marginal_confidence"), 0.95),
    (("analysis", "primary", "resamples"), 10000),
    (("analysis", "primary", "coverage"), "exact_simultaneous"),
    (("analysis", "secondary", "scope"), "confirmatory"),
    (("analysis", "resampling"), "independent_frames"),
    (("analysis", "optional_stopping"), True),
    (("analysis", "equivalence_claim"), True),
    (("failures", "finite_nonconvergence"), "drop"),
    (("failures", "unknown_dispatch_outcome"), "retry"),
    (("exclusions", "verification"), "complete"),
    (("runtime", "numpy"), "different"),
])
def test_changed_scientific_choices_are_rejected(inputs, path, replacement):
    value = inputs[0]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement
    with pytest.raises(ValueError):
        validate(inputs)


@pytest.mark.parametrize("section", ["population", "experiment", "analysis", "failures"])
def test_private_or_unexpected_fields_are_rejected(inputs, section):
    inputs[0][section]["private_seed"] = "do-not-use"
    with pytest.raises(ValueError):
        validate(inputs)


def test_selected_release_cannot_be_replaced_by_a_self_consistent_claim(inputs):
    value, evidence = inputs
    altered = json.loads(evidence["selected_settings_bytes"])
    altered["families"]["random_probe_v2"]["policy"]["random_amplitude_nm"] = 80.0
    evidence["selected_settings_bytes"] = encode(altered)
    value["selection"] = {"artifact_sha256": digest(encode(altered)), **altered}
    with pytest.raises(ValueError, match="trusted release"):
        validate(inputs)


@pytest.mark.parametrize("mutation", ["omitted_family", "inactive_field", "candidate", "extra_policy_field"])
def test_full_selected_policy_fields_are_bound(inputs, mutation):
    family = inputs[0]["selection"]["families"]
    if mutation == "omitted_family":
        del family["no_action_v2"]
    elif mutation == "inactive_field":
        family["no_action_v2"]["policy"]["random_amplitude_nm"] = 20.0
    elif mutation == "candidate":
        family["random_probe_v2"]["candidate_id"] = "random_amp_080"
    else:
        family["unprobed_mle_v1"]["policy"]["extra_fit"] = True
    with pytest.raises(ValueError, match="frozen selection"):
        validate(inputs)


@pytest.mark.parametrize("mutation", ["bytes", "missing", "extra", "role", "selected_source"])
def test_source_content_inventory_roles_and_selected_implementation_are_checked(inputs, mutation):
    value, evidence = inputs
    path = "src/probebench/optics.py"
    if mutation in {"bytes", "selected_source"}:
        evidence["source_files"][path] += b"\n# altered\n"
        if mutation == "selected_source":
            value["sources"]["files"][path] = digest(evidence["source_files"][path])
    elif mutation == "missing":
        del evidence["source_files"][path]
    elif mutation == "extra":
        evidence["source_files"]["undeclared.py"] = b""
    else:
        value["sources"]["roles"]["policies"] = "src/probebench/run.py"
    with pytest.raises(ValueError):
        validate(inputs)


def test_source_inventory_count_is_checked_before_copying_caller_keys(inputs, monkeypatch):
    evidence = inputs[1]
    files = evidence["source_files"]
    files["undeclared.py"] = b"synthetic extra file"
    real_set = set

    def checked_set(value):
        if value is files:
            pytest.fail("Copied caller source keys before checking count")
        return real_set(value)

    monkeypatch.setattr(contract_api, "set", checked_set, raising=False)
    with pytest.raises(ValueError, match="complete declared inventory"):
        validate(inputs)


def test_exclusion_bytes_are_bound_without_claiming_physical_verification(inputs):
    inputs[1]["exclusion_inventory_bytes"] = b'{"different":true}'
    with pytest.raises(ValueError, match="Exclusion inventory digest"):
        validate(inputs)


@pytest.mark.parametrize("path", ["../escape.py", "/absolute.py", "folder//file.py", "x\\y.py"])
def test_source_paths_cannot_escape_a_snapshot(inputs, path):
    value, evidence = inputs
    evidence["source_files"][path] = b"synthetic extra source"
    value["sources"]["files"][path] = digest(evidence["source_files"][path])
    with pytest.raises(ValueError, match="normalized relative paths"):
        validate(inputs)


def test_content_binding_changes_for_a_different_declared_host_or_cohort(inputs):
    original = validate(inputs)
    inputs[0]["cohort_id"] = "synthetic_second_contract"
    changed = validate(inputs)
    assert original.sha256 != changed.sha256
    inputs[0]["runtime"]["machine"] = "synthetic-second-machine"
    assert validate(inputs).sha256 != changed.sha256


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'{"a":-9999999999999999999999999}',
    b'"\xff"', b'[' * 1000 + b'0' + b']' * 1000,
])
def test_ambiguous_nonfinite_or_deep_json_is_rejected(raw):
    with pytest.raises(ValueError):
        contract_api.parse_json(raw)


def test_oversized_in_memory_values_fail_before_json_encoding(monkeypatch):
    monkeypatch.setattr(contract_api.json, "dumps", lambda *a, **k: pytest.fail("Encoded oversized input"))
    for value in (-1 << 100000, "x" * contract_api.MAX_JSON_BYTES,
                  ["x" * 10000] * 20, {"x" * 100000: None}):
        with pytest.raises(ValueError):
            contract_api.canonical_bytes(value)


def test_oversized_encoded_input_is_rejected_before_json_parser(monkeypatch):
    monkeypatch.setattr(contract_api.json, "loads", lambda *a, **k: pytest.fail("Parsed oversized input"))
    with pytest.raises(ValueError, match="bounded bytes"):
        contract_api.parse_json(b" " * (contract_api.MAX_JSON_BYTES + 1))


def custody(identity):
    return {"schema": "probebench.comparison-custody/1", "status": "synthetic_fixture",
            "scientific_contract_sha256": identity,
            "commitment_scheme": "sha256_domain_separated_256bit_keys_v1",
            "commitments": {name: digest(f"synthetic statement only: {name}".encode())
                            for name in ("wavefront", "detector", "policy", "bootstrap")}}


def test_later_custody_statement_has_its_own_identity_and_no_admission(inputs):
    scientific = validate(inputs)
    result = contract_api.validate_custody_receipt(
        encode(custody(scientific.sha256)), contract=scientific, **inputs[1],
    )
    assert result["scientific_contract_sha256"] == scientific.sha256
    assert result["custody_statement_sha256"] != scientific.sha256
    assert result["execution_admitted"] is False
    assert result["status"] == "statement_bound_not_verified"
    assert scientific.canonical_json == encode(inputs[0])
    with pytest.raises(ValueError, match="scientific contract identity"):
        contract_api.validate_custody_receipt(encode(custody("0" * 64)), contract=scientific, **inputs[1])


@pytest.mark.parametrize("mutation", ["duplicate_domain", "missing_domain", "extra_seed", "asserted_registration"])
def test_custody_rejects_ambiguous_domains_secrets_and_registration_claims(inputs, mutation):
    scientific = validate(inputs)
    value = custody(scientific.sha256)
    if mutation == "duplicate_domain":
        value["commitments"]["detector"] = value["commitments"]["wavefront"]
    elif mutation == "missing_domain":
        del value["commitments"]["bootstrap"]
    elif mutation == "extra_seed":
        value["seed_hex"] = "never-create-a-real-key-in-tests"
    else:
        value["status"] = "registered"
    with pytest.raises(ValueError):
        contract_api.validate_custody_receipt(encode(value), contract=scientific, **inputs[1])


@pytest.mark.parametrize("mutation", ["status", "scientific_content", "non_json", "noncanonical"])
def test_constructed_comparison_must_match_revalidated_scientific_contents(inputs, mutation):
    scientific = validate(inputs)
    payload, status = scientific.canonical_json, scientific.status
    if mutation == "status":
        status = "proposal"
    elif mutation == "scientific_content":
        changed = json.loads(payload)
        changed["population"]["independent_base_count"] = 12
        payload = encode(changed)
    elif mutation == "non_json":
        payload = b"not a scientific contract"
    else:
        payload += b"\n"
    forged = contract_api.ValidatedComparison(payload, digest(payload), status)
    statement = custody(forged.sha256)
    if mutation == "status":
        statement["status"] = "unverified_statement"
    with pytest.raises(ValueError):
        contract_api.validate_custody_receipt(encode(statement), contract=forged, **inputs[1])


@pytest.mark.parametrize("payload,identity,status", [
    (None, "0" * 64, "synthetic_fixture"),
    (bytearray(b"{}"), "0" * 64, "synthetic_fixture"),
    (b" " * (contract_api.MAX_JSON_BYTES + 1), "0" * 64, "synthetic_fixture"),
    (b"{}", None, "synthetic_fixture"),
    (b"{}", "0" * 64, "registered"),
])
def test_forged_contract_fields_are_bounded_before_custody_hashing(inputs, monkeypatch, payload, identity, status):
    forged = contract_api.ValidatedComparison(payload, identity, status)
    monkeypatch.setattr(contract_api, "_sha", lambda *a: pytest.fail("Hashed unbounded contract fields"))
    with pytest.raises(ValueError):
        contract_api.validate_custody_receipt(b"{}", contract=forged, **inputs[1])


def test_stdlib_only_import_in_isolated_interpreter():
    script = """
import builtins, importlib.util, sys
real_import = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'matplotlib', 'probebench'}:
        raise AssertionError('Scientific dependency imported: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = checked
spec = importlib.util.spec_from_file_location('isolated_contract', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.PURPOSE == 'classical_heldout_comparison_v1'
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                                str(ROOT / "src/probebench/comparison_contract.py")],
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
