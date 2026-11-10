"""Bounded, non-executing validation of a prospective classical comparison.

This module uses only the standard library. Validation is not registration,
resource admission, exclusion verification, or permission to open a reserve.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


SCHEMA = "probebench.classical-comparison/1"
CUSTODY_SCHEMA = "probebench.comparison-custody/1"
PURPOSE = "classical_heldout_comparison_v1"
SELECTED_SHA256 = "c858028008125b277bb6f9dcf1696e9ada1e0b5bada1ab19a577688cb0e50cab"
SELECTION_SHA256 = "03e63c3f3b98b8169de50c0473727110218486f30f0d885199d4f6ed1f1777d2"
SELECTION_STUDY_SHA256 = "ef9d4511cbed4dad803bbfd27d63cb86446354d28b9fdb81fb9145df92ee9e3f"
MAX_JSON_BYTES = 1_048_576
MAX_SOURCE_BYTES = 16_777_216
MAX_SOURCE_FILE_BYTES = 1_048_576
MAX_NODES = 20_000
MAX_DEPTH = 16

FAMILIES = {
    "no_action_v2": "no_action_default",
    "unprobed_mle_v1": "unprobed_default",
    "random_probe_v2": "random_amp_020",
    "coordinate_search_v2": "coordinate_step_040",
    "fixed_diversity_mle_v1": "fixed_probe_080",
    "phase_diversity_mle_v1": "adaptive_probe_080",
}
REFERENCE = "fixed_diversity_mle_v1"
TARGET = "phase_diversity_mle_v1"
SOURCE_ROLES = {
    "optical_model": "src/probebench/optics.py",
    "environment": "src/probebench/environment.py",
    "measurement_score": "src/probebench/policies.py",
    "policies": "src/probebench/budget_policies.py",
    "estimator": "src/probebench/estimation.py",
    "likelihood": "src/probebench/likelihood.py",
    "cohort_sampler": "src/probebench/cohorts.py",
    "selection": "src/probebench/selection.py",
    "selection_statistics": "src/probebench/selection_statistics.py",
    "comparison_statistics": "src/probebench/paired_statistics.py",
    "observation_archive": "src/probebench/sequential.py",
    "study_storage": "src/probebench/study.py",
    "publication_primitives": "src/probebench/qualification.py",
    "source_inventory": "src/probebench/run.py",
    "contract_validator": "src/probebench/comparison_contract.py",
    "package": "src/probebench/__init__.py",
    "dependencies": "pyproject.toml",
    "dependency_lock": "uv.lock",
}
SELECTED_SOURCE_SHA256 = {
    "src/probebench/optics.py": "abdb6315033fa7607102803eb528608e048fd1593bdde9c14415924bc770123f",
    "src/probebench/environment.py": "88f5b3ba7dfb883638e005693547498403c7976e1ed9ed8a1fffc91e7635d2b5",
    "src/probebench/policies.py": "c84de28ac19fbc9f961225fbb0f57c37277c75f3ab66082d8b2cb31042354e80",
    "src/probebench/budget_policies.py": "57a7d3f172d726c5b1d9879aaffa3b81e59f8cf7d9ac73b9a79432b7fa75c29c",
    "src/probebench/estimation.py": "95118ed3fe9061c834af5404689476e1ea31693152465525a6b8882a81312674",
    "src/probebench/likelihood.py": "7cfd9020d8881149c096212a1f27661b12703adfb30a074d1f19e81f9b592419",
}


@dataclass(frozen=True)
class ValidatedComparison:
    """Detached bytes and identity; never an execution-admission token."""

    canonical_json: bytes
    sha256: str
    status: str


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"Unexpected {name} fields")


def _digest(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {name} SHA-256")


def _label(value, name):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError(f"Invalid {name}")


def _bounded(value):
    # Conservative JSON-size bound before encoding; never serialize a huge object
    # merely to discover that it exceeds the allowance. Repeated/cyclic containers
    # also terminate at the depth/node bound.
    stack, nodes, budget = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            raise ValueError("JSON structure exceeds its bound")
        kind = type(item)
        if kind is dict:
            if nodes + len(stack) + len(item) > MAX_NODES:
                raise ValueError("JSON object exceeds its bound")
            budget += 2 + 2 * len(item)
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                budget += 2 + 12 * len(key)
                stack.append((child, depth + 1))
        elif kind is list:
            if nodes + len(stack) + len(item) > MAX_NODES:
                raise ValueError("JSON array exceeds its bound")
            budget += 2 + len(item)
            stack.extend((child, depth + 1) for child in item)
        elif kind is str:
            budget += 2 + 12 * len(item)
        elif kind is int:
            if item.bit_length() > 63:
                raise ValueError("JSON integer exceeds its bound")
            budget += 32
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            budget += 32
        elif kind is bool or item is None:
            budget += 5
        else:
            raise ValueError("Only plain JSON values are supported")
        if budget > MAX_JSON_BYTES:
            raise ValueError("JSON encoded-size bound exceeded")


def canonical_bytes(value):
    """Return bounded canonical JSON; this performs no scientific validation."""
    _bounded(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _integer_text(value):
    if len(value.lstrip("-")) > 19:
        raise ValueError("JSON integer exceeds its bound")
    result = int(value)
    if result.bit_length() > 63:
        raise ValueError("JSON integer exceeds its bound")
    return result


def _constant(_):
    raise ValueError("JSON numbers must be finite")


def parse_json(payload):
    """Decode a bounded UTF-8 JSON document with unambiguous keys/numbers."""
    if type(payload) is not bytes or len(payload) > MAX_JSON_BYTES:
        raise ValueError("JSON input must be bounded bytes")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique,
                           parse_int=_integer_text, parse_constant=_constant)
        _bounded(value)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded JSON") from error
    return value


def _same(actual, expected, name):
    # Canonical comparison distinguishes bool/int and int/float as well as fields.
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ValueError(f"Changed {name}")


def _selected(payload):
    if type(payload) is not bytes or len(payload) > MAX_JSON_BYTES or _sha(payload) != SELECTED_SHA256:
        raise ValueError("Selected-settings bytes do not match the trusted release")
    selected = parse_json(payload)
    _keys(selected, {"schema", "purpose", "study_sha256", "selection_sha256", "families"},
          "selected-settings")
    expected = {
        "schema": 1, "purpose": "classical_parameter_selection_v1",
        "study_sha256": SELECTION_STUDY_SHA256, "selection_sha256": SELECTION_SHA256,
        "families": {family: {"candidate_id": candidate, "policy": {
            "coordinate_step_floor_nm": 5.0, "coordinate_step_nm": 40.0,
            "diversity_probe_nm": 80.0, "max_mle_iterations": 200,
            "random_amplitude_nm": 20.0 if family == "random_probe_v2" else 60.0,
        }} for family, candidate in FAMILIES.items()},
    }
    _same(selected, expected, "selected-settings contents")
    return selected


def _sources(value, files):
    _keys(value, {"revision", "roles", "files"}, "source")
    if type(value["revision"]) is not str or re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is None:
        raise ValueError("Invalid source revision")
    _same(value["roles"], SOURCE_ROLES, "source roles")
    declared = value["files"]
    if type(declared) is not dict or not 1 <= len(declared) <= 128:
        raise ValueError("Source inventory exceeds its file-count allowance")
    if not set(SOURCE_ROLES.values()) <= set(declared):
        raise ValueError("Required source files are missing")
    if type(files) is not dict or len(files) != len(declared) or set(files) != set(declared):
        raise ValueError("Source bytes must match the complete declared inventory")
    total = 0
    for path, expected in declared.items():
        if (type(path) is not str or len(path) > 240
                or re.fullmatch(r"[A-Za-z0-9_./-]+", path) is None
                or PurePosixPath(path).is_absolute()
                or any(part in {"", ".", ".."} for part in path.split("/"))):
            raise ValueError("Source inventory requires normalized relative paths")
        _digest(expected, "source")
        payload = files[path]
        if type(payload) is not bytes or len(payload) > MAX_SOURCE_FILE_BYTES:
            raise ValueError("Source file exceeds its byte allowance")
        total += len(payload)
        if total > MAX_SOURCE_BYTES:
            raise ValueError("Source inventory exceeds its byte allowance")
        if _sha(payload) != expected:
            raise ValueError("Source file digest mismatch")
    for path, expected in SELECTED_SOURCE_SHA256.items():
        if declared[path] != expected:
            raise ValueError("Selected optical or policy implementation changed")


def validate_comparison_contract(payload, *, selected_settings_bytes,
                                 source_files, exclusion_inventory_bytes):
    """Validate explicit design choices and supplied bytes, without opening a reserve.

    Inputs are caller-supplied bytes, not paths. Hash matching does not authenticate
    their origin or certify the completeness/independence of the exposure history.
    No numerical modules, optical models, acquisition or analysis are invoked.
    """
    contract = parse_json(payload)
    _keys(contract, {"schema", "purpose", "status", "cohort_id", "selection", "population",
                     "experiment", "analysis", "failures", "sources", "runtime",
                     "exclusions", "custody_schema"}, "comparison contract")
    _same(contract["schema"], SCHEMA, "schema")
    _same(contract["purpose"], PURPOSE, "purpose")
    if type(contract["status"]) is not str or contract["status"] not in {"proposal", "synthetic_fixture"}:
        raise ValueError("Contract validation cannot register or admit an experiment")
    _label(contract["cohort_id"], "cohort ID")
    if contract["status"] == "synthetic_fixture" and not contract["cohort_id"].startswith("synthetic_"):
        raise ValueError("Synthetic contracts must identify their fixture namespace")
    selected = _selected(selected_settings_bytes)
    _same(contract["selection"], {"artifact_sha256": SELECTED_SHA256, **selected}, "frozen selection")
    _same(contract["population"], {
        "sampler": "finite_pupil_isotropic_uniform_rms_pcg64_v1", "split": "reserved",
        "independent_base_count": 96, "stratum_weighting": "equal",
        "strata": [{"id": name, "rms_min_nm": low, "rms_max_nm": high, "base_count": 32}
                   for name, low, high in (("low", 10.0, 25.0), ("medium", 25.0, 50.0),
                                           ("high", 50.0, 90.0))],
        "optics": {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0},
        "mode_order": ["defocus", "astigmatism_cos", "astigmatism_sin", "coma_x", "coma_y", "spherical"],
        "command_limit_nm": 200.0, "overlap_tolerance_nm": 1e-6,
        "conjugate_signs": [-1, -1, -1, 1, 1, -1],
        "singleton_rule": "one_member_same_base_weight",
        "overlap_action": "stop_without_replacement",
        "sample_size_rationale": "fixed_descriptive_precision_budget_not_powered",
    }, "population")
    _same(contract["experiment"], {
        "track": "classical", "photons_per_exposure": [2000, 20000],
        "exposure_budgets": [3, 7, 13], "detector_replicates": 2,
        "read_noise_electrons": 1.0, "actual_command_gain": 1.0,
        "nominal_command_gain": 1.0, "incident_wavefront": "static",
        "initial_zero_command_exposure_charged": True, "final_submission_exposure": False,
        "paired_stream_identity": ["protocol", "base_wavefront_id", "member", "replicate",
                                   "photons_per_exposure"],
        "stream_domains": ["wavefront", "detector", "policy", "bootstrap"],
        "policy_inputs": "public_optics_instrument_policy_randomness_observations_only",
        "budget_semantics": "equal_exposures_and_expected_photons_not_equal_compute",
    }, "experiment")
    _same(contract["analysis"], {
        "outcome": "final_residual_wfe_nm", "contrast": "target_minus_reference",
        "hierarchy": ["replicate", "member", "base", "stratum"],
        "cell_order": [[photons, budget] for photons in (2000, 20000) for budget in (3, 7, 13)],
        "primary": {"reference": REFERENCE, "target": TARGET, "family_size": 6,
                    "family_alpha": 0.05, "marginal_confidence": 1.0 - 0.05 / 6,
                    "resamples": 50000, "multiplicity": "bonferroni_six_cells",
                    "coverage": "nominal_conditional_on_marginal_bootstrap_coverage"},
        "secondary": {"reference": REFERENCE,
                      "targets": sorted(set(FAMILIES) - {REFERENCE, TARGET}),
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
    }, "analysis")
    _same(contract["failures"], {
        "finite_nonconvergence": "score_and_retain_all_start_statuses",
        "numerical_fit_failure": "existing_best_measured_camera_score_fallback",
        "unexpected_policy_error": "retain_failure_withhold_complete_contrasts",
        "nonfinite_evaluator": "retain_failure_withhold_complete_contrasts",
        "invalid_or_missing_evidence": "stop_preserve_no_replacement",
        "resource_interruption": "preserve_prefix_no_outcome_conditioned_exclusion",
        "unknown_dispatch_outcome": "stop_reconcile_no_invisible_retry",
        "completed_case": "immutable_no_reexecution",
        "completion_gate": "all_declared_terminal_identities_and_raw_evidence_verified",
        "accounting": ["logical_exposures", "expected_photons", "simulator_replay_calls",
                       "fit_calls", "optimizer_starts", "optimizer_iterations",
                       "interrupted_attempts", "wall_time"],
    }, "failure rules")
    _sources(contract["sources"], source_files)
    runtime = contract["runtime"]
    _keys(runtime, {"python", "numpy", "scipy", "matplotlib", "platform", "machine",
                    "interpreter_sha256"}, "runtime")
    for name, expected in {"python": "3.11.15", "numpy": "2.4.6", "scipy": "1.17.1",
                           "matplotlib": "3.11.2"}.items():
        _same(runtime[name], expected, f"{name} runtime pin")
    for name in ("platform", "machine"):
        _label(runtime[name], name)
    _digest(runtime["interpreter_sha256"], "interpreter")
    exclusions = contract["exclusions"]
    _keys(exclusions, {"inventory_sha256", "verification"}, "exclusions")
    _digest(exclusions["inventory_sha256"], "exclusion inventory")
    _same(exclusions["verification"], "pending_full_precision_identity_and_orbit_closure",
          "exclusion verification state")
    # A supplied inventory is content-bound, not certified by this non-executing API.
    parse_json(exclusion_inventory_bytes)
    if _sha(exclusion_inventory_bytes) != exclusions["inventory_sha256"]:
        raise ValueError("Exclusion inventory digest mismatch")
    _same(contract["custody_schema"], CUSTODY_SCHEMA, "custody schema")
    encoded = canonical_bytes(contract)
    return ValidatedComparison(encoded, _sha(encoded), contract["status"])


def validate_custody_receipt(payload, *, contract, selected_settings_bytes,
                            source_files, exclusion_inventory_bytes):
    """Bind a later custody statement without rewriting the scientific contract.

    This checks statement format/binding, not the existence, entropy, timing or
    secrecy of keys. The supplied evidence is revalidated; the dataclass alone is
    not proof of an earlier validation. This creates no keys and admits no work.
    """
    # The dataclass is a public value object, not an unforgeable capability.
    # Bound its fields before hashing a caller-constructed instance.
    if (type(contract) is not ValidatedComparison
            or type(contract.canonical_json) is not bytes
            or len(contract.canonical_json) > MAX_JSON_BYTES
            or type(contract.status) is not str
            or contract.status not in {"proposal", "synthetic_fixture"}):
        raise ValueError("An intact validated comparison identity is required")
    _digest(contract.sha256, "comparison identity")
    if _sha(contract.canonical_json) != contract.sha256:
        raise ValueError("An intact validated comparison identity is required")
    refreshed = validate_comparison_contract(
        contract.canonical_json, selected_settings_bytes=selected_settings_bytes,
        source_files=source_files, exclusion_inventory_bytes=exclusion_inventory_bytes,
    )
    if (refreshed.canonical_json != contract.canonical_json
            or refreshed.sha256 != contract.sha256 or refreshed.status != contract.status):
        raise ValueError("Comparison value disagrees with its canonical scientific contents")
    value = parse_json(payload)
    _keys(value, {"schema", "status", "scientific_contract_sha256", "commitment_scheme",
                 "commitments"}, "custody receipt")
    _same(value["schema"], CUSTODY_SCHEMA, "custody schema")
    _same(value["scientific_contract_sha256"], contract.sha256, "scientific contract identity")
    expected_status = "synthetic_fixture" if contract.status == "synthetic_fixture" else "unverified_statement"
    _same(value["status"], expected_status, "custody status")
    _same(value["commitment_scheme"], "sha256_domain_separated_256bit_keys_v1", "commitment scheme")
    commitments = value["commitments"]
    _keys(commitments, {"wavefront", "detector", "policy", "bootstrap"}, "commitments")
    for name, digest in commitments.items():
        _digest(digest, name)
    if len(set(commitments.values())) != 4:
        raise ValueError("Random-domain commitments must be distinct")
    return {"scientific_contract_sha256": contract.sha256,
            "custody_statement_sha256": _sha(canonical_bytes(value)),
            "status": "statement_bound_not_verified", "execution_admitted": False}
