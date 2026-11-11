"""Metadata-only case inventories for the frozen classical comparison.

An inventory describes required observations, not independent-wavefront proof,
historical exclusion closure, custody verification or execution admission.
"""
from __future__ import annotations

from collections import Counter
import hashlib

from .comparison_contract import FAMILIES, _keys, _label, canonical_bytes, parse_json

SCHEMA = "probebench.comparison-inventory/1"
DEVELOPMENT = "public_development_fixture"
PROSPECTIVE = "prospective_classical_comparison"
CELLS = tuple((photons, budget) for photons in (2000, 20000) for budget in (3, 7, 13))
STRATA = ("low", "medium", "high")
MAX_CASES = 96 * 2 * 2 * 6 * 6


def public_fixture_inventory():
    """Describe only the already public mixed_a fixture; create no optical data."""
    return {"schema": SCHEMA, "mode": DEVELOPMENT, "cohort_id": "public_mixed_a",
            "bases": [{"base_wavefront_id": "public_fixture:mixed_a",
                       "stratum_id": "public", "members": ["primary"]}]}


def validate_inventory(payload, *, required_mode, cohort_id):
    """Validate exact design counts before expanding cases or inspecting outcomes.

    Member sets are declared per base and must later be verified against the
    evaluator's physical descriptor. Metadata validation cannot prove them.
    """
    value = parse_json(payload)
    _keys(value, {"schema", "mode", "cohort_id", "bases"}, "comparison inventory")
    if value["schema"] != SCHEMA or required_mode not in {DEVELOPMENT, PROSPECTIVE}:
        raise ValueError("Unexpected comparison inventory schema or mode")
    if value["mode"] != required_mode or value["cohort_id"] != cohort_id:
        raise ValueError("Comparison inventory identity differs")
    _label(cohort_id, "cohort ID")
    if required_mode == DEVELOPMENT:
        if canonical_bytes(value) != canonical_bytes(public_fixture_inventory()):
            raise ValueError("Development mode accepts only the fixed public fixture")
        return value
    rows = value["bases"]
    if type(rows) is not list or len(rows) != 96:
        raise ValueError("The prospective comparison requires exactly 96 base identities")
    seen, strata = set(), Counter()
    for row in rows:
        _keys(row, {"base_wavefront_id", "stratum_id", "members"}, "base inventory")
        _label(row["base_wavefront_id"], "base identity")
        if row["base_wavefront_id"] in seen:
            raise ValueError("Duplicate base identity")
        seen.add(row["base_wavefront_id"])
        if type(row["stratum_id"]) is not str or row["stratum_id"] not in STRATA:
            raise ValueError("Unexpected comparison stratum")
        strata[row["stratum_id"]] += 1
        if row["members"] not in (["primary"], ["primary", "conjugate"]):
            raise ValueError("Each base must declare its exact primary/conjugate membership")
    if strata != {name: 32 for name in STRATA}:
        raise ValueError("The prospective comparison requires 32 bases per stratum")
    return value


def _validated(value):
    # Revalidate caller-created dictionaries; they are not admission capabilities.
    if type(value) is not dict:
        raise ValueError("Expected a plain inventory object")
    return validate_inventory(canonical_bytes(value), required_mode=value.get("mode"),
                              cohort_id=value.get("cohort_id"))


def case_inventory(inventory):
    """Expand deterministic metadata only; no wavefront, key or noise generation."""
    value = _validated(inventory)
    replicates = range(1 if value["mode"] == DEVELOPMENT else 2)
    cases = []
    for base in sorted(value["bases"], key=lambda row: (row["stratum_id"], row["base_wavefront_id"])):
        for member in base["members"]:
            for replicate in replicates:
                for photons, budget in CELLS:
                    for method in sorted(FAMILIES):
                        row = {"method": method, "track": "classical",
                               "stratum_id": base["stratum_id"],
                               "base_wavefront_id": base["base_wavefront_id"],
                               "member": member, "replicate": replicate,
                               "photons_per_exposure": photons, "exposure_budget": budget}
                        identity = {"cohort_id": value["cohort_id"], "mode": value["mode"], **row}
                        cases.append({"id": "case-" + hashlib.sha256(canonical_bytes(identity)).hexdigest(),
                                      **row})
    if len(cases) > MAX_CASES:
        raise ValueError("Comparison case inventory exceeds the fixed design")
    return cases


def inventory_counts(inventory):
    value = _validated(inventory)
    cases = case_inventory(value)
    return {"declared_base_identities": len(value["bases"]),
            "declared_members": sum(len(base["members"]) for base in value["bases"]),
            "cases": len(cases), "logical_exposures": sum(row["exposure_budget"] for row in cases),
            "expected_photons": sum(row["photons_per_exposure"] * row["exposure_budget"] for row in cases),
            "independence_verified": False}


def terminal_inventory(inventory, terminals):
    """Keep terminal completeness distinct from available scientific outcomes.

    A recorded failure is a terminal outcome, but cannot be given an invented
    finite penalty to make the complete-contrast reducer accept it.
    """
    expected = {row["id"] for row in case_inventory(inventory)}
    if type(terminals) is not list or len(terminals) > len(expected):
        raise ValueError("Terminal inventory exceeds the declared case count")
    observed, unavailable = set(), []
    for row in terminals:
        _keys(row, {"case_id", "status"}, "terminal identity")
        identity, status = row["case_id"], row["status"]
        if type(identity) is not str or identity not in expected or identity in observed:
            raise ValueError("Unknown or duplicate terminal identity")
        if type(status) is not str or status not in {"completed", "failed"}:
            raise ValueError("Unexpected terminal status")
        observed.add(identity)
        if status == "failed":
            unavailable.append(identity)
    missing = sorted(expected - observed)
    return {"required": len(expected), "terminal": len(observed),
            "missing_case_ids": missing, "failed_case_ids": sorted(unavailable),
            "all_terminal_identities_present": not missing,
            "all_scientific_outcomes_available": not missing and not unavailable,
            "artifact_verification": "not_performed_by_metadata_validator"}
