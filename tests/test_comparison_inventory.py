"""Synthetic metadata checks; these IDs are not scientific cohort reservations."""
from copy import deepcopy

import pytest

from probebench import comparison_inventory as inventory_api
from probebench.comparison_contract import canonical_bytes


def synthetic_inventory(singleton=False):
    rows = [{"base_wavefront_id": f"synthetic_{stratum}_{i:02d}", "stratum_id": stratum,
             "members": ["primary", "conjugate"]}
            for stratum in ("low", "medium", "high") for i in range(32)]
    if singleton:
        rows[0]["members"] = ["primary"]
    return {"schema": inventory_api.SCHEMA, "mode": inventory_api.PROSPECTIVE,
            "cohort_id": "synthetic_contract_only", "bases": rows}


def test_exact_global_counts_and_singleton_weighting_metadata():
    value = synthetic_inventory(singleton=True)
    cases = inventory_api.case_inventory(value)
    assert len(cases) == 72*191
    assert len({row["id"] for row in cases}) == len(cases)
    assert {row["replicate"] for row in cases} == {0, 1}
    assert {(row["photons_per_exposure"], row["exposure_budget"]) for row in cases} == {
        (2000,3), (2000,7), (2000,13), (20000,3), (20000,7), (20000,13)}
    single = [row for row in cases if row["base_wavefront_id"] == "synthetic_low_00"]
    assert len(single) == 72 and {row["member"] for row in single} == {"primary"}
    shuffled = deepcopy(value)
    shuffled["bases"].reverse()
    assert inventory_api.case_inventory(shuffled) == cases


@pytest.mark.parametrize("corruption", ["missing_base", "duplicate", "wrong_stratum", "member", "extra"])
def test_global_inventory_rejects_consistent_design_omissions(corruption):
    value = synthetic_inventory()
    if corruption == "missing_base":
        value["bases"].pop()
    elif corruption == "duplicate":
        value["bases"][1]["base_wavefront_id"] = value["bases"][0]["base_wavefront_id"]
    elif corruption == "wrong_stratum":
        value["bases"][0]["stratum_id"] = "high"
    elif corruption == "member":
        value["bases"][0]["members"] = ["conjugate"]
    else:
        value["bases"][0]["seed"] = 1
    with pytest.raises(ValueError):
        inventory_api.case_inventory(value)


def test_public_fixture_is_exact_and_never_has_population_sample_size():
    public = inventory_api.public_fixture_inventory()
    counts = inventory_api.inventory_counts(public)
    assert counts == {"declared_base_identities": 1, "declared_members": 1,
        "cases": 36, "logical_exposures": 276, "expected_photons": 3036000,
        "independence_verified": False}
    public["bases"][0]["base_wavefront_id"] = "synthetic_relabelled"
    with pytest.raises(ValueError, match="fixed public"):
        inventory_api.case_inventory(public)
    with pytest.raises(ValueError):
        inventory_api.validate_inventory(canonical_bytes(synthetic_inventory()),
            required_mode=inventory_api.DEVELOPMENT, cohort_id="synthetic_contract_only")


def test_terminal_presence_does_not_turn_failed_evidence_into_a_score():
    value = inventory_api.public_fixture_inventory()
    cases = inventory_api.case_inventory(value)
    terminals = [{"case_id": row["id"], "status": "completed"} for row in cases]
    terminals[0]["status"] = "failed"
    result = inventory_api.terminal_inventory(value, terminals)
    assert result["all_terminal_identities_present"]
    assert not result["all_scientific_outcomes_available"]
    assert result["failed_case_ids"] == [cases[0]["id"]]
    assert result["artifact_verification"] == "not_performed_by_metadata_validator"
    with pytest.raises(ValueError, match="duplicate"):
        inventory_api.terminal_inventory(value, terminals[:-1]+[terminals[0]])
