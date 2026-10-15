from copy import deepcopy
import math
import sys

import pytest

from probebench.selection_statistics import select_candidates


def records(rows, candidates=(("classical", "a"), ("classical", "b"))):
    """Rows hold (cell, stratum, base, member, replicate, candidate losses)."""
    expected, outcomes = [], []
    for index, (cell, stratum, base, member, replicate, losses) in enumerate(rows):
        for candidate_index, (family, candidate) in enumerate(candidates):
            identity = f"case_{index}_{candidate_index}"
            expected.append({
                "id": identity, "family": family, "candidate_id": candidate,
                "base_wavefront_id": base, "stratum_id": stratum, "member": member,
                "replicate": replicate, "photons_per_exposure": cell[0],
                "exposure_budget": cell[1],
            })
            outcomes.append({"case_id": identity, "residual_wfe_nm": losses[candidate_index],
                             "outcome_status": "completed", "failure_reason": None})
    return expected, outcomes


def simple_records():
    return records([((2000, 3), "low", "base_1", "primary", 0, [1., 2.]),
                    ((2000, 3), "low", "base_2", "primary", 0, [3., 4.])])


def test_equal_cells_strata_and_bases_override_pooled_trial_average():
    expected, outcomes = records([
        ((2000, 3), "low", "low_1", "primary", 0, [0., 8.]),
        ((2000, 3), "low", "low_2", "primary", 0, [0., 8.]),
        ((2000, 3), "high", "high_1", "primary", 0, [20., 8.]),
        ((20000, 3), "low", "low_1", "primary", 0, [2., 5.]),
        ((20000, 3), "high", "high_1", "primary", 0, [6., 5.]),
    ])
    result = select_candidates(expected, outcomes)
    a, b = result["candidates"]
    # A: cell 1 = (mean(0, 0) + 20)/2 = 10; cell 2 = (2+6)/2 = 4.
    # B: cell 1 = 8; cell 2 = 5. Pooled rows would incorrectly prefer A.
    assert a["loss_nm"] == 7
    assert b["loss_nm"] == 6.5
    assert result["selected_candidates"] == {"classical": "b"}
    assert [row["loss_nm"] for row in a["cells"]] == [10, 4]
    strata = {row["stratum_id"]: row for row in a["cells"][0]["strata"]}
    assert strata["low"]["loss_nm"] == 0
    assert len(strata["low"]["bases"]) == 2
    assert a["case_count"] == 5 and a["failure_count"] == 0


def test_singleton_base_and_unequal_member_replicates_keep_hierarchical_weights():
    expected, outcomes = records([
        ((2000, 3), "low", "paired", "primary", 0, [0., 5.]),
        ((2000, 3), "low", "paired", "primary", 1, [4., 5.]),
        ((2000, 3), "low", "paired", "conjugate", 0, [10., 5.]),
        ((2000, 3), "low", "singleton", "primary", 0, [2., 5.]),
    ])
    result = select_candidates(expected, outcomes)
    a = result["candidates"][0]
    # Paired base = (mean(0,4)+10)/2 = 6; singleton = 2; base mean = 4.
    assert a["loss_nm"] == 4
    bases = {row["base_wavefront_id"]: row for row in a["cells"][0]["strata"][0]["bases"]}
    assert bases["paired"]["loss_nm"] == 6
    assert bases["singleton"]["loss_nm"] == 2
    assert len(bases["singleton"]["members"]) == 1


def test_policy_failures_retain_finite_fallback_loss_reason_and_all_counts():
    expected, outcomes = simple_records()
    outcomes[0].update(outcome_status="policy_failure", failure_reason="numerical failure",
                       residual_wfe_nm=100.)
    result = select_candidates(expected, outcomes)
    a = result["candidates"][0]
    assert a["loss_nm"] == 51.5
    assert a["case_count"] == 2 and a["failure_count"] == 1
    cell = a["cells"][0]
    stratum = cell["strata"][0]
    base = stratum["bases"][0]
    member = base["members"][0]
    assert all(row["failure_count"] == 1 for row in (cell, stratum, base, member))
    assert member["replicates"][0] == {"replicate": 0, **outcomes[0]}
    assert result["selected_candidates"] == {"classical": "b"}


def test_failures_are_counted_without_disqualifying_a_candidate():
    expected, outcomes = simple_records()
    for row in outcomes:
        if row["case_id"].endswith("_0"):
            row.update(outcome_status="policy_failure", failure_reason="fallback applied")
    result = select_candidates(expected, outcomes)
    assert result["selected_candidates"] == {"classical": "a"}
    assert result["candidates"][0]["failure_count"] == 2


def test_global_selection_is_separate_per_family_and_ties_are_exact():
    expected, outcomes = records([
        ((2000, 3), "low", "base", "primary", 0,
         [1., 1., 1., math.nextafter(1., 0.)]),
    ], candidates=(("first", "z"), ("first", "a"), ("second", "a"), ("second", "z")))
    result = select_candidates(expected, outcomes)
    assert result["selected_candidates"] == {"first": "a", "second": "z"}
    assert len(result["candidates"]) == 4


def test_ordering_is_stable_and_inputs_are_not_mutated():
    expected, outcomes = simple_records()
    before = deepcopy((expected, outcomes))
    reference = select_candidates(expected, outcomes)
    assert select_candidates(list(reversed(expected)), list(reversed(outcomes))) == reference
    assert (expected, outcomes) == before


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra"])
def test_outcome_inventory_must_be_exact(change):
    expected, outcomes = simple_records()
    if change == "missing":
        outcomes.pop()
    elif change == "duplicate":
        outcomes.append(deepcopy(outcomes[0]))
    else:
        outcomes.append({**outcomes[0], "case_id": "unknown"})
    with pytest.raises(ValueError, match="[Ii]nventory|Duplicate outcome"):
        select_candidates(expected, outcomes)


def test_duplicate_expected_ids_and_duplicate_semantic_cases_are_rejected():
    expected, outcomes = simple_records()
    with pytest.raises(ValueError, match="Duplicate expected case ID"):
        select_candidates(expected + [deepcopy(expected[0])], outcomes)
    duplicate = {**expected[0], "id": "different_id"}
    with pytest.raises(ValueError, match="Duplicate semantic case"):
        select_candidates(expected + [duplicate], outcomes + [{**outcomes[0], "case_id": "different_id"}])


@pytest.mark.parametrize("field,value", [
    ("replicate", 2), ("member", "conjugate"), ("base_wavefront_id", "other_base"),
    ("photons_per_exposure", 20000), ("exposure_budget", 7),
])
def test_candidate_coverage_includes_every_pairing_dimension(field, value):
    expected, outcomes = simple_records()
    expected[0][field] = value
    with pytest.raises(ValueError, match="identical semantic case coverage"):
        select_candidates(expected, outcomes)


def test_candidate_coverage_applies_across_families():
    expected, outcomes = simple_records()
    for row in expected:
        if row["candidate_id"] == "b":
            row["family"] = "another_family"
    expected.pop()
    outcomes.pop()
    with pytest.raises(ValueError, match="identical semantic case coverage"):
        select_candidates(expected, outcomes)


def test_base_cannot_belong_to_different_strata():
    expected, outcomes = simple_records()
    expected[0]["stratum_id"] = "high"
    with pytest.raises(ValueError, match="base-to-stratum"):
        select_candidates(expected, outcomes)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True, "1", 10**1000])
def test_invalid_residual_scores_are_rejected(value):
    expected, outcomes = simple_records()
    outcomes[0]["residual_wfe_nm"] = value
    with pytest.raises(ValueError, match="residual_wfe_nm"):
        select_candidates(expected, outcomes)


@pytest.mark.parametrize("update", [
    {"outcome_status": "interrupted"},
    {"outcome_status": "policy_failure", "failure_reason": None},
    {"outcome_status": "policy_failure", "failure_reason": " "},
    {"failure_reason": "unexpected"},
    {"extra": "field"},
])
def test_outcome_status_and_failure_reason_contract(update):
    expected, outcomes = simple_records()
    outcomes[0].update(update)
    with pytest.raises(ValueError):
        select_candidates(expected, outcomes)


@pytest.mark.parametrize("update", [
    {"replicate": -1}, {"replicate": True}, {"exposure_budget": 0},
    {"exposure_budget": 1.5}, {"photons_per_exposure": 0},
    {"photons_per_exposure": float("nan")}, {"family": " "}, {"extra": 1},
])
def test_invalid_expected_case_contract(update):
    expected, outcomes = simple_records()
    expected[0].update(update)
    with pytest.raises(ValueError):
        select_candidates(expected, outcomes)


def test_nonempty_expected_inventory_is_required():
    with pytest.raises(ValueError, match="nonempty list"):
        select_candidates([], [])


@pytest.mark.parametrize("score", [0., 1e308, sys.float_info.max, math.ulp(0.)])
def test_finite_constant_scores_survive_all_averages(score):
    expected, outcomes = records([
        ((2000, 3), "low", f"base_{index}", "primary", 0, [score, score])
        for index in range(3)
    ])
    for row in outcomes:
        row["residual_wfe_nm"] = score
    result = select_candidates(expected, outcomes)
    assert all(row["loss_nm"] == score for row in result["candidates"])


def test_numeric_photon_representations_do_not_create_distinct_cells():
    expected, outcomes = simple_records()
    expected[0]["photons_per_exposure"] = 2000.
    result = select_candidates(expected, outcomes)
    assert len(result["candidates"][0]["cells"]) == 1
