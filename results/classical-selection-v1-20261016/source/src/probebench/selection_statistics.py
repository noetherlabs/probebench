"""Pure, complete-inventory reduction for global candidate selection."""
from __future__ import annotations

from collections import defaultdict
import math


CRITERION = "equal_cell_stratum_base_member_replicate_mean_wfe_v1"
CASE_FIELDS = frozenset({
    "id", "family", "candidate_id", "base_wavefront_id", "stratum_id", "member",
    "replicate", "photons_per_exposure", "exposure_budget",
})
OUTCOME_FIELDS = frozenset({
    "case_id", "residual_wfe_nm", "outcome_status", "failure_reason",
})


def _fields(value, expected, label):
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"Unexpected {label} fields")


def _label(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _number(value, label, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"Invalid {label}")
    return result


def _integer(value, label, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _mean(values):
    # Scaling preserves constant subnormal losses and prevents overflow when
    # several finite losses approach float64's maximum.
    scale = max(values)
    if scale == 0:
        return 0.0
    return scale * (math.fsum(value / scale for value in values) / len(values))


def _summary(children):
    return {
        "loss_nm": _mean([child["loss_nm"] for child in children]),
        "case_count": sum(child["case_count"] for child in children),
        "failure_count": sum(child["failure_count"] for child in children),
    }


def select_candidates(expected_cases: list[dict], outcomes: list[dict]) -> dict:
    """Select one globally best candidate per family from complete paired cases.

    A cell is one photon/exposure-budget pair. Loss is averaged equally over
    replicates within each member, members within each independent base, bases
    within each stratum, strata within each cell, and finally cells. A base with
    one conjugate member therefore receives the same weight as a two-member base.
    All candidates, including candidates in different families, must have exactly
    the same semantic cases. Each base must belong to one stratum globally.

    Policy failures remain in every average using their finite fallback residual
    WFE and are counted at every aggregation level. Exact equal final losses use
    lexicographic candidate-ID order. No tolerance, eligibility filter, held-out
    comparison, or statistical performance claim is applied here.

    The result contains ``selected_candidates`` (family -> candidate ID) and a
    sorted ``candidates`` list. Each candidate retains loss/counts and nested
    ``cells``, ``strata``, ``bases``, ``members`` and individual ``replicates``.
    Inputs are not mutated. This function performs no I/O or random draws.
    """
    if type(expected_cases) is not list or not expected_cases:
        raise ValueError("Expected cases must be a nonempty list")
    if type(outcomes) is not list:
        raise ValueError("Outcomes must be a list")

    cases, coverage, base_strata = {}, defaultdict(set), {}
    for value in expected_cases:
        _fields(value, CASE_FIELDS, "expected case")
        case = {key: _label(value[key], key) for key in (
            "id", "family", "candidate_id", "base_wavefront_id", "stratum_id", "member",
        )}
        case["replicate"] = _integer(value["replicate"], "replicate", 0)
        case["exposure_budget"] = _integer(value["exposure_budget"], "exposure_budget", 1)
        case["photons_per_exposure"] = _number(
            value["photons_per_exposure"], "photons_per_exposure", positive=True)
        if case["id"] in cases:
            raise ValueError(f'Duplicate expected case ID: {case["id"]}')
        base, stratum = case["base_wavefront_id"], case["stratum_id"]
        if base in base_strata and base_strata[base] != stratum:
            raise ValueError(f"Inconsistent base-to-stratum assignment: {base}")
        base_strata[base] = stratum
        candidate = (case["family"], case["candidate_id"])
        semantic = (case["photons_per_exposure"], case["exposure_budget"],
                    stratum, base, case["member"], case["replicate"])
        if semantic in coverage[candidate]:
            raise ValueError(f"Duplicate semantic case for candidate: {candidate}")
        coverage[candidate].add(semantic)
        cases[case["id"]] = case
    reference = coverage[min(coverage)]
    if any(identities != reference for identities in coverage.values()):
        raise ValueError("All candidates must have identical semantic case coverage")

    observed = {}
    for value in outcomes:
        _fields(value, OUTCOME_FIELDS, "outcome")
        identity = _label(value["case_id"], "case_id")
        if identity in observed:
            raise ValueError(f"Duplicate outcome case ID: {identity}")
        status = value["outcome_status"]
        if type(status) is not str or status not in {"completed", "policy_failure"}:
            raise ValueError("Unknown outcome status")
        reason = value["failure_reason"]
        if status == "completed" and reason is not None:
            raise ValueError("Completed outcomes must have no failure reason")
        if status == "policy_failure":
            _label(reason, "failure_reason")
        observed[identity] = {
            "case_id": identity,
            "residual_wfe_nm": _number(value["residual_wfe_nm"], "residual_wfe_nm"),
            "outcome_status": status,
            "failure_reason": reason,
        }
    if set(observed) != set(cases):
        raise ValueError("Outcome inventory must match expected cases exactly")

    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))))
    for identity, case in cases.items():
        key = (case["family"], case["candidate_id"])
        cell = (case["photons_per_exposure"], case["exposure_budget"])
        groups[key][cell][case["stratum_id"]][case["base_wavefront_id"]][
            case["member"]].append({"replicate": case["replicate"], **observed[identity]})

    candidates = []
    for (family, candidate_id), cells in sorted(groups.items()):
        cell_rows = []
        for (photons, budget), strata in sorted(cells.items()):
            stratum_rows = []
            for stratum_id, bases in sorted(strata.items()):
                base_rows = []
                for base_id, members in sorted(bases.items()):
                    member_rows = []
                    for member, replicates in sorted(members.items()):
                        replicates = sorted(replicates, key=lambda row: row["replicate"])
                        member_rows.append({
                            "member": member,
                            "loss_nm": _mean([r["residual_wfe_nm"] for r in replicates]),
                            "case_count": len(replicates),
                            "failure_count": sum(r["outcome_status"] == "policy_failure"
                                                 for r in replicates),
                            "replicates": replicates,
                        })
                    base_rows.append({"base_wavefront_id": base_id,
                                      **_summary(member_rows), "members": member_rows})
                stratum_rows.append({"stratum_id": stratum_id,
                                     **_summary(base_rows), "bases": base_rows})
            cell_rows.append({"photons_per_exposure": photons, "exposure_budget": budget,
                              **_summary(stratum_rows), "strata": stratum_rows})
        candidates.append({"family": family, "candidate_id": candidate_id,
                           **_summary(cell_rows), "cells": cell_rows})

    winners = {}
    for candidate in candidates:
        family = candidate["family"]
        score = (candidate["loss_nm"], candidate["candidate_id"])
        if family not in winners or score < winners[family]:
            winners[family] = score
    return {"schema": 1, "criterion": CRITERION,
            "selected_candidates": {family: score[1] for family, score in sorted(winners.items())},
            "candidates": candidates}
