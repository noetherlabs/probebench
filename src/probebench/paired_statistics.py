"""Complete paired WFE curves with stratified base-cluster percentile intervals."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
import platform

import numpy as np

from .cohorts import stream_seed
from .qualification import canonical_hash
from .selection_statistics import _fields, _integer, _label, _number

PURPOSE = "paired_wavefront_curve_analysis_v1"
CRITERION = "equal_stratum_base_member_replicate_paired_wfe_v1"
CASE_FIELDS = frozenset({"id", "method", "track", "stratum_id", "base_wavefront_id",
    "member", "replicate", "photons_per_exposure", "exposure_budget"})
OUTCOME_FIELDS = frozenset({"case_id", "residual_wfe_nm", "outcome_status", "failure_reason"})
MEANS = ("reference_mean_wfe_nm", "target_mean_wfe_nm", "mean_difference_nm")
MAX_PAIRED_CASES = 100_000
MAX_CELLS = 64
MAX_BOOTSTRAP_VALUES = 1_000_000
MAX_RESAMPLED_VALUES = 100_000_000
BOOTSTRAP_WORK_VALUES = 131_072


def signed_mean(values):
    """Mean finite signed values without overflowing their intermediate sum."""
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        # Scaling all values by the largest can erase a tiny but representable
        # remainder after cancellation. Exact arithmetic is needed only when the
        # compensated sum overflows; the mean itself stays within finite inputs.
        return float(sum(map(Fraction, values), Fraction()) / len(values))


def _aggregate(rows):
    return {**{key: signed_mean([row[key] for row in rows]) for key in MEANS},
        **{key: sum(row[key] for row in rows) for key in (
            "case_count_per_method", "reference_failure_count", "target_failure_count")}}


def _validate(expected_cases, outcomes, reference, target, track):
    if type(expected_cases) is not list or not 1 <= len(expected_cases) <= 2*MAX_PAIRED_CASES:
        raise ValueError("Expected cases must be a bounded nonempty list")
    if type(outcomes) is not list or len(outcomes) > 2*MAX_PAIRED_CASES:
        raise ValueError("Outcomes must be a bounded list")
    cases, pairs, base_strata, grids = {}, defaultdict(dict), {}, defaultdict(set)
    for row in expected_cases:
        _fields(row, CASE_FIELDS, "expected case")
        case = {key: _label(row[key], key) for key in (
            "id", "method", "track", "stratum_id", "base_wavefront_id", "member")}
        for key, minimum in (("replicate", 0), ("photons_per_exposure", 1), ("exposure_budget", 1)):
            case[key] = _integer(row[key], key, minimum)
        if case["track"] != track or case["method"] not in {reference, target}:
            raise ValueError("Expected cases must use the requested track and exactly the two methods")
        if case["member"] not in {"primary", "conjugate"}:
            raise ValueError("Unknown conjugate member")
        if case["id"] in cases:
            raise ValueError("Duplicate expected case ID")
        base, stratum = case["base_wavefront_id"], case["stratum_id"]
        if base in base_strata and base_strata[base] != stratum:
            raise ValueError("Inconsistent base-to-stratum assignment")
        base_strata[base] = stratum
        key = (stratum, base, case["photons_per_exposure"], case["exposure_budget"],
               case["member"], case["replicate"])
        if case["method"] in pairs[key]:
            raise ValueError("Duplicate semantic case")
        pairs[key][case["method"]] = case["id"]
        grids[key[2:4]].add((stratum, base, case["member"], case["replicate"]))
        cases[case["id"]] = case
    if len(grids) > MAX_CELLS:
        raise ValueError(f"At most {MAX_CELLS} curve cells are supported")
    if any(set(pair) != {reference, target} for pair in pairs.values()):
        raise ValueError("Every expected case requires both paired methods")
    first_grid = grids[min(grids)]
    if any(grid != first_grid for grid in grids.values()):
        raise ValueError("Every cell requires the same full base/member/replicate grid")
    memberships = defaultdict(set)
    strata = defaultdict(set)
    for stratum, base, member, _ in first_grid:
        strata[stratum].add(base)
        memberships[base].add(member)
    if any("primary" not in members for members in memberships.values()):
        raise ValueError("Each base requires its primary member")
    if any(len(bases) < 2 for bases in strata.values()):
        raise ValueError("At least two independent bases are required in each stratum")
    observed = {}
    for row in outcomes:
        _fields(row, OUTCOME_FIELDS, "outcome")
        identity = _label(row["case_id"], "case_id")
        if identity in observed:
            raise ValueError("Duplicate outcome case ID")
        status, reason = row["outcome_status"], row["failure_reason"]
        if type(status) is not str or status not in {"completed", "policy_failure"}:
            raise ValueError("Unknown outcome status")
        if status == "completed" and reason is not None:
            raise ValueError("Completed outcomes must have no failure reason")
        if status == "policy_failure":
            _label(reason, "failure_reason")
        observed[identity] = {"case_id": identity,
            "residual_wfe_nm": _number(row["residual_wfe_nm"], "residual_wfe_nm"),
            "outcome_status": status, "failure_reason": reason}
    if set(observed) != set(cases):
        raise ValueError("Outcome inventory must match expected cases exactly")
    return cases, pairs, observed, {name: sorted(bases) for name, bases in sorted(strata.items())}


def _reduce(pairs, observed, reference, target):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    paired = []
    for (stratum, base, photons, budget, member, replicate), identities in sorted(pairs.items()):
        left, right = observed[identities[reference]], observed[identities[target]]
        delta = right["residual_wfe_nm"] - left["residual_wfe_nm"]
        paired.append({"stratum_id": stratum, "base_wavefront_id": base,
            "photons_per_exposure": photons, "exposure_budget": budget,
            "member": member, "replicate": replicate,
            "reference": dict(left), "target": dict(right), "difference_nm": delta})
        groups[(photons, budget)][stratum][base][member].append({
            "reference_mean_wfe_nm": left["residual_wfe_nm"],
            "target_mean_wfe_nm": right["residual_wfe_nm"], "mean_difference_nm": delta,
            "case_count_per_method": 1,
            "reference_failure_count": int(left["outcome_status"] == "policy_failure"),
            "target_failure_count": int(right["outcome_status"] == "policy_failure")})
    cells = []
    for (photons, budget), strata in sorted(groups.items()):
        stratum_rows = []
        for stratum, bases in sorted(strata.items()):
            base_rows = []
            for base, members in sorted(bases.items()):
                member_rows = [_aggregate(rows) for _, rows in sorted(members.items())]
                base_rows.append({"base_wavefront_id": base, "member_count": len(members),
                                  **_aggregate(member_rows)})
            stratum_rows.append({"stratum_id": stratum, "base_count": len(bases),
                                **_aggregate(base_rows), "bases": base_rows})
        cells.append({"photons_per_exposure": photons, "exposure_budget": budget,
            "total_expected_photons": photons * budget,
            **_aggregate(stratum_rows), "strata": stratum_rows})
    return cells, paired


def _bootstrap(cells, strata, seed_hex, resamples):
    # Columns are entire curve cells; no resampling occurs inside a base cluster.
    matrices = {name: np.array([[cell["strata"][s]["bases"][b]["mean_difference_nm"]
        for cell in cells] for b in range(len(bases))], dtype=np.float64)
        for s, (name, bases) in enumerate(strata.items())}
    draws = np.empty((resamples, len(cells)))
    seeds, generators = {}, {}
    for name, bases in strata.items():
        seed = stream_seed(seed_hex, "bootstrap", {"protocol": PURPOSE,
            "stratum_id": name, "base_wavefront_ids": bases})
        seeds[name] = str(seed)
        generators[name] = np.random.Generator(np.random.PCG64(seed))
    # Batch only indices. Each draw uses the same compensated hierarchical mean
    # as the point estimate, without normalizing away small cancellation terms.
    batch = max(1, min(256, BOOTSTRAP_WORK_VALUES // sum(map(len, strata.values()))))
    for start in range(0, resamples, batch):
        count = min(batch, resamples-start)
        indices = {name: generators[name].integers(0, len(bases), size=(count, len(bases)))
                   for name, bases in strata.items()}
        for draw in range(count):
            stratum_means = [[signed_mean(matrices[name][indices[name][draw], column].tolist())
                              for column in range(len(cells))] for name in strata]
            draws[start+draw] = [signed_mean([row[column] for row in stratum_means])
                                 for column in range(len(cells))]
    return draws, seeds


def linear_quantile(sorted_values, probability):
    """Linear empirical quantile with safe interpolation of finite signed endpoints."""
    position = (len(sorted_values)-1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    left, right = float(sorted_values[lower]), float(sorted_values[upper])
    fraction = Fraction(position - lower)
    # Two exact products avoid both overflow of (right-left) and underflow from
    # separately rounded weighted endpoints. Convert to float only at the end.
    return float((1-fraction)*Fraction(left) + fraction*Fraction(right))


def paired_curve(expected_cases, outcomes, *, reference_method, target_method, track,
                 seed_hex, resamples, confidence_level):
    """Analyze one fixed two-method contrast; perform no selection or acquisition.

    Expected cases must come from a separately frozen, source-verified inventory.
    Case track denotes the analysis track, including its classical reference arm.
    Finite protocol-defined failure outcomes remain scored; incomplete inventories
    raise. Returned intervals are pointwise and conditional on the frozen methods.
    This reducer cannot certify independent wavefronts, held-out provenance, adequate
    sample size, simultaneous coverage, equivalence or physical-telescope validity.
    """
    reference_method = _label(reference_method, "reference_method")
    target_method = _label(target_method, "target_method")
    if reference_method == target_method:
        raise ValueError("Reference and target methods must be distinct")
    if type(track) is not str or track not in {"classical", "direct_action", "tool_assisted"}:
        raise ValueError("Unknown analysis track")
    _integer(resamples, "resamples", 2)
    if resamples > 100_000:
        raise ValueError("At most 100000 bootstrap resamples are supported")
    confidence_level = _number(confidence_level, "confidence_level", positive=True)
    if confidence_level >= 1:
        raise ValueError("Confidence level must be strictly between zero and one")
    stream_seed(seed_hex, "bootstrap", {})
    cases, pairs, observed, strata = _validate(expected_cases, outcomes,
                                               reference_method, target_method, track)
    cell_count = len({(case["photons_per_exposure"], case["exposure_budget"])
                      for case in cases.values()})
    if resamples * cell_count > MAX_BOOTSTRAP_VALUES:
        raise ValueError("Bootstrap matrix exceeds the bounded value count")
    if resamples * cell_count * sum(map(len, strata.values())) > MAX_RESAMPLED_VALUES:
        raise ValueError("Bootstrap work exceeds the bounded resampled value count")
    cells, paired = _reduce(pairs, observed, reference_method, target_method)
    draws, seeds = _bootstrap(cells, strata, seed_hex, resamples)
    tail = (1. - confidence_level) / 2.
    for index, cell in enumerate(cells):
        ordered = np.sort(draws[:, index])
        interval = [linear_quantile(ordered, probability) for probability in (tail, 1.-tail)]
        cell["confidence_interval_nm"] = interval
        cell["interval_degenerate"] = interval[0] == interval[1]
    return {"schema": 1, "purpose": PURPOSE, "criterion": CRITERION,
        "reference_method": reference_method, "target_method": target_method, "track": track,
        "contrast": "target_minus_reference", "units": "nm", "interval_scope": "pointwise",
        "interval_method": "stratified_base_cluster_percentile", "confidence_level": confidence_level,
        "quantile_method": "linear", "independent_base_count": sum(map(len, strata.values())),
        "stratum_base_counts": {name: len(bases) for name, bases in strata.items()},
        "input_sha256": canonical_hash({"expected_cases": [cases[k] for k in sorted(cases)],
                                        "outcomes": [observed[k] for k in sorted(observed)]}),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "cells": cells, "paired_outcomes": paired,
        "bootstrap": {"seed_hex": seed_hex, "resamples": resamples, "bit_generator": "PCG64",
            "stratum_seeds": seeds, "difference_curves_nm": draws.tolist()}}
