"""Pure in-memory checks for a fixed, paired wavefront comparison."""
from copy import deepcopy
from fractions import Fraction
from itertools import permutations
import math
import sys

import numpy as np
import pytest

from probebench.cohorts import stream_seed
import probebench.paired_statistics as paired_statistics
from probebench.paired_statistics import linear_quantile, paired_curve, signed_mean
from probebench.qualification import canonical_hash


def records(rows, *, track="classical", methods=("reference", "target")):
    """Rows are (cell, stratum, base, member, replicate, paired losses)."""
    expected, outcomes = [], []
    for index, (cell, stratum, base, member, replicate, losses) in enumerate(rows):
        for method_index, method in enumerate(methods):
            identity = f"case_{index:04d}_{method_index}"
            expected.append({
                "id": identity, "method": method, "track": track,
                "stratum_id": stratum, "base_wavefront_id": base,
                "member": member, "replicate": replicate,
                "photons_per_exposure": cell[0], "exposure_budget": cell[1],
            })
            outcomes.append({
                "case_id": identity, "residual_wfe_nm": float(losses[method_index]),
                "outcome_status": "completed", "failure_reason": None,
            })
    return expected, outcomes


def analyze(expected, outcomes, **overrides):
    options = {"reference_method": "reference", "target_method": "target",
               "track": "classical", "seed_hex": "a1" * 32,
               "resamples": 128, "confidence_level": .95}
    options.update(overrides)
    return paired_curve(expected, outcomes, **options)


def simple_records(*, cells=((2000, 3),), track="classical"):
    return records([
        (cell, "low", base, "primary", 0, losses)
        for cell in cells
        for base, losses in (("a", [1., 2.]), ("b", [3., 7.]))
    ], track=track)


def differences(result):
    return np.asarray(result["bootstrap"]["difference_curves_nm"])


def test_hierarchical_weights_with_unbalanced_strata_bases_members_and_replicates():
    expected, outcomes = records([
        ((2000, 3), "low", "low_a", "primary", 0, [10, 10]),
        ((2000, 3), "low", "low_a", "primary", 1, [10, 14]),
        ((2000, 3), "low", "low_a", "conjugate", 0, [10, 20]),
        ((2000, 3), "low", "low_b", "primary", 0, [10, 12]),
        ((2000, 3), "low", "low_c", "primary", 0, [10, 14]),
        ((2000, 3), "high", "high_a", "primary", 0, [10, 30]),
        ((2000, 3), "high", "high_b", "primary", 0, [10, 10]),
    ])
    result = analyze(expected, outcomes)
    cell = result["cells"][0]
    # low_a = (mean(0,4)+10)/2 = 6; low = mean(6,2,4) = 4;
    # high = mean(20,0) = 10; equal-stratum difference = mean(4,10) = 7.
    assert cell["reference_mean_wfe_nm"] == 10
    assert cell["target_mean_wfe_nm"] == 17
    assert cell["mean_difference_nm"] == 7
    assert cell["mean_difference_nm"] != pytest.approx(40 / 7)
    assert cell["case_count_per_method"] == 7
    assert result["independent_base_count"] == 5
    assert result["stratum_base_counts"] == {"high": 2, "low": 3}
    strata = {row["stratum_id"]: row for row in cell["strata"]}
    assert strata["low"]["mean_difference_nm"] == 4
    assert strata["high"]["mean_difference_nm"] == 10
    bases = {row["base_wavefront_id"]: row for row in strata["low"]["bases"]}
    assert [bases[name]["mean_difference_nm"] for name in sorted(bases)] == [6, 2, 4]
    assert bases["low_a"]["case_count_per_method"] == 3
    assert bases["low_b"]["case_count_per_method"] == 1
    assert all(row["reference_failure_count"] == row["target_failure_count"] == 0
               for row in (cell, *strata.values(), *bases.values()))


def correlated_records():
    return records([
        (cell, stratum, base, "primary", 0, [100, 100 + factor * effect])
        for cell, factor in (((20000, 7), -1), ((2000, 7), 2), ((2000, 3), 1))
        for stratum, effects in (("low", [-3, 0, 3]), ("high", [2, 6]))
        for index, effect in enumerate(effects)
        for base in [f"{stratum}_{index}"]
    ])


def test_every_bootstrap_draw_preserves_shared_base_indices_across_the_curve():
    expected, outcomes = correlated_records()
    result = analyze(expected, outcomes, resamples=257)
    assert [(row["photons_per_exposure"], row["exposure_budget"])
            for row in result["cells"]] == [(2000, 3), (2000, 7), (20000, 7)]
    draws = differences(result)
    assert draws.shape == (257, 3)
    np.testing.assert_array_equal(draws[:, 1], 2 * draws[:, 0])
    np.testing.assert_array_equal(draws[:, 2], -draws[:, 0])
    np.testing.assert_array_equal(draws[:128], differences(analyze(expected, outcomes)))
    assert result["criterion"] == "equal_stratum_base_member_replicate_paired_wfe_v1"
    assert result["interval_scope"] == "pointwise"
    assert result["bootstrap"]["seed_hex"] == "a1" * 32
    assert result["bootstrap"]["resamples"] == 257
    assert result["bootstrap"]["bit_generator"] == "PCG64"


def test_strata_use_independent_streams_and_equal_stratum_weights():
    expected, outcomes = records([
        ((2000, 3), stratum, f"{stratum}_{index}", "primary", 0, [0, effect])
        for stratum, effects in (("low", [0, 2]), ("high", [0, 200]))
        for index, effect in enumerate(effects)
    ])
    draws = differences(analyze(expected, outcomes, resamples=1024))[:, 0]
    # Independent two-base resamples have all nine combinations; reusing the
    # same indices in the two strata would produce just three combinations.
    assert set(np.round(draws, 8)) == {0, .5, 1, 50, 50.5, 51, 100, 100.5, 101}


def test_pointwise_percentile_endpoints_use_linear_interpolation():
    expected, outcomes = correlated_records()
    result = analyze(expected, outcomes, resamples=19, confidence_level=.73)
    draws = differences(result)
    for index, cell in enumerate(result["cells"]):
        ordered = sorted(draws[:, index])
        endpoints = []
        for quantile in (.135, .865):
            position = (len(ordered) - 1) * quantile
            lower = math.floor(position)
            fraction = position - lower
            endpoints.append(ordered[lower] * (1 - fraction)
                             + ordered[lower + 1] * fraction)
        assert cell["confidence_interval_nm"] == pytest.approx(endpoints)
        assert cell["interval_degenerate"] is bool(endpoints[0] == endpoints[1])


def test_constant_paired_shift_has_zero_width_interval_despite_common_mode_variation():
    expected, outcomes = records([
        (cell, "low", base, member, replicate, [baseline, baseline + 3])
        for cell in ((2000, 3), (20000, 7))
        for base, baseline in (("a", 100), ("b", 10000))
        for member in ("primary", "conjugate")
        for replicate in (0, 1)
    ])
    result = analyze(expected, outcomes)
    np.testing.assert_array_equal(differences(result), np.full((128, 2), 3.))
    for cell in result["cells"]:
        assert cell["mean_difference_nm"] == 3
        assert cell["confidence_interval_nm"] == [3, 3]
        assert cell["interval_degenerate"] is True


def test_paired_common_mode_offsets_cancel_before_aggregation_and_bootstrap():
    expected, outcomes = records([
        ((2000, 3), "low", base, "primary", replicate, [10, 10 + effect])
        for base, effects in (("a", [-2, 2]), ("b", [5, 7]))
        for replicate, effect in enumerate(effects)
    ])
    original = analyze(expected, outcomes)
    shifted = deepcopy(outcomes)
    for index, outcome in enumerate(shifted):
        outcome["residual_wfe_nm"] += 1000 * (index // 2 + 1)
    result = analyze(expected, shifted)
    np.testing.assert_array_equal(differences(result), differences(original))
    for field in ("mean_difference_nm", "confidence_interval_nm", "interval_degenerate"):
        assert result["cells"][0][field] == original["cells"][0][field]
    assert result["cells"][0]["reference_mean_wfe_nm"] > original["cells"][0]["reference_mean_wfe_nm"]


def test_finite_failure_fallbacks_remain_in_paired_means_counts_and_provenance():
    expected, outcomes = simple_records()
    outcomes[1].update(residual_wfe_nm=100., outcome_status="policy_failure",
                       failure_reason="target failed; protocol fallback")
    outcomes[2].update(residual_wfe_nm=30., outcome_status="policy_failure",
                       failure_reason="reference failed; protocol fallback")
    result = analyze(expected, outcomes)
    cell = result["cells"][0]
    assert cell["reference_mean_wfe_nm"] == pytest.approx(15.5)
    assert cell["target_mean_wfe_nm"] == pytest.approx(53.5)
    assert cell["mean_difference_nm"] == pytest.approx(38)
    assert cell["reference_failure_count"] == cell["target_failure_count"] == 1
    assert cell["case_count_per_method"] == 2
    assert result["paired_outcomes"][0]["target"] == outcomes[1]
    assert result["paired_outcomes"][1]["reference"] == outcomes[2]
    assert len(result["paired_outcomes"]) == 2


def test_canonical_inputs_pair_provenance_semantic_order_and_immutability():
    expected, outcomes = correlated_records()
    before = deepcopy((expected, outcomes))
    result = analyze(expected, outcomes)
    assert analyze(list(reversed(expected)), list(reversed(outcomes))) == result
    assert (expected, outcomes) == before
    assert result["input_sha256"] == canonical_hash({
        "expected_cases": sorted(expected, key=lambda row: row["id"]),
        "outcomes": sorted(outcomes, key=lambda row: row["case_id"]),
    })
    rows = result["paired_outcomes"]
    order = ("stratum_id", "base_wavefront_id", "photons_per_exposure",
             "exposure_budget", "member", "replicate")
    assert [tuple(row[key] for key in order) for row in rows] == sorted(
        tuple(row[key] for key in order) for row in rows)
    cases_by_id = {row["id"]: row for row in expected}
    outcomes_by_id = {row["case_id"]: row for row in outcomes}
    for row in rows:
        assert set(row) == {*order, "reference", "target", "difference_nm"}
        for arm in ("reference", "target"):
            identity = row[arm]["case_id"]
            assert row[arm] == outcomes_by_id[identity]
            assert cases_by_id[identity]["method"] == arm
            assert all(cases_by_id[identity][key] == row[key] for key in order)
        assert row["difference_nm"] == row["target"]["residual_wfe_nm"] - row["reference"]["residual_wfe_nm"]
    other_seed = analyze(expected, outcomes, seed_hex="b2" * 32)
    assert result["input_sha256"] == other_seed["input_sha256"]
    assert not np.array_equal(differences(result), differences(other_seed))


@pytest.mark.parametrize("track", ["classical", "direct_action", "tool_assisted"])
def test_analysis_track_is_explicit_and_shared_by_both_methods(track):
    expected, outcomes = simple_records(track=track)
    result = analyze(expected, outcomes, track=track)
    assert result["track"] == track
    assert result["reference_method"] == "reference"
    assert result["target_method"] == "target"
    assert result["confidence_level"] == .95


def test_swapping_method_names_reverses_every_paired_difference_and_interval():
    expected, outcomes = simple_records()
    forward = analyze(expected, outcomes)
    reverse = analyze(expected, outcomes, reference_method="target", target_method="reference")
    np.testing.assert_array_equal(differences(reverse), -differences(forward))
    cell, reversed_cell = forward["cells"][0], reverse["cells"][0]
    assert reversed_cell["mean_difference_nm"] == -cell["mean_difference_nm"]
    assert reversed_cell["confidence_interval_nm"] == pytest.approx(
        [-endpoint for endpoint in reversed(cell["confidence_interval_nm"])])
    assert reversed_cell["reference_mean_wfe_nm"] == cell["target_mean_wfe_nm"]


@pytest.mark.parametrize("replicates,members", [(8, ("primary",)), (1, ("primary", "conjugate")),
                                                (8, ("primary", "conjugate"))])
def test_duplicate_observations_do_not_inflate_independent_sample_count(replicates, members):
    original = analyze(*simple_records())
    expected, outcomes = records([
        ((2000, 3), "low", base, member, replicate, losses)
        for base, losses in (("a", [1, 2]), ("b", [3, 7]))
        for member in members for replicate in range(replicates)
    ])
    result = analyze(expected, outcomes)
    assert result["independent_base_count"] == 2
    assert result["stratum_base_counts"] == {"low": 2}
    assert result["cells"][0]["case_count_per_method"] == 2 * replicates * len(members)
    np.testing.assert_array_equal(differences(result), differences(original))
    assert result["cells"][0]["confidence_interval_nm"] == original["cells"][0]["confidence_interval_nm"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_outcome_inventory_must_match_exactly(change):
    expected, outcomes = simple_records()
    if change == "missing":
        outcomes.pop()
    elif change == "duplicate":
        outcomes.append(deepcopy(outcomes[0]))
    else:
        outcomes[-1]["case_id"] = "unknown"
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


@pytest.mark.parametrize("semantic", [False, True])
def test_duplicate_expected_frames_reject_even_with_different_case_ids(semantic):
    expected, outcomes = simple_records()
    case, outcome = deepcopy(expected[0]), deepcopy(outcomes[0])
    if semantic:
        case["id"] = outcome["case_id"] = "different_id_same_frame"
    with pytest.raises(ValueError):
        analyze(expected + [case], outcomes + [outcome])


@pytest.mark.parametrize("field,value", [
    ("replicate", 2), ("member", "conjugate"), ("base_wavefront_id", "other"),
    ("photons_per_exposure", 20000), ("exposure_budget", 7),
])
def test_both_methods_must_match_every_pairing_dimension(field, value):
    expected, outcomes = simple_records()
    expected[0][field] = value
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


@pytest.mark.parametrize("field,value", [
    ("replicate", 2), ("member", "conjugate"), ("base_wavefront_id", "other"),
])
def test_every_cell_requires_the_same_complete_grid(field, value):
    expected, outcomes = simple_records(cells=((2000, 3), (20000, 7)))
    for row in expected[-2:]:
        row[field] = value
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


def test_a_missing_complete_pair_in_one_cell_is_rejected():
    expected, outcomes = simple_records(cells=((2000, 3), (20000, 7)))
    with pytest.raises(ValueError):
        analyze(expected[:-2], outcomes[:-2])


def test_base_cannot_move_between_strata():
    expected, outcomes = simple_records(cells=((2000, 3), (20000, 7)))
    for row in expected[-2:]:
        row["stratum_id"] = "high"
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


def test_every_stratum_requires_at_least_two_independent_bases():
    expected, outcomes = records([
        ((2000, 3), stratum, base, "primary", 0, [1, 2])
        for stratum, bases in (("low", ["low_a", "low_b"]), ("high", ["high_a"]))
        for base in bases
    ])
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


@pytest.mark.parametrize("update", [
    {"method": "third"}, {"track": "direct_action"}, {"member": "unknown"},
    {"replicate": -1}, {"replicate": True}, {"replicate": 1.5},
    {"photons_per_exposure": 0}, {"photons_per_exposure": float("nan")},
    {"exposure_budget": 0}, {"exposure_budget": 1.5}, {"id": " "},
    {"stratum_id": " "}, {"base_wavefront_id": None}, {"extra": "field"},
])
def test_invalid_expected_case_fields_are_rejected(update):
    expected, outcomes = simple_records()
    expected[0].update(update)
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


@pytest.mark.parametrize("remove_from,field", [("expected", "track"), ("outcomes", "failure_reason")])
def test_missing_required_fields_are_rejected(remove_from, field):
    expected, outcomes = simple_records()
    del (expected if remove_from == "expected" else outcomes)[0][field]
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1., True, "1", 10**1000])
def test_invalid_residuals_are_rejected(value):
    expected, outcomes = simple_records()
    outcomes[0]["residual_wfe_nm"] = value
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


@pytest.mark.parametrize("update", [
    {"outcome_status": "interrupted"}, {"outcome_status": None},
    {"outcome_status": "policy_failure", "failure_reason": None},
    {"outcome_status": "policy_failure", "failure_reason": " "},
    {"failure_reason": "unexpected"}, {"extra": "field"},
])
def test_partial_results_and_invalid_failure_metadata_are_rejected(update):
    expected, outcomes = simple_records()
    outcomes[0].update(update)
    with pytest.raises(ValueError):
        analyze(expected, outcomes)


@pytest.mark.parametrize("which,value", [("expected", []), ("expected", ()),
                                       ("expected", None), ("outcomes", ()),
                                       ("outcomes", None)])
def test_list_inventory_contract(which, value):
    expected, outcomes = simple_records()
    with pytest.raises(ValueError):
        analyze(value if which == "expected" else expected,
                value if which == "outcomes" else outcomes)


@pytest.mark.parametrize("options", [
    {"reference_method": "target"}, {"reference_method": "missing"},
    {"target_method": " "}, {"track": "unknown"}, {"track": None},
    {"seed_hex": "a1" * 31}, {"seed_hex": "a1" * 33}, {"seed_hex": "g1" * 32},
    {"seed_hex": None}, {"resamples": 1}, {"resamples": 100001},
    {"resamples": True}, {"resamples": 2.5}, {"confidence_level": 0},
    {"confidence_level": 1}, {"confidence_level": -1},
    {"confidence_level": float("nan")}, {"confidence_level": float("inf")},
    {"confidence_level": True}, {"confidence_level": "0.95"},
])
def test_analysis_options_are_validated(options):
    with pytest.raises(ValueError):
        analyze(*simple_records(), **options)


def test_bootstrap_matrix_resource_bound_is_checked_before_sampling(monkeypatch):
    expected, outcomes = simple_records(cells=tuple((2000, index + 1) for index in range(11)))
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized bootstrap matrix reached reduction or sampling")

    monkeypatch.setattr(paired_statistics, "_reduce", forbidden)
    monkeypatch.setattr(paired_statistics, "_bootstrap", forbidden)
    with pytest.raises(ValueError, match="bounded value count"):
        analyze(expected, outcomes, resamples=100000)


@pytest.mark.parametrize("score", [0., 1e308, sys.float_info.max, math.ulp(0.)])
def test_extreme_finite_constant_scores_preserve_means_and_zero_difference(score):
    expected, outcomes = records([
        ((2000, 3), "low", f"base_{index}", "primary", 0, [score, score])
        for index in range(3)
    ])
    result = analyze(expected, outcomes)
    cell = result["cells"][0]
    assert cell["reference_mean_wfe_nm"] == cell["target_mean_wfe_nm"] == score
    assert cell["mean_difference_nm"] == 0
    assert cell["confidence_interval_nm"] == [0, 0]
    assert cell["interval_degenerate"] is True
    assert np.isfinite(differences(result)).all()


def test_percentile_interpolation_between_opposite_extremes_does_not_overflow():
    extreme = sys.float_info.max
    seed_hex = f"{1:064x}"
    seed = stream_seed(seed_hex, "bootstrap", {
        "protocol": "paired_wavefront_curve_analysis_v1",
        "stratum_id": "low", "base_wavefront_ids": ["a", "b"],
    })
    # This seed samples the negative extreme twice, then the positive extreme
    # twice. The interval must interpolate between these opposing finite draws.
    rng = np.random.Generator(np.random.PCG64(seed))
    assert rng.integers(0, 2, size=(2, 2)).tolist() == [[0, 0], [1, 1]]
    expected, outcomes = records([
        ((2000, 3), "low", "a", "primary", 0, [extreme, 0]),
        ((2000, 3), "low", "b", "primary", 0, [0, extreme]),
    ])
    with np.errstate(over="raise", invalid="raise"):
        result = analyze(expected, outcomes, seed_hex=seed_hex,
                         resamples=2, confidence_level=.5)
    cell = result["cells"][0]
    assert cell["reference_mean_wfe_nm"] == cell["target_mean_wfe_nm"] == extreme / 2
    assert cell["mean_difference_nm"] == 0
    assert differences(result).tolist() == [[-extreme], [extreme]]
    assert cell["confidence_interval_nm"] == [-extreme / 2, extreme / 2]
    assert cell["interval_degenerate"] is False


@pytest.mark.parametrize("values,expected", [
    ([1e308, -1e308, 1e-100], 1e-100 / 3),
    ([2., -2., 3 * math.ulp(0.)], math.ulp(0.)),
])
def test_compensated_mean_preserves_small_cancellation_remainders_in_every_order(values, expected):
    assert expected > 0
    for ordered in permutations(values):
        assert signed_mean(ordered) == expected


@pytest.mark.parametrize("remainder", [1e-100, 5 * math.ulp(0.)])
def test_overflow_fallback_preserves_cancellation_remainder(remainder):
    extreme = sys.float_info.max
    values = [extreme, extreme, -extreme, -extreme, remainder]
    expected = float(Fraction(remainder) / 5)
    # Some permutations overflow math.fsum's intermediate sum; others do not.
    # Both paths must retain the same representable remainder.
    with pytest.raises(OverflowError):
        math.fsum(values)
    for ordered in set(permutations(values)):
        assert signed_mean(ordered) == expected


def losses_for_difference(difference):
    """Represent any finite signed difference using nonnegative residuals."""
    return [max(0., -difference), max(0., difference)]


@pytest.mark.parametrize("large,remainder", [(1e308, 1e-100), (2., 3 * math.ulp(0.))])
@pytest.mark.parametrize("level", ["replicate", "stratum"])
def test_hierarchical_cancellation_keeps_tiny_point_and_bootstrap_effects(level, large, remainder):
    effects = [large, -large, remainder]
    rows = []
    for cell, sign in (((2000, 3), 1), ((20000, 7), -1)):
        for index, effect in enumerate(effects):
            for base_index in range(2):
                stratum = "low" if level == "replicate" else f"stratum_{index}"
                base = f"{stratum}_{base_index}"
                replicate = index if level == "replicate" else 0
                rows.append((cell, stratum, base, "primary", replicate,
                             losses_for_difference(sign * effect)))
    result = analyze(*records(rows), resamples=17)
    exact_mean = float(Fraction(remainder) / 3)
    assert exact_mean > 0
    assert [cell["mean_difference_nm"] for cell in result["cells"]] == [exact_mean, -exact_mean]
    assert differences(result).tolist() == [[exact_mean, -exact_mean]] * 17
    for cell, value in zip(result["cells"], (exact_mean, -exact_mean)):
        assert cell["confidence_interval_nm"] == [value, value]
        assert cell["interval_degenerate"] is True


@pytest.mark.parametrize("large,remainder", [(1e308, 1e-100), (2., 3 * math.ulp(0.))])
def test_base_cluster_bootstrap_preserves_cancellation_against_exact_oracle(large, remainder):
    effects = [large, -large, remainder]
    base_ids = ["a", "b", "c"]
    seed_hex = "a1" * 32
    resamples = 128
    expected, outcomes = records([
        (cell, "low", base, "primary", 0, losses_for_difference(sign * effect))
        for cell, sign in (((2000, 3), 1), ((20000, 7), -1))
        for base, effect in zip(base_ids, effects)
    ])
    result = analyze(expected, outcomes, seed_hex=seed_hex, resamples=resamples)
    point = float(Fraction(remainder) / 3)
    assert result["cells"][0]["mean_difference_nm"] == point
    seed = stream_seed(seed_hex, "bootstrap", {
        "protocol": "paired_wavefront_curve_analysis_v1",
        "stratum_id": "low", "base_wavefront_ids": base_ids,
    })
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, 3, size=(resamples, 3))
    oracle = [float(sum((Fraction(effects[index]) for index in sample), Fraction()) / 3)
              for sample in indices]
    assert differences(result).tolist() == [[value, -value] for value in oracle]
    cancellation_draws = [index for index, sample in enumerate(indices) if set(sample) == {0, 1, 2}]
    assert cancellation_draws
    assert all(differences(result)[index, 0] == point > 0 for index in cancellation_draws)


@pytest.mark.parametrize("endpoints,probability,expected", [
    ([math.ulp(0.), math.ulp(0.)], .5, math.ulp(0.)),
    ([math.ulp(0.), 2 * math.ulp(0.)], .5, 2 * math.ulp(0.)),
    ([math.ulp(0.), 3 * math.ulp(0.)], .25, 2 * math.ulp(0.)),
    ([math.ulp(0.), 3 * math.ulp(0.)], .75, 2 * math.ulp(0.)),
    ([-math.ulp(0.), math.ulp(0.)], .5, 0.),
    ([math.ulp(0.), sys.float_info.max], 0., math.ulp(0.)),
    ([-sys.float_info.max, -math.ulp(0.)], 1., -math.ulp(0.)),
    ([-sys.float_info.max, sys.float_info.max], 0., -sys.float_info.max),
    ([-sys.float_info.max, sys.float_info.max], .25, -sys.float_info.max / 2),
    ([-sys.float_info.max, sys.float_info.max], .5, 0.),
    ([-sys.float_info.max, sys.float_info.max], .75, sys.float_info.max / 2),
    ([-sys.float_info.max, sys.float_info.max], 1., sys.float_info.max),
])
def test_linear_quantile_keeps_finite_endpoints_and_rounds_subnormals_once(endpoints, probability, expected):
    assert linear_quantile(endpoints, probability) == expected


@pytest.mark.parametrize("multipliers,interval,degenerate", [
    ((1, 1), (1, 1), True),
    ((1, 3), (2, 2), True),
    ((0, 1), (0, 1), False),
    ((-1, 1), (0, 0), True),
])
def test_subnormal_curve_intervals_use_endpoint_equality_for_degeneracy(multipliers, interval, degenerate):
    unit = math.ulp(0.)
    expected, outcomes = records([
        ((2000, 3), "low", base, "primary", 0, losses_for_difference(multiplier * unit))
        for base, multiplier in zip(("a", "b"), multipliers)
    ])
    result = analyze(expected, outcomes, seed_hex=f"{1:064x}", resamples=2, confidence_level=.5)
    assert differences(result).tolist() == [[value * unit] for value in multipliers]
    cell = result["cells"][0]
    assert cell["confidence_interval_nm"] == [value * unit for value in interval]
    assert cell["interval_degenerate"] is degenerate


def test_resampled_work_cap_counts_all_strata_and_precedes_reduction_or_sampling(monkeypatch):
    assert paired_statistics.MAX_RESAMPLED_VALUES == 100_000_000
    # The output has only 100,000 values, but 1,001 independent bases make
    # sampling exceed the separate 100M work bound. Neither stratum alone does.
    expected, outcomes = records([
        ((2000, 3), stratum, f"{stratum}_{index:04d}", "primary", 0, [1, 2])
        for stratum, count in (("low", 500), ("high", 501))
        for index in range(count)
    ])
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized resampling work reached reduction or sampling")

    monkeypatch.setattr(paired_statistics, "_reduce", forbidden)
    monkeypatch.setattr(paired_statistics, "_bootstrap", forbidden)
    with pytest.raises(ValueError, match="bounded resampled value count"):
        analyze(expected, outcomes, resamples=100000)


def test_resampled_work_bound_is_inclusive_and_uses_bases_cells_and_draws(monkeypatch):
    expected, outcomes = records([
        (cell, "low", base, "primary", replicate, losses)
        for cell in ((2000, 3), (20000, 7))
        for base, losses in (("a", [1, 2]), ("b", [3, 7]))
        for replicate in range(8)
    ])
    # 2 bases * 2 cells * 2 draws = 8 resampled values; repeated exposures
    # remain inside each base and do not multiply independent sampling work.
    monkeypatch.setattr(paired_statistics, "MAX_RESAMPLED_VALUES", 8)
    assert differences(analyze(expected, outcomes, resamples=2)).shape == (2, 2)
    monkeypatch.setattr(paired_statistics, "MAX_RESAMPLED_VALUES", 7)
    with pytest.raises(ValueError, match="bounded resampled value count"):
        analyze(expected, outcomes, resamples=2)


@pytest.mark.parametrize("batch", [1, 7, 255, 256])
def test_index_batch_boundaries_preserve_entire_curves_and_resample_prefixes(monkeypatch, batch):
    expected, outcomes = correlated_records()
    reference = analyze(expected, outcomes, resamples=257)
    monkeypatch.setattr(paired_statistics, "BOOTSTRAP_WORK_VALUES", batch * 5)
    assert analyze(expected, outcomes, resamples=257) == reference
    np.testing.assert_array_equal(differences(analyze(expected, outcomes, resamples=19)),
                                  differences(reference)[:19])


def test_adding_curve_cells_does_not_change_existing_draws():
    expected, outcomes = correlated_records()
    full_curve = analyze(expected, outcomes, resamples=257)
    one_cell = [row for row in expected
                if (row["photons_per_exposure"], row["exposure_budget"]) == (2000, 3)]
    case_ids = {row["id"] for row in one_cell}
    one_outcomes = [row for row in outcomes if row["case_id"] in case_ids]
    result = analyze(one_cell, one_outcomes, resamples=257)
    np.testing.assert_array_equal(differences(result)[:, 0], differences(full_curve)[:, 0])
