"""Synthetic complete-design analysis; no reserved wavefronts or physical results."""
from copy import deepcopy
import json

import numpy as np
import pytest

from probebench import comparison_report as report_api
from probebench import paired_statistics
from probebench.comparison_contract import TARGET
from probebench.comparison_inventory import case_inventory, public_fixture_inventory
from test_comparison_contract import inputs as contract_fixture
from test_comparison_inventory import synthetic_inventory


@pytest.fixture
def inputs():
    return contract_fixture.__wrapped__()


def result(case, delta=0.0):
    # Synthetic metrics and commitments cannot masquerade as measured evidence.
    return {"case_id": case["id"], "status": "completed", "case_manifest_sha256": "1"*64,
            "final_residual_wfe_nm": 100.0+delta, "final_strehl_peak_ratio": 0.5,
            "peak_acquired_probe_wfe_increase_nm": 1.0,
            "acquired_frame_wfe_nm": [101.0]*case["exposure_budget"],
            "final_command_has_camera_observation": False, "fresh_final_confirmation_exposure": False,
            "final_command_within_static_bounds": True, "logical_exposures": case["exposure_budget"],
            "expected_photons": case["exposure_budget"]*case["photons_per_exposure"],
            "wall_seconds": 0.0, "fit_calls": 0,
            "optimizer": {"fit_status_counts": {}, "recorded_starts": 0, "finite_starts": 0,
                "successful_starts": 0, "fits_without_start_accounting": 0, "known_iterations": 0,
                "starts_with_unknown_iterations": 0, "known_function_evaluations": 0,
                "starts_with_unknown_function_evaluations": 0},
            "simulator_replay_calls": 0, "interrupted_attempts": 0}


def analyze(inputs, inventory, rows, **kwargs):
    contract, evidence = inputs
    return report_api.analyze_complete(inventory, rows, contract_bytes=json.dumps(contract).encode(),
        **evidence, seed_hex=kwargs.get("seed_hex", "2"*64), raw_manifest_sha256="3"*64,
        custody_statement_sha256="4"*64)


def no_reducer(monkeypatch):
    monkeypatch.setattr(report_api, "paired_curve", lambda *a, **k: pytest.fail("Reducer reached"))


@pytest.mark.parametrize("omission", ["base", "member", "replicate", "method", "cell"])
def test_consistent_global_omissions_withhold_before_two_arm_reducer(inputs, monkeypatch, omission):
    inventory = synthetic_inventory()
    cases = case_inventory(inventory)
    def remove(case):
        return {"base": case["base_wavefront_id"] == "synthetic_low_00",
            "member": case["member"] == "conjugate", "replicate": case["replicate"] == 1,
            "method": case["method"] == "random_probe_v2", "cell": case["exposure_budget"] == 13}[omission]
    rows = [result(case) for case in cases if not remove(case)]
    no_reducer(monkeypatch)
    checked = analyze(inputs, inventory, rows)
    assert checked["status"] == "withheld_incomplete_or_failed_evidence"
    assert not checked["completeness"]["all_terminal_identities_present"]
    assert checked["contrasts"] == []


@pytest.mark.parametrize("corruption", ["unknown", "duplicate", "nonfinite", "penalty", "fresh_exposure", "budget"])
def test_bad_full_evidence_refuses_before_reduction(inputs, monkeypatch, corruption):
    inventory = synthetic_inventory(singleton=True)
    rows = [result(case) for case in case_inventory(inventory)]
    if corruption == "unknown":
        rows[0]["case_id"] = "undeclared_case"
    elif corruption == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif corruption == "nonfinite":
        rows[0]["final_residual_wfe_nm"] = float("nan")
    elif corruption == "penalty":
        rows[0]["status"] = "failed"  # finite WFE retained as an invented penalty is forbidden
    elif corruption == "fresh_exposure":
        rows[0]["fresh_final_confirmation_exposure"] = True
    else:
        rows[0]["logical_exposures"] -= 1
    no_reducer(monkeypatch)
    with pytest.raises(ValueError):
        analyze(inputs, inventory, rows)


def test_declared_terminal_failure_preserved_and_contrasts_withheld(inputs, monkeypatch):
    inventory = synthetic_inventory()
    rows = [result(case) for case in case_inventory(inventory)]
    rows[0] = {"case_id": rows[0]["case_id"], "status": "failed",
               "case_manifest_sha256": "5"*64, "failure_kind": "synthetic_unexpected_policy_error"}
    no_reducer(monkeypatch)
    checked = analyze(inputs, inventory, rows)
    assert checked["completeness"]["all_terminal_identities_present"]
    assert not checked["completeness"]["all_scientific_outcomes_available"]
    retained = next(row for row in checked["case_results"] if row["case_id"] == rows[0]["case_id"])
    assert retained == rows[0]


def test_real_hierarchy_with_singleton_shared_cells_and_exact_analysis_roles(inputs, monkeypatch, tmp_path):
    inventory = synthetic_inventory(singleton=True)
    rows = []
    for case in case_inventory(inventory):
        # Low stratum base00 has delta32; other31 bases have zero. Its one member
        # still has equal base weight. Across3 equally weighted strata: mean=1/3.
        delta = 32.0 if case["method"] == TARGET and case["base_wavefront_id"] == "synthetic_low_00" else 0.0
        rows.append(result(case, delta))
    calls = []
    def fixed_draws(cells, strata, seed_hex, resamples):
        # Retain actual production inventory validation/hierarchical reduction;
        # bootstrap randomness itself is independently covered by reducer tests.
        calls.append((resamples, len(cells), {k: len(v) for k, v in strata.items()}, seed_hex))
        return np.tile([cell["mean_difference_nm"] for cell in cells], (resamples,1)), {name: "0" for name in strata}
    monkeypatch.setattr(paired_statistics, "_bootstrap", fixed_draws)
    checked = analyze(inputs, inventory, rows)
    assert [call[0] for call in calls] == [50000,10000,10000,10000,10000]
    assert all(call[1:3] == (6, {"high":32,"low":32,"medium":32}) for call in calls)
    assert {call[3] for call in calls} == {"2"*64}
    primary = checked["contrasts"][0]
    assert primary["parameters"]["confidence_level"] == 1.0-0.05/6
    assert all(cell["mean_difference_nm"] == pytest.approx(1/3) for cell in primary["reduction"]["cells"])
    assert all(cell["interval_degenerate"] for cell in primary["reduction"]["cells"])
    assert primary["cell_verdicts"] == ["disadvantage"]*6
    assert checked["contract_status"] == "synthetic_fixture"
    assert not checked["execution_admitted"] and not checked["held_out_status_established"]
    assert not checked["provenance_authenticated"]
    assert len(checked["binding"]["case_result_shards"]) == 96
    # Caller input is detached; mutating it after analysis cannot alter a report.
    original_case = rows[0]["case_id"]
    retained_case = next(row for row in checked["case_results"] if row["case_id"] == original_case)
    retained_wfe = retained_case["final_residual_wfe_nm"]
    rows[0]["final_residual_wfe_nm"] += 1.0
    assert retained_case["final_residual_wfe_nm"] == retained_wfe
    rows[0]["final_residual_wfe_nm"] -= 1.0
    # Every changed output must be refused before a destination is created,
    # even when its original envelope/input hash and dimensions remain intact.
    mutations = [(retained_case, "final_residual_wfe_nm", retained_wfe+1),
                 (primary, "role", "secondary"),
                 (primary, "cell_verdicts", ["advantage"]*6),
                 (primary["reduction"], "input_sha256", "8"*64),
                 (primary["reduction"]["bootstrap"], "seed_hex", "9"*64)]
    for index, (container, key, changed) in enumerate(mutations):
        original = container[key]
        container[key] = changed
        output = tmp_path/f"changed-{index}"
        with pytest.raises(ValueError, match="output changed"):
            report_api.write_analysis(output, checked)
        assert not output.exists()
        container[key] = original
    draw = primary["reduction"]["bootstrap"]["difference_curves_nm"][0]
    original = draw[0]
    draw[0] += 1
    with pytest.raises(ValueError, match="output changed"):
        report_api.write_analysis(tmp_path/"changed-draw", checked)
    assert not (tmp_path/"changed-draw").exists()
    draw[0] = original
    report_api.write_analysis(tmp_path/"private-analysis", checked)
    retained = json.loads((tmp_path/"private-analysis"/"analysis.json").read_text())
    assert retained["analysis_binding_sha256"] == checked["analysis_binding_sha256"]
    folder = tmp_path/"private-analysis"/TARGET
    assert len(list(folder.glob("bootstrap-*.json"))) == 50
    assert len(json.loads((folder/"bootstrap-049000.json").read_text())) == 1000
    # Changing only the bootstrap root leaves cases/outcomes fixed but changes
    # the enclosing analysis binding, unlike the reducer's input-only digest.
    again = analyze(inputs, inventory, rows, seed_hex="6"*64)
    assert again["analysis_binding_sha256"] != checked["analysis_binding_sha256"]
    assert again["contrasts"][0]["reduction"]["input_sha256"] == primary["reduction"]["input_sha256"]


def test_public_fixture_never_enters_population_analysis(inputs, monkeypatch):
    no_reducer(monkeypatch)
    with pytest.raises(ValueError, match="population"):
        analyze(inputs, public_fixture_inventory(), [])


def test_changed_primary_definition_refuses_before_any_reduction(inputs, monkeypatch):
    contract, _ = inputs
    contract["analysis"]["primary"]["resamples"] = 49999
    no_reducer(monkeypatch)
    with pytest.raises(ValueError, match="analysis"):
        analyze(inputs, synthetic_inventory(), [])


def test_missing_new_runner_binding_refuses(inputs, monkeypatch):
    contract, evidence = inputs
    name = "src/probebench/comparison.py"
    contract["sources"]["files"].pop(name)
    evidence["source_files"].pop(name)
    no_reducer(monkeypatch)
    with pytest.raises(ValueError, match="runner/report"):
        analyze(inputs, synthetic_inventory(), [])


def test_cost_accounting_counts_final_cumulative_fits_once_and_keeps_unknowns():
    case = case_inventory(public_fixture_inventory())[0]
    physical = {"final": {"residual_wfe_nm": 5., "strehl_peak_ratio": .5},
        "peak_probe_wfe_increase_nm": 2., "trajectory": [{"residual_wfe_nm": 7.}]*3,
        "final_command_has_camera_observation": False}
    checked = {"evaluation": {"physical": physical},
        "terminal": {"observations_retained": 3,"expected_photons_used":6000,"wall_seconds":1.},
        "fits": [{"fit_status":"finite","starts":[{"success":False,"finite":True,"iterations":7,
                  "function_evaluations":12}]}, {"fit_status":"numerical_failure"}]}
    row = report_api.summarize_episode(case, checked, "7"*64)
    assert row["fit_calls"] == 2 and row["optimizer"]["recorded_starts"] == 1
    assert row["optimizer"]["successful_starts"] == 0
    assert row["optimizer"]["known_function_evaluations"] == 12
    assert row["optimizer"]["fits_without_start_accounting"] == 1


def test_finite_nonconvergence_keeps_scientific_outcome_separate_from_optimizer_success():
    inventory = public_fixture_inventory()
    rows = [result(case) for case in case_inventory(inventory)]
    row = rows[0]
    row["fit_calls"] = 1
    row["optimizer"].update(fit_status_counts={"finite": 1}, recorded_starts=3,
                            finite_starts=3, successful_starts=0, known_iterations=600,
                            known_function_evaluations=1200)
    _, retained, completeness = report_api.validate_results(inventory, rows)
    assert completeness["all_scientific_outcomes_available"]
    assert retained[row["case_id"]]["optimizer"]["successful_starts"] == 0
