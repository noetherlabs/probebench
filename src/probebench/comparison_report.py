"""Complete-inventory analysis and retained native comparison reports.

Statistical reduction is conditional on supplied evidence. Hashes alone cannot
prove historical exclusion closure, key custody, independence or held-out status.
The public development command never computes population confidence intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path

from . import comparison_pilot as pilot
from .comparison import _seal, verify_development
from .comparison_contract import (FAMILIES, REFERENCE, TARGET, _digest, canonical_bytes,
                                  validate_comparison_contract, parse_json)
from .comparison_inventory import CELLS, PROSPECTIVE, _validated, case_inventory, terminal_inventory
from .paired_statistics import paired_curve

PRIMARY = {"reference_method": REFERENCE, "target_method": TARGET, "track": "classical",
           "resamples": 50000, "confidence_level": 1.0-0.05/6}
SECONDARY_TARGETS = sorted(set(FAMILIES)-{REFERENCE, TARGET})


def _sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _number(value):
    return (type(value) in (int, float) and (type(value) is not int or value.bit_length() <= 63)
            and math.isfinite(value) and value >= 0)


def summarize_episode(case, checked, manifest_sha256):
    """Read final cumulative fits once; do not sum overlapping fit snapshots."""
    physical, terminal, fits = (checked["evaluation"]["physical"], checked["terminal"], checked["fits"])
    statuses, success, finite, starts, iterations, evaluations = Counter(), 0, 0, 0, 0, 0
    unknown_starts, unknown_iterations, unknown_evaluations = 0, 0, 0
    for fit in fits:
        statuses[str(fit.get("fit_status", "unknown"))] += 1
        candidates = fit.get("starts")
        if type(candidates) is not list:
            unknown_starts += 1
            continue
        for start in candidates:
            starts += 1
            success += start.get("success") is True
            finite += start.get("finite") is True
            for key in ("iterations", "function_evaluations"):
                value = start.get(key)
                if type(value) is not int or value < 0:
                    if key == "iterations":
                        unknown_iterations += 1
                    else:
                        unknown_evaluations += 1
                elif key == "iterations":
                    iterations += value
                else:
                    evaluations += value
    return {"case_id": case["id"], "status": "completed", "case_manifest_sha256": manifest_sha256,
            "final_residual_wfe_nm": physical["final"]["residual_wfe_nm"],
            "final_strehl_peak_ratio": physical["final"]["strehl_peak_ratio"],
            "peak_acquired_probe_wfe_increase_nm": physical["peak_probe_wfe_increase_nm"],
            "acquired_frame_wfe_nm": [row["residual_wfe_nm"] for row in physical["trajectory"]],
            "final_command_has_camera_observation": physical["final_command_has_camera_observation"],
            "fresh_final_confirmation_exposure": False,
            "final_command_within_static_bounds": True,
            "logical_exposures": terminal["observations_retained"],
            "expected_photons": terminal["expected_photons_used"],
            "wall_seconds": terminal["wall_seconds"], "fit_calls": len(fits),
            "optimizer": {"fit_status_counts": dict(statuses), "recorded_starts": starts,
                          "finite_starts": finite, "successful_starts": success,
                          "fits_without_start_accounting": unknown_starts,
                          "known_iterations": iterations, "starts_with_unknown_iterations": unknown_iterations,
                          "known_function_evaluations": evaluations,
                          "starts_with_unknown_function_evaluations": unknown_evaluations},
            "simulator_replay_calls": 0, "interrupted_attempts": 0}


RESULT_KEYS = {"case_id", "status", "case_manifest_sha256", "final_residual_wfe_nm",
    "final_strehl_peak_ratio", "peak_acquired_probe_wfe_increase_nm", "acquired_frame_wfe_nm",
    "final_command_has_camera_observation", "fresh_final_confirmation_exposure",
    "final_command_within_static_bounds", "logical_exposures", "expected_photons", "wall_seconds",
    "fit_calls", "optimizer", "simulator_replay_calls", "interrupted_attempts"}
OPTIMIZER_KEYS = {"fit_status_counts", "recorded_starts", "finite_starts", "successful_starts",
    "fits_without_start_accounting", "known_iterations", "starts_with_unknown_iterations",
    "known_function_evaluations", "starts_with_unknown_function_evaluations"}


def validate_results(inventory, records):
    """Check the independently declared global inventory before arm filtering."""
    expected = {case["id"]: case for case in case_inventory(inventory)}
    if type(records) is not list or len(records) > len(expected):
        raise ValueError("Result inventory exceeds the declared comparison")
    terminals, rows = [], {}
    for row in records:
        if type(row) is not dict:
            raise ValueError("Expected a plain case result")
        identity, status = row.get("case_id"), row.get("status")
        terminals.append({"case_id": identity, "status": status})
        if type(identity) is not str or identity not in expected or identity in rows:
            raise ValueError("Unknown or duplicate result identity")
        case = expected[identity]
        if status == "failed":
            if set(row) != {"case_id", "status", "case_manifest_sha256", "failure_kind"}:
                raise ValueError("Failure must be retained without an invented metric")
            if type(row["failure_kind"]) is not str or not 1 <= len(row["failure_kind"]) <= 256:
                raise ValueError("Invalid retained failure category")
        elif status == "completed":
            if len(row) != len(RESULT_KEYS) or set(row) != RESULT_KEYS:
                raise ValueError("Unexpected completed-result fields")
            if any(not _number(row[key]) for key in ("final_residual_wfe_nm", "final_strehl_peak_ratio",
                    "peak_acquired_probe_wfe_increase_nm", "wall_seconds")):
                raise ValueError("Invalid finite evaluator metric or cost")
            trace = row["acquired_frame_wfe_nm"]
            if (type(trace) is not list or len(trace) != case["exposure_budget"]
                    or any(not _number(value) for value in trace)):
                raise ValueError("Incomplete acquired-frame metric trace")
            if (type(row["logical_exposures"]) is not int or row["logical_exposures"] != case["exposure_budget"]
                    or type(row["expected_photons"]) is not int
                    or row["expected_photons"] != case["exposure_budget"]*case["photons_per_exposure"]
                    or row["fresh_final_confirmation_exposure"] is not False
                    or row["final_command_within_static_bounds"] is not True
                    or type(row["final_command_has_camera_observation"]) is not bool):
                raise ValueError("Changed exposure budget or final-command semantics")
            for key in ("fit_calls", "simulator_replay_calls", "interrupted_attempts"):
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError("Invalid operational accounting")
            if row["interrupted_attempts"] or row["simulator_replay_calls"]:
                raise ValueError("This native runner cannot certify interrupted or replayed work")
            optimizer = row["optimizer"]
            if (type(optimizer) is not dict or len(optimizer) != len(OPTIMIZER_KEYS)
                    or set(optimizer) != OPTIMIZER_KEYS):
                raise ValueError("Incomplete optimizer accounting")
            for key in OPTIMIZER_KEYS-{"fit_status_counts"}:
                if type(optimizer[key]) is not int or optimizer[key] < 0:
                    raise ValueError("Invalid optimizer count")
            statuses = optimizer["fit_status_counts"]
            if (type(statuses) is not dict or len(statuses) > 3
                    or any(key not in {"finite", "numerical_failure", "unknown"}
                           or type(value) is not int or value < 0 for key, value in statuses.items())
                    or sum(statuses.values()) != row["fit_calls"]):
                raise ValueError("Fit statuses disagree with charged fit calls")
            if (optimizer["successful_starts"] > optimizer["recorded_starts"]
                    or optimizer["finite_starts"] > optimizer["recorded_starts"]
                    or optimizer["fits_without_start_accounting"] > row["fit_calls"]):
                raise ValueError("Impossible optimizer accounting")
        else:
            raise ValueError("Unknown result state")
        _digest(row["case_manifest_sha256"], "case evidence")
        # Bound each record before downstream serialization; never encode the
        # entire multi-thousand-case inventory through the contract parser.
        rows[identity] = parse_json(canonical_bytes(row))
    completeness = terminal_inventory(inventory, terminals)
    return expected, rows, completeness


def analyze_complete(inventory, records, *, contract_bytes, selected_settings_bytes,
                     source_files, exclusion_inventory_bytes, seed_hex,
                     raw_manifest_sha256, custody_statement_sha256):
    """Reduce supplied complete prospective-design evidence, without admission.

    This function does not open a dataset, establish provenance or authorize
    acquisition. Even a complete synthetic input remains synthetic evidence.
    Native callers must first verify all retained case artifacts and custody.
    """
    inventory = _validated(inventory)
    if inventory["mode"] != PROSPECTIVE:
        raise ValueError("A single public development fixture cannot support population intervals")
    contract = validate_comparison_contract(contract_bytes,
        selected_settings_bytes=selected_settings_bytes, source_files=source_files,
        exclusion_inventory_bytes=exclusion_inventory_bytes)
    definition = parse_json(contract.canonical_json)
    if definition["cohort_id"] != inventory["cohort_id"]:
        raise ValueError("Contract and case inventory identities differ")
    for digest, name in ((raw_manifest_sha256, "raw manifest"),
                         (custody_statement_sha256, "custody statement")):
        _digest(digest, name)
    _digest(seed_hex, "private bootstrap root")
    cases, rows, completeness = validate_results(inventory, records)
    required_sources = {"src/probebench/"+name for name in ("comparison.py", "comparison_pilot.py",
                        "comparison_inventory.py", "comparison_report.py", "nominal_calibration.py")}
    if not required_sources <= set(definition["sources"]["files"]):
        raise ValueError("Complete runner/report source bindings are required")
    shards = {}
    for identity, case in cases.items():
        if identity in rows:
            shards.setdefault(case["base_wavefront_id"], {})[identity] = _sha(rows[identity])
    binding = {"scientific_contract_sha256": contract.sha256, "inventory_sha256": _sha(inventory),
               "raw_manifest_sha256": raw_manifest_sha256,
               "custody_statement_sha256": custody_statement_sha256,
               "source_inventory_sha256": _sha(definition["sources"]),
               "runtime": definition["runtime"], "primary": dict(PRIMARY),
               "secondary_targets": SECONDARY_TARGETS, "secondary_resamples": 10000,
               "secondary_confidence_level": 0.95,
               "bootstrap_root_sha256": hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest(),
               "case_result_shards": {base: _sha(values) for base, values in sorted(shards.items())}}
    report = {"schema": "probebench.comparison-analysis/1", "inventory": inventory, "binding": binding,
              "analysis_binding_sha256": _sha(binding), "completeness": completeness,
              "scientific_status": "conditional_analysis_of_supplied_evidence",
              "provenance_authenticated": False, "execution_admitted": False,
              "held_out_status_established": False,
              "primary_family_scope": "six_cells_bonferroni_nominal_conditional_on_marginal_coverage",
              "secondary_scope": "descriptive_pointwise_no_confirmatory_claim",
              "metric_semantics": "final_wfe_and_strehl_are_evaluator_computations",
              "contract_status": contract.status,
              "case_results": [rows[identity] for identity in sorted(rows)], "contrasts": []}
    if not completeness["all_scientific_outcomes_available"]:
        report["status"] = "withheld_incomplete_or_failed_evidence"
        report["output_sha256"] = _output_fingerprint(report)
        return report
    report["status"] = "complete_conditional_analysis"
    for target in [TARGET, *SECONDARY_TARGETS]:
        parameters = dict(PRIMARY) if target == TARGET else {
            "reference_method": REFERENCE, "target_method": target, "track": "classical",
            "resamples": 10000, "confidence_level": 0.95}
        subset = [case for case in cases.values() if case["method"] in {REFERENCE, target}]
        outcomes = [{"case_id": case["id"], "residual_wfe_nm": rows[case["id"]]["final_residual_wfe_nm"],
                     "outcome_status": "completed", "failure_reason": None} for case in subset]
        reduction = paired_curve(subset, outcomes, seed_hex=seed_hex, **parameters)
        if ([(cell["photons_per_exposure"], cell["exposure_budget"]) for cell in reduction["cells"]] != list(CELLS)
                or reduction["independent_base_count"] != 96
                or reduction["stratum_base_counts"] != {name: 32 for name in ("low", "medium", "high")}):
            raise ValueError("Reducer output disagrees with the complete comparison design")
        verdicts = []
        for cell in reduction["cells"]:
            lower, upper = cell["confidence_interval_nm"]
            verdicts.append("advantage" if upper < 0 else "disadvantage" if lower > 0 else "uncertain")
        report["contrasts"].append({"role": "primary" if target == TARGET else "secondary",
            "parameters": parameters, "cell_verdicts": verdicts,
            "all_six_primary_cells_advantage": target == TARGET and all(x == "advantage" for x in verdicts),
            "reduction": reduction})
    report["output_sha256"] = _output_fingerprint(report)
    return report


def _output_fingerprint(report):
    """Bind every returned value using bounded, ordered JSON fragments.

    This is integrity against accidental mutation, not authentication. It binds
    raw returned seeds and draws as well as the envelope and caller-independent
    case summaries, without passing a multi-megabyte object to the small parser.
    """
    digest = hashlib.sha256()
    def add(label, value):
        payload, name = canonical_bytes(value), label.encode("ascii")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    def chunks(label, values, maximum, count):
        if type(values) is not list or len(values) > maximum:
            raise ValueError("Analysis evidence exceeds its fragment bound")
        add(label+"-count", len(values))
        for start in range(0, len(values), count):
            add(label+f"-{start}", values[start:start+count])
    header = {key: value for key, value in report.items()
              if key not in {"contrasts", "case_results", "completeness", "output_sha256"}}
    add("header", header)
    completeness = dict(report["completeness"])
    for name in ("missing_case_ids", "failed_case_ids"):
        chunks(name, completeness.pop(name), 13824, 500)
    add("completeness", completeness)
    chunks("cases", report["case_results"], 13824, 50)
    contrasts = report["contrasts"]
    if type(contrasts) is not list or len(contrasts) > 5:
        raise ValueError("Analysis contrast count exceeds its bound")
    add("contrast-count", len(contrasts))
    for index, contrast in enumerate(contrasts):
        prefix = f"contrast-{index}"
        add(prefix, {key: value for key, value in contrast.items() if key != "reduction"})
        reduction = contrast["reduction"]
        add(prefix+"-reduction", {key: value for key, value in reduction.items()
                                if key not in {"cells", "paired_outcomes", "bootstrap"}})
        chunks(prefix+"-cells", reduction["cells"], 6, 1)
        chunks(prefix+"-pairs", reduction["paired_outcomes"], 2304, 100)
        bootstrap = reduction["bootstrap"]
        add(prefix+"-bootstrap", {key: value for key, value in bootstrap.items()
                                  if key != "difference_curves_nm"})
        chunks(prefix+"-draws", bootstrap["difference_curves_nm"], 50000, 1000)
    return digest.hexdigest()


def write_analysis(output, report):
    """Write private reduction evidence in bounded shards, retaining every draw.

    This output includes private bootstrap seeds. Publication requires a separate
    review; never copy this directory wholesale into a public release.
    """
    if type(report) is not dict or report.get("schema") != "probebench.comparison-analysis/1":
        raise ValueError("Expected a comparison analysis")
    inventory = _validated(report["inventory"])
    _, _, completeness = validate_results(inventory, report["case_results"])
    if (completeness != report["completeness"]
            or report["analysis_binding_sha256"] != _sha(report["binding"])):
        raise ValueError("Changed analysis binding or completeness")
    contrasts = report["contrasts"]
    if type(contrasts) is not list or len(contrasts) not in (0, 5):
        raise ValueError("Unexpected comparison contrast inventory")
    if bool(contrasts) != completeness["all_scientific_outcomes_available"]:
        raise ValueError("Complete contrasts require every scientific outcome")
    names = []
    for contrast in contrasts:
        target = contrast["parameters"]["target_method"]
        expected = dict(PRIMARY) if target == TARGET else {
            "reference_method": REFERENCE, "target_method": target, "track": "classical",
            "resamples": 10000, "confidence_level": 0.95}
        if target not in [TARGET, *SECONDARY_TARGETS] or contrast["parameters"] != expected:
            raise ValueError("Changed analysis parameters")
        names.append(target)
        reduction = contrast["reduction"]
        if (len(reduction["cells"]) != 6 or len(reduction["paired_outcomes"]) > 2304
                or len(reduction["bootstrap"]["difference_curves_nm"]) != expected["resamples"]):
            raise ValueError("Analysis evidence exceeds its declared dimensions")
        if any(type(row) is not list or len(row) != 6
               for row in reduction["bootstrap"]["difference_curves_nm"]):
            raise ValueError("Invalid six-cell bootstrap shape")
    if contrasts and names != [TARGET, *SECONDARY_TARGETS]:
        raise ValueError("Changed analysis contrast order")
    _digest(report.get("output_sha256"), "complete analysis output")
    if _output_fingerprint(report) != report["output_sha256"]:
        raise ValueError("Analysis output changed after reduction")
    root = Path(output)
    root.mkdir()
    children = []
    header = {key: value for key, value in report.items() if key not in {"contrasts", "case_results"}}
    header["completeness"] = dict(report["completeness"])
    root_files = {"analysis.json"}
    for field in ("missing_case_ids", "failed_case_ids"):
        values = header["completeness"].pop(field)
        header["completeness"][field+"_count"] = len(values)
        for offset in range(0, len(values), 500):
            name = f"{field}-{offset:06d}.json"
            pilot._json(root/name, values[offset:offset+500])
            root_files.add(name)
    for offset in range(0, len(report["case_results"]), 50):
        name = f"case-results-{offset:06d}.json"
        pilot._json(root/name, report["case_results"][offset:offset+50])
        root_files.add(name)
    for contrast in report["contrasts"]:
        name = contrast["parameters"]["target_method"]
        folder = root/name
        folder.mkdir()
        reduction = contrast["reduction"]
        metadata = {key: value for key, value in reduction.items()
                    if key not in {"cells", "paired_outcomes", "bootstrap"}}
        metadata["bootstrap"] = {key: value for key, value in reduction["bootstrap"].items()
                                  if key != "difference_curves_nm"}
        metadata["role"] = contrast["role"]
        metadata["cell_verdicts"] = contrast["cell_verdicts"]
        files = {"summary.json"}
        pilot._json(folder/"summary.json", metadata)
        for i, cell in enumerate(reduction["cells"]):
            name_cell = f"cell-{i}.json"
            pilot._json(folder/name_cell, cell)
            files.add(name_cell)
        for prefix, values, chunk in (("pairs", reduction["paired_outcomes"], 100),
                ("bootstrap", reduction["bootstrap"]["difference_curves_nm"], 1000)):
            for offset in range(0, len(values), chunk):
                name_chunk = f"{prefix}-{offset:06d}.json"
                pilot._json(folder/name_chunk, values[offset:offset+chunk])
                files.add(name_chunk)
        _seal(folder, kind="analysis-contrast", files=files, children={})
        children.append(name)
    pilot._json(root/"analysis.json", header)
    return _seal(root, kind="analysis", files=root_files, children=children)


def write_development_report(packet, output):
    """Verify the native packet first and publish a descriptive single-fixture report."""
    verified = verify_development(packet)
    root = Path(output)
    root.mkdir()
    records = [summarize_episode(row["case"], row, row["manifest_sha256"])
               for row in verified["outcomes"]]
    _, _, completeness = validate_results(verified["inventory"], records)
    if not completeness["all_scientific_outcomes_available"]:
        raise ValueError("Native development packet is incomplete")
    files = {"summary.json", "README.md"}
    for record in records:
        name = record["case_id"]+".json"
        pilot._json(root/name, record)
        files.add(name)
    summary = {key: value for key, value in verified.items() if key not in {"outcomes", "study"}}
    summary.update({"completeness": completeness, "source_sha256": verified["study"]["source_sha256"],
                    "runtime": verified["study"]["runtime"], "statistical_inference": "none",
                    "independent_population_sample": False,
                    "final_metrics": "simulated_evaluator_computations_no_new_final_exposure"})
    pilot._json(root/"summary.json", summary)
    lines = ["# Classical comparison: public development fixture", "",
        "One already exposed mixed_a wavefront; six frozen methods and six sensing budgets.",
        "These cases check execution and retained evidence. They do not estimate population performance.",
        "Final WFE and Strehl are simulated evaluator quantities. Submission acquires no new image.",
        "An identical earlier command is not a fresh post-decision confirmation.", "",
        "| Method | Photons/frame | Frames | Final WFE (nm) | Maximum acquired-frame WFE increase (nm) | Fits |",
        "|---|---:|---:|---:|---:|---:|"]
    for case, record in zip(case_inventory(verified["inventory"]), records):
        lines.append(f'| {case["method"]} | {case["photons_per_exposure"]} | {case["exposure_budget"]} '
            f'| {record["final_residual_wfe_nm"]:.6g} | {record["peak_acquired_probe_wfe_increase_nm"]:.6g} '
            f'| {record["fit_calls"]} |')
    pilot._write(root/"README.md", ("\n".join(lines)+"\n").encode())
    return _seal(root, kind="development-report", files=files, children={})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_development_report(args.packet, args.output)


if __name__ == "__main__":
    main()
