"""Retain solver outcomes and every nonmonotone phase-diversity budget comparison."""
import argparse
import json
from pathlib import Path

from probebench.run import digest, write_json
from probebench.sequential import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_run(args.run)
    records = [json.loads(p.read_text()) for p in sorted((args.run/"cases").glob("episode_*.json"))]
    fits = [fit for record in records for fit in record["result"]["fit_records"]]
    groups = {}
    for record in records:
        case = record["case"]
        if case["method"] == "phase_diversity_mle_v1":
            identity = (case["truth_index"], case["replicate"], case["photons_per_exposure"])
            groups.setdefault(identity, []).append(record)
    comparisons, increases = 0, []
    for group in groups.values():
        group.sort(key=lambda record: record["case"]["exposure_budget"])
        for before, after in zip(group, group[1:]):
            comparisons += 1
            first, last = (r["result"]["final"]["residual_wfe_nm"] for r in (before, after))
            if last > first:
                increases.append({"before_episode":before["case"]["id"],
                    "after_episode":after["case"]["id"], "before_wfe_nm":first, "after_wfe_nm":last})
    summary = {
        "input_manifest_sha256":digest(args.run/"manifest.json"),
        "check_source_sha256":digest(Path(__file__)),
        "episodes":len(records), "selected_fits":len(fits),
        "optimizer_starts":sum(len(f.get("starts", [])) for f in fits),
        "selected_nonconvergence":sum(f["fit_status"] == "finite" and not f["optimizer_success"] for f in fits),
        "numerical_failures":sum(f["fit_status"] == "numerical_failure" for f in fits),
        "start_nonconvergence":sum(not s["success"] for f in fits for s in f.get("starts", [])),
        "unmeasured_final_commands":sum(not r["result"]["final_command_has_camera_observation"] for r in records),
        "phase_diversity_adjacent_budget_comparisons":comparisons,
        "phase_diversity_nonmonotone_budget_pairs":increases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
