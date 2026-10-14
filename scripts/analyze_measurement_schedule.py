"""Verify acquisition controls and report every paired measurement-schedule outcome."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probebench.environment import Observation
from probebench.estimation import phase_objective
from probebench.optics import OpticalModel, OpticsConfig
from probebench.run import digest, write_json
from probebench.sequential import verify_run


STYLES = {
    "unprobed_mle_v1": ("Unprobed estimation", "#ad5e34"),
    "fixed_diversity_mle_v1": ("Fixed diversity", "#80668d"),
    "phase_diversity_mle_v1": ("Adaptive diversity", "#246c70"),
}
CONJUGACY = np.array([-1., -1., -1., 1., 1., -1.])


def observations(root, record):
    case = record["case"]
    history = []
    for index in range(case["exposure_budget"]):
        with np.load(root/"frames"/case["id"]/f"frame_{index:04d}.npz", allow_pickle=False) as saved:
            meta = json.loads(saved["record_json"].item())["observation"]
            history.append(Observation(index, tuple(meta["applied_opd_nm"]),
                meta["expected_photons"], meta["cumulative_expected_photons"],
                saved["image_electrons"].copy()))
    return history


def analyze(root):
    manifest = verify_run(root)
    config = json.loads((root/"study.json").read_text())["configuration"]
    if set(config["methods"]) != set(STYLES):
        raise ValueError("The measurement-schedule analysis requires all three declared controls")
    model = OpticalModel(OpticsConfig(**config["optics"]))
    records = [json.loads(p.read_text()) for p in sorted((root/"cases").glob("episode_*.json"))]
    rows, symmetry, indexed = [], [], {}
    for record in records:
        case, result = record["case"], record["result"]
        key = (case["photons_per_exposure"], case["exposure_budget"],
               case["truth_index"], case["replicate"], case["method"])
        if key in indexed:
            raise ValueError("Duplicate paired case")
        indexed[key] = record
        truth = np.array(case["truth_opd_nm"])
        command = np.array(result["final_applied_opd_nm"])
        other_error = model.evaluate(CONJUGACY*truth-command)["residual_wfe_nm"]
        fits = result["fit_records"]
        rows.append({"episode":case["id"], "method":case["method"],
            "truth_index":case["truth_index"], "replicate":case["replicate"],
            "photons_per_exposure":case["photons_per_exposure"], "exposures":case["exposure_budget"],
            "final_wfe_nm":result["final"]["residual_wfe_nm"],
            "conjugate_wfe_nm":other_error,
            "wfe_modulo_conjugacy_nm":min(result["final"]["residual_wfe_nm"],other_error),
            "strehl_peak_ratio":result["final"]["strehl_peak_ratio"],
            "peak_probe_wfe_increase_nm":result["peak_probe_wfe_increase_nm"],
            "fit_count":len(fits),
            "failed_fits":sum(f["fit_status"] == "numerical_failure" for f in fits),
            "unconverged_fits":sum(f["fit_status"] == "finite" and not f["optimizer_success"] for f in fits),
            "optimizer_starts":sum(len(f.get("starts",[])) for f in fits),
            "unconverged_starts":sum(not s["success"] for f in fits for s in f.get("starts",[])),
            "decision_seconds":sum(result["decision_seconds"])+result["submission_seconds"],
            "final_command_has_camera_observation":result["final_command_has_camera_observation"]})
        if case["method"] == "unprobed_mle_v1":
            history = observations(root,record)
            if any(any(o.applied_opd_nm) for o in history):
                raise ValueError("Unprobed control contains a nonzero command")
            direct = phase_objective(model,history,config["read_noise_electrons"],truth/100)[0]
            conjugate = phase_objective(model,history,config["read_noise_electrons"],CONJUGACY*truth/100)[0]
            delta = abs(direct-conjugate)
            if delta > 1e-10:
                raise ValueError("Unprobed likelihood violates the conjugate control")
            symmetry.append({"episode":case["id"], "nll_per_photon_absolute_difference":delta})

    comparisons, equal_controls, statistics = [], [], []
    for photons in config["photons_per_exposure"]:
        for budget in config["exposure_budgets"]:
            for truth in range(len(config["truths_opd_nm"])):
                for replicate in range(config["replicates"]):
                    identity = (photons,budget,truth,replicate)
                    fixed = indexed[identity+("fixed_diversity_mle_v1",)]
                    adaptive = indexed[identity+("phase_diversity_mle_v1",)]
                    delta = fixed["result"]["final"]["residual_wfe_nm"]-adaptive["result"]["final"]["residual_wfe_nm"]
                    comparisons.append({"photons_per_exposure":photons, "exposures":budget,
                        "truth_index":truth,"replicate":replicate,
                        "fixed_episode":fixed["case"]["id"],"adaptive_episode":adaptive["case"]["id"],
                        "fixed_minus_adaptive_wfe_nm":delta})
                    if budget <= 2:
                        for first, second in zip(observations(root,fixed),observations(root,adaptive)):
                            if (first.metadata() != second.metadata()
                                or not np.array_equal(first.image_electrons,second.image_electrons)):
                                raise ValueError("Matched acquisition controls have different frames")
                        if fixed["result"]["final_applied_opd_nm"] != adaptive["result"]["final_applied_opd_nm"]:
                            raise ValueError("Matched acquisition controls have different submissions")
                        equal_controls.append({"fixed_episode":fixed["case"]["id"],
                                               "adaptive_episode":adaptive["case"]["id"]})
            for method in config["methods"]:
                selected = [r for r in rows if (r["photons_per_exposure"],r["exposures"],r["method"]) == (photons,budget,method)]
                statistics.append({"photons_per_exposure":photons,"exposures":budget,"method":method,
                    "episodes":len(selected),
                    **{f"mean_{key}":float(np.mean([r[key] for r in selected])) for key in (
                        "final_wfe_nm", "wfe_modulo_conjugacy_nm", "strehl_peak_ratio",
                        "peak_probe_wfe_increase_nm", "decision_seconds")},
                    **{key:sum(r[key] for r in selected) for key in (
                        "fit_count", "failed_fits", "unconverged_fits", "optimizer_starts", "unconverged_starts")},
                    "unmeasured_final_commands":sum(not r["final_command_has_camera_observation"] for r in selected),
                    "closer_to_conjugate_than_truth":sum(r["conjugate_wfe_nm"] < r["final_wfe_nm"]-1e-10 for r in selected)})
    return {"episode_count":manifest["episode_count"],"exposure_count":manifest["exposure_count"],
        "configuration":config,"episodes":rows,"statistics":statistics,"paired_comparisons":comparisons,
        "identical_fixed_adaptive_controls":equal_controls,"unprobed_likelihood_symmetry":symmetry}


def plot(data, output):
    config = data["configuration"]
    budgets = config["exposure_budgets"]
    x = np.asarray(budgets,dtype=float)
    fig, axes = plt.subplots(len(config["photons_per_exposure"]),3,
        figsize=(13.5,4*len(config["photons_per_exposure"])), squeeze=False, constrained_layout=True)
    fig.set_facecolor("#fbfaf7")
    for row_index, photons in enumerate(config["photons_per_exposure"]):
        left, middle, right = axes[row_index]
        for ax in axes[row_index]:
            ax.set_facecolor("#fbfaf7")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_xticks(x,labels=[str(v) for v in budgets])
            ax.minorticks_off()
            ax.grid(color="#d9d7d2",linewidth=.6,alpha=.65)
            ax.set_axisbelow(True)
            ax.set_xlabel("Acquired exposures (log scale)")
        for method in config["methods"]:
            label,color = STYLES[method]
            selected = [r for r in data["episodes"] if r["method"] == method and r["photons_per_exposure"] == photons]
            for identity in sorted({(r["truth_index"],r["replicate"]) for r in selected}):
                trajectory = sorted((r for r in selected if (r["truth_index"],r["replicate"]) == identity),key=lambda r:r["exposures"])
                for ax,key in ((left,"final_wfe_nm"),(right,"peak_probe_wfe_increase_nm")):
                    ax.plot(x,[r[key] for r in trajectory],"o-",color=color,alpha=.2,markersize=2,linewidth=.6)
            for ax,key in ((left,"final_wfe_nm"),(right,"peak_probe_wfe_increase_nm")):
                ax.plot(x,[np.mean([r[key] for r in selected if r["exposures"] == b]) for b in budgets],
                    "o-",color=color,linewidth=2,markersize=4,label=label)
        paired = [r for r in data["paired_comparisons"] if r["photons_per_exposure"] == photons]
        for index,budget in enumerate(budgets):
            values = [r["fixed_minus_adaptive_wfe_nm"] for r in paired if r["exposures"] == budget]
            middle.scatter(x[index]*(1+np.linspace(-.04,.04,len(values))),values,s=18,color="#626b72",alpha=.75)
            middle.plot(x[index],np.mean(values),"D",color="#246c70",markersize=5)
        middle.axhline(0,color="#434943",linewidth=.8)
        left.set_ylabel("Final evaluator WFE (nm)")
        if all(r["final_wfe_nm"] > 0 for r in data["episodes"]):
            left.set_yscale("log")
        middle.set_ylabel("Fixed − adaptive WFE (nm)")
        right.set_ylabel("Peak WFE increase above initial (nm)")
        left.set_title(f"{photons:,} photons / exposure · final error",fontsize=10)
        middle.set_title("Paired differences · positive favours adaptive",fontsize=10)
        right.set_title("Temporary measurement degradation",fontsize=10)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("ProbeBench · measurement schedules on known development wavefronts\n"
        "Thin lines and dots: all paired cases. Bold lines and diamonds: descriptive means.",fontsize=12)
    fig.savefig(output/"measurement-schedules.png",dpi=180)
    fig.savefig(output/"measurement-schedules.svg")
    plt.close(fig)


def report(data):
    rows = data["episodes"]
    lines = ["# Measurement-schedule development results", "",
        f'{data["episode_count"]} episodes and {data["exposure_count"]} charged exposures. '
        "Every configured case is included. Final scores are evaluator quantities; final commands may be unmeasured.", "",
        "| Photons/exposure | Exposures | Method | Mean WFE (nm) | Mean WFE modulo conjugacy (nm) | Mean Strehl | Mean peak WFE increase (nm) | Failed / nonconverged selected fits |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in data["statistics"]:
        lines.append(f'| {r["photons_per_exposure"]} | {r["exposures"]} | {r["method"]} | '
            f'{r["mean_final_wfe_nm"]:.4f} | {r["mean_wfe_modulo_conjugacy_nm"]:.4f} | '
            f'{r["mean_strehl_peak_ratio"]:.6f} | {r["mean_peak_probe_wfe_increase_nm"]:.4f} | '
            f'{r["failed_fits"]} / {r["unconverged_fits"]} |')
    lines += ["", "## Fixed versus adaptive diversity", "",
        "Differences are fixed minus adaptive WFE. Positive values favour adaptive acquisition.", "",
        "| Photons/exposure | Exposures | Mean difference (nm) | Adaptive lower | Fixed lower | Exact ties |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for photons in data["configuration"]["photons_per_exposure"]:
        for budget in data["configuration"]["exposure_budgets"]:
            values = [r["fixed_minus_adaptive_wfe_nm"] for r in data["paired_comparisons"]
                      if (r["photons_per_exposure"],r["exposures"]) == (photons,budget)]
            lines.append(f'| {photons} | {budget} | {np.mean(values):.6f} | '
                         f'{sum(v > 0 for v in values)} | {sum(v < 0 for v in values)} | {sum(v == 0 for v in values)} |')
    symmetry = max(r["nll_per_photon_absolute_difference"] for r in data["unprobed_likelihood_symmetry"])
    lines += ["", f'{sum(r["fit_count"] for r in rows)} selected fits and '
        f'{sum(r["optimizer_starts"] for r in rows)} starts were retained. '
        f'{sum(r["failed_fits"] for r in rows)} selected fits failed numerically; '
        f'{sum(r["unconverged_fits"] for r in rows)} selected fits and '
        f'{sum(r["unconverged_starts"] for r in rows)} individual starts did not converge.', "",
        f'{len(data["identical_fixed_adaptive_controls"])} paired fixed/adaptive cases at one or two '
        "exposures have identical images and submitted commands. "
        f'The largest unprobed conjugate-likelihood difference is {symmetry:.3e} per expected photon.', "",
        "All single-exposure methods abstain and submit zero. Modulo-conjugacy error is an evaluator-only "
        "diagnostic, not the correction metric. A small value can coexist with a wrong physical correction.", "",
        "The four wavefront fixtures and shared noise streams are correlated. These descriptive results "
        "do not support population significance claims, a generally preferred schedule, or conclusions "
        "about model mismatch, multimodal agents or telescope performance. Numerical convergence does "
        "not establish identifiability. All paired differences, per-episode metrics and checks are retained "
        "in analysis.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    data = analyze(args.run)
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"analysis.json",data)
    (args.output/"report.md").write_text(report(data))
    plot(data,args.output)
    write_json(args.output/"manifest.json",{
        "input_manifest_sha256":digest(args.run/"manifest.json"),
        "analysis_source_sha256":digest(Path(__file__)),
        "files":{p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    })
    print(json.dumps({"episodes":data["episode_count"],"exposures":data["exposure_count"],
                     "paired_comparisons":len(data["paired_comparisons"])}))


if __name__ == "__main__":
    main()
