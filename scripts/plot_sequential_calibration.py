"""Plot every retained sequential episode and summarize solver/measurement status."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probebench.run import digest, write_json
from probebench.sequential import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_run(args.run)
    rows = json.loads((args.run / "summary.json").read_text())
    photons = sorted({r["photons_per_exposure"] for r in rows})
    if len(photons) != 1:
        raise ValueError("This figure requires a single declared photon level per exposure")
    if any(r["residual_wfe_nm"] <= 0 for r in rows):
        raise ValueError("Zero-error outcomes require a linear scale or an explicit zero panel")
    args.output.mkdir(parents=True, exist_ok=False)
    methods = ["no_action_v2", "random_probe_v2", "coordinate_search_v2", "phase_diversity_mle_v1"]
    labels = ["No action", "Random probes", "Coordinate search", "Sequential phase diversity"]
    colors = ["#626b72", "#ad5e34", "#80668d", "#246c70"]
    budgets = sorted({r["exposures"] for r in rows})
    statistics = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    fig.set_facecolor("#fbfaf7")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#d9d7d2", linewidth=.6, alpha=.65)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.set_xticks(budgets, labels=[str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel(f"Acquired exposures · {photons[0]:,} expected photons each")
    for method, label, color in zip(methods, labels, colors):
        selected = [r for r in rows if r["method"] == method]
        if not selected:
            continue
        identities = sorted({(r["truth_index"], r["replicate"]) for r in selected})
        for truth, replicate in identities:
            trajectory = sorted((r for r in selected if (r["truth_index"],r["replicate"]) == (truth,replicate)),
                                key=lambda r:r["exposures"])
            for ax, key in zip(axes, ["residual_wfe_nm", "peak_probe_wfe_increase_nm"]):
                ax.plot([r["exposures"] for r in trajectory], [r[key] for r in trajectory],
                        "o-", color=color, alpha=.22, markersize=2.5, linewidth=.7)
        for ax, key in zip(axes, ["residual_wfe_nm", "peak_probe_wfe_increase_nm"]):
            ax.plot(budgets, [np.mean([r[key] for r in selected if r["exposures"] == b]) for b in budgets],
                    "o-", color=color, linewidth=2, markersize=4, label=label)
        for budget in budgets:
            subset = [r for r in selected if r["exposures"] == budget]
            statistics.append({"method":method,"exposures":budget,"episodes":len(subset),
                "expected_photons_per_episode":budget*photons[0],
                "mean_residual_wfe_nm":float(np.mean([r["residual_wfe_nm"] for r in subset])),
                "mean_strehl_peak_ratio":float(np.mean([r["strehl_peak_ratio"] for r in subset])),
                "mean_peak_probe_wfe_increase_nm":float(np.mean([r["peak_probe_wfe_increase_nm"] for r in subset])),
                "mean_decision_seconds":float(np.mean([r["decision_seconds"] for r in subset])),
                "unmeasured_final_commands":sum(not r["final_command_has_camera_observation"] for r in subset),
                "fit_count":sum(r["fit_count"] for r in subset),
                "failed_fits":sum(r["failed_fits"] for r in subset),
                "unconverged_fits":sum(r["unconverged_fits"] for r in subset)})
    axes[0].set(yscale="log", ylabel="Final evaluator WFE (nm)",
                title="Residual phase after the submitted command")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=8,
               loc="outside lower center", ncol=4)
    axes[1].set(ylabel="Peak exposure WFE above initial WFE (nm)",
                title="Temporary degradation during measurement")
    axes[1].set_ylim(bottom=-2)
    fig.suptitle("ProbeBench · sequential calibration on known development wavefronts\n"
                 "Thin lines: paired fixture/noise trajectories. Bold lines: descriptive means.", fontsize=11)
    fig.savefig(args.output / "budget-curves.png", dpi=180)
    fig.savefig(args.output / "budget-curves.svg")
    plt.close(fig)
    write_json(args.output / "descriptive-statistics.json", statistics)
    write_json(args.output / "provenance.json", {
        "input_run":args.run.name,"input_manifest_sha256":digest(args.run/"manifest.json"),
        "episode_count":manifest["episode_count"],"exposure_count":manifest["exposure_count"],
        "plot_source_sha256":digest(Path(__file__)),
        "outputs_sha256":{name:digest(args.output/name) for name in
            ("budget-curves.png","budget-curves.svg","descriptive-statistics.json")},
        "interpretation":"Known, correlated development fixtures. No population inference; final evaluator scores need not have a corresponding measured image.",
    })


if __name__ == "__main__":
    main()
