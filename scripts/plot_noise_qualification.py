"""Plot every retained estimator fit; never rewrite the source run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probebench.qualification import verify_qualification
from probebench.run import digest, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = verify_qualification(args.run)
rows = json.loads((args.run / "summary.json").read_text())
if any(row["status"] != "finite" for row in rows):
    raise ValueError("This finite-fit figure needs an explicit failure panel before use on failed fits")
args.output.mkdir(parents=True, exist_ok=False)
methods = ["least_squares", "poisson_gaussian_mle"]
labels = ["Unweighted least squares", "Poisson–Gaussian likelihood"]
colors = ["#ad5e34", "#246c70"]
budgets = sorted({row["expected_photons_used"] for row in rows})
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
fig.set_facecolor("#fbfaf7")
for ax in axes:
    ax.set_facecolor("#fbfaf7")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#d9d7d2", linewidth=.6, alpha=.6)
    ax.set_axisbelow(True)
for method, label, color in zip(methods, labels, colors):
    selected = [row for row in rows if row["method"] == method]
    identities = sorted({(r["truth_index"], r["replicate"]) for r in selected})
    for truth, replicate in identities:
        trajectory = sorted((r for r in selected if (r["truth_index"], r["replicate"]) == (truth, replicate)),
                            key=lambda r: r["expected_photons_used"])
        axes[0].loglog([r["expected_photons_used"] for r in trajectory],
                       [r["residual_wfe_nm"] for r in trajectory], "o-", color=color,
                       alpha=.25, markersize=3, linewidth=.7)
    axes[0].loglog(budgets, [np.mean([r["residual_wfe_nm"] for r in selected
                                    if r["expected_photons_used"] == b]) for b in budgets],
                   "o-", color=color, linewidth=2.2, markersize=5, label=label)
axes[0].set(xlabel="Expected photons across three exposures", ylabel="Residual pupil WFE (nm)",
            title="Every fit; bold lines show the means")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
for budget, marker, color in zip(budgets, ["o", "s", "^"], ["#7e506d", "#68752d", "#2c6a89"]):
    case_ids = sorted({r["case_id"] for r in rows if r["expected_photons_used"] == budget})
    paired = [{r["method"]: r["residual_wfe_nm"] for r in rows if r["case_id"] == case_id}
              for case_id in case_ids]
    axes[1].scatter([p[methods[0]] for p in paired], [p[methods[1]] for p in paired],
                    label=f"{budget:,} photons", color=color, marker=marker, s=34)
limits = [.1, 10]
axes[1].plot(limits, limits, "--", color="#777777", linewidth=1, label="Equal residual error")
axes[1].set(xscale="log", yscale="log", xlim=limits, ylim=limits,
            xlabel="Least-squares residual WFE (nm)", ylabel="Noise-aware residual WFE (nm)",
            title="24 paired cases on identical measurements")
axes[1].set_aspect("equal", adjustable="box")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("ProbeBench · noise qualification on known development fixtures", fontsize=12)
fig.savefig(args.output / "all-fits.png", dpi=180)
fig.savefig(args.output / "all-fits.svg")
plt.close(fig)
write_json(args.output / "provenance.json", {
    "input_run": str(args.run), "input_manifest_sha256": digest(args.run / "manifest.json"),
    "case_count": manifest["case_count"], "fit_count": manifest["fit_count"],
    "plot_source_sha256": digest(Path(__file__)),
    "outputs_sha256": {name: digest(args.output / name) for name in ("all-fits.png", "all-fits.svg")},
    "interpretation": "Descriptive development fixtures; no population inference or isolated likelihood-effect claim",
})
