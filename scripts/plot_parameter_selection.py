"""Plot all retained tuning candidates and the globally selected reference curves."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = {
    "no_action_v2": ("No action", "#6d7278"),
    "unprobed_mle_v1": ("Unprobed MLE", "#9272a4"),
    "random_probe_v2": ("Random probe", "#d99738"),
    "coordinate_search_v2": ("Coordinate search", "#4487b8"),
    "fixed_diversity_mle_v1": ("Fixed diversity", "#157f78"),
    "phase_diversity_mle_v1": ("Adaptive diversity", "#bc5865"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if (selection["schema"] != 1 or selection["criterion"] !=
            "equal_cell_stratum_base_member_replicate_mean_wfe_v1"):
        raise ValueError("Unsupported selection reduction")
    rows = selection["candidates"]
    by_id = {row["candidate_id"]: row for row in rows}
    if (len(by_id) != len(rows) or set(selection["selected_candidates"]) != set(FAMILIES)
            or {row["family"] for row in rows} != set(FAMILIES)):
        raise ValueError("Incomplete or duplicate family/candidate inventory")
    chosen = {family: by_id[identity] for family, identity in
              selection["selected_candidates"].items()}
    if any(row["family"] != family for family, row in chosen.items()):
        raise ValueError("Selected candidate belongs to another family")
    if any(not np.isfinite(row["loss_nm"]) or row["loss_nm"] <= 0 for row in rows):
        raise ValueError("Logarithmic figure requires finite positive losses")
    cells = {(cell["photons_per_exposure"], cell["exposure_budget"])
             for cell in rows[0]["cells"]}
    for row in rows:
        identities = [(cell["photons_per_exposure"], cell["exposure_budget"])
                      for cell in row["cells"]]
        if set(identities) != cells or len(identities) != len(cells):
            raise ValueError("Candidate cell coverage differs")
        if any(not np.isfinite(cell["loss_nm"]) or cell["loss_nm"] <= 0
               for cell in row["cells"]):
            raise ValueError("Logarithmic curve requires finite positive losses")
    base_count = len({base["base_wavefront_id"] for cell in rows[0]["cells"]
                      for stratum in cell["strata"] for base in stratum["bases"]})
    output = args.output.absolute()
    source = args.selection.absolute().parent.resolve()
    if (output.exists() or output.is_symlink() or output.resolve().is_relative_to(source)
            or source.is_relative_to(output.resolve())):
        raise ValueError("Use a new output directory outside the input evidence directory")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b5b7ba", "svg.hashsalt": "probebench-selection-v1"})

    ordered = [row for family in FAMILIES for row in rows if row["family"] == family]
    fig, axis = plt.subplots(figsize=(9.0, 7.6), layout="constrained")
    for position, row in enumerate(ordered):
        family = row["family"]
        selected = selection["selected_candidates"][family] == row["candidate_id"]
        color = FAMILIES[family][1]
        axis.scatter(row["loss_nm"], position, marker="D" if selected else "o",
                     s=60 if selected else 38, color=color,
                     facecolors=color if selected else "none", zorder=3)
        axis.annotate(f"{row['loss_nm']:.5f}", (row["loss_nm"], position),
                      xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    axis.set_yticks(range(len(ordered)), [row["candidate_id"] for row in ordered])
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlim(min(row["loss_nm"] for row in rows) / 1.25,
                  max(row["loss_nm"] for row in rows) * 1.65)
    axis.set_xlabel("Global tuning loss: residual WFE (nm; logarithmic scale)")
    axis.grid(axis="x", color="#e4e5e7", linewidth=0.6)
    axis.set_title("All tuning candidates\nFilled diamonds: one global setting per family", loc="left", pad=16)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"candidate-losses.{suffix}", dpi=180,
                    metadata={"Date": None} if suffix == "svg" else {})
    plt.close(fig)

    photons = sorted({p for p, _ in cells})
    fig, axes = plt.subplots(1, len(photons), figsize=(10.0, 5.2), squeeze=False, sharey=True)
    for axis, level in zip(axes[0], photons, strict=True):
        for family, row in chosen.items():
            values = sorted((cell["exposure_budget"], cell["loss_nm"])
                            for cell in row["cells"] if cell["photons_per_exposure"] == level)
            axis.plot([p[0] for p in values], [p[1] for p in values], marker="o",
                      markersize=4, linewidth=1.6, color=FAMILIES[family][1],
                      label=FAMILIES[family][0])
        axis.set_title(f"{level:,.0f} expected photons per exposure", fontsize=11)
        axis.set_xlabel("Exposures per episode")
        axis.set_xticks(sorted({b for p, b in cells if p == level}))
        axis.set_yscale("log")
        axis.grid(color="#e4e5e7", linewidth=0.6)
    axes[0, 0].set_ylabel("Tuning residual WFE (nm; logarithmic scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Globally selected settings across the six tuning cells", x=0.075, ha="left", fontsize=13)
    fig.text(0.075, 0.92, f"{base_count} base wavefronts · equal RMS-stratum weights · no held-out inference", fontsize=10)
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.075, right=0.98, wspace=0.08)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"selected-curves.{suffix}", dpi=180,
                    metadata={"Date": None} if suffix == "svg" else {})
    plt.close(fig)


if __name__ == "__main__":
    main()
