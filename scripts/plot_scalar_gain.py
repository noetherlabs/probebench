"""Plot the frozen scalar-response study's retained descriptive intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.analysis.read_bytes()
    analysis = json.loads(raw)
    if (analysis["purpose"] != "scalar_gain_development_v1"
            or analysis["status"] != "descriptive_complete"
            or analysis["interval_scope"] != "pointwise_descriptive"
            or analysis["independent_bases"] != 24):
        raise ValueError("Require the complete, frozen descriptive analysis")
    groups = [("Equal-stratum mean · 24 bases", analysis["equal_stratum"])]
    groups.extend((title, analysis["strata"][key]) for key, title in (
        ("low", "10–25 nm initial RMS · 8 bases"),
        ("medium", "25–50 nm initial RMS · 8 bases"),
        ("high", "50–90 nm initial RMS · 8 bases")))
    plotted = []
    for _, rows in groups:
        if [row["gain"] for row in rows] != [0.9, 1.0, 1.1]:
            raise ValueError("Unexpected command-response grid")
        values = np.array([[row["adaptive_minus_fixed"]["mean_nm"],
                            *row["adaptive_minus_fixed"]["interval_nm"]]
                           for row in rows])
        if (values.shape != (3, 3) or not np.isfinite(values).all()
                or np.any(values[:, 1] > values[:, 2])):
            raise ValueError("Invalid retained contrast or percentile interval")
        plotted.append(values)
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "scalar-command-response-v1"})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True, sharey=True)
    for ax, (title, _), values in zip(axes.flat, groups, plotted):
        ax.axhline(0, color="#7b8088", linewidth=1, linestyle="--")
        # Percentile intervals need not bracket the point estimate. Draw their
        # endpoints directly rather than treating them as nonnegative errors.
        ax.vlines([0.9, 1.0, 1.1], values[:, 1], values[:, 2], color="#256778", lw=1.8)
        for gain, (_, low, high) in zip([0.9, 1.0, 1.1], values):
            ax.hlines([low, high], gain - 0.008, gain + 0.008, color="#256778", lw=1.8)
        ax.scatter([0.9, 1.0, 1.1], values[:, 0], color="#153d4a", s=35, zorder=3)
        ax.set_title(title, loc="left", fontsize=10, pad=12)
        ax.set_xticks([0.9, 1.0, 1.1])
        ax.set_xlim(0.87, 1.13)
        ax.grid(axis="y", alpha=0.16)
    fig.supxlabel("Actual / nominal command response", fontsize=11, y=0.08)
    fig.supylabel("Adaptive − fixed residual WFE (nm)", fontsize=11, x=0.025)
    fig.text(0.5, 0.02, "Negative favors adaptive. Bars: pointwise 95% base-bootstrap intervals.\n"
             "Static simulated wavefronts; development data; no simultaneous or hardware claim.",
             ha="center", fontsize=9, color="#44494f")
    fig.tight_layout(rect=(0.045, 0.12, 1, 1), h_pad=2.3, w_pad=2)
    fig.savefig(args.out / "paired-contrasts.png", dpi=200,
                metadata={"Software": "ProbeBench"})
    fig.savefig(args.out / "paired-contrasts.svg",
                metadata={"Date": None, "Creator": "ProbeBench"})
    plt.close(fig)
    metadata = {"analysis_sha256": hashlib.sha256(raw).hexdigest(),
                "matplotlib_version": matplotlib.__version__,
                "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(args.out.iterdir())}}
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
