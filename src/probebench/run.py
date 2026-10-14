"""Retained development fixtures. No reserved benchmark cohort is defined here."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy

from .environment import Episode, EpisodeConfig
from .optics import MODES, OpticalModel, OpticsConfig
from .policies import Policy

METHODS = ("no_action", "random_probe", "coordinate_search", "phase_diversity")
# These exposed development fixtures must never be relabelled held-out results.
FIXTURES = (
    ("positive_defocus", [45, 0, 0, 0, 0, 0]),
    ("negative_defocus", [-45, 0, 0, 0, 0, 0]),
    ("mixed_a", [30, 12, -10, 8, -6, 15]),
    ("mixed_b", [-25, -8, 14, -10, 9, -12]),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def source_inventory():
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    paths = sorted(package.glob("*.py")) + [root / "pyproject.toml"]
    if (root / "uv.lock").exists():
        paths.append(root / "uv.lock")
    return root, {str(p.relative_to(root)): digest(p) for p in paths}


def verify_files(path):
    """Shared artifact-integrity check; callers validate their own study schema."""
    root = Path(path)
    if root.is_symlink():
        raise ValueError("Run root must not be a symbolic link")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Manifest must not be a symbolic link")
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Missing or unlisted run artifact")
    for name, expected in manifest["files"].items():
        file = root / name
        if (file.is_symlink() or not file.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in file.parents if p != root.parent)):
            raise ValueError("Unsafe artifact path")
        if digest(file) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return manifest


def verify_run(path):
    root = Path(path)
    manifest = verify_files(root)
    if manifest.get("schema") != 1 or manifest.get("purpose") != "development_only":
        raise ValueError("Unknown run schema or purpose")
    summaries = json.loads((root / "summary.json").read_text())
    if len(summaries) != manifest["episodes"]:
        raise ValueError("Incomplete episode summary")
    return manifest


def render_ambiguity(run_root):
    """Make the figure only from saved measured images and evaluator records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = json.loads((run_root / "summary.json").read_text())
    rows = [s for s in summaries if s["method"] == "phase_diversity"
            and s["fixture"] in ("positive_defocus", "negative_defocus")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for row, item in enumerate(rows):
        record = json.loads((run_root / "evaluator" / f'{item["episode_id"]}.json').read_text())
        with np.load(run_root / "observations" / f'{item["episode_id"]}.npz', allow_pickle=False) as data:
            images = data["image_electrons"]
            photons = data["expected_photons"]
            for col in range(3):
                scaled = np.log1p(np.maximum(images[col], 0)) / np.log1p(photons[col])
                ax = axes[row, col]
                ax.imshow(scaled, origin="lower", cmap="magma", vmin=0, vmax=1)
                metric = record["trajectory"][col]["residual_wfe_nm"]
                ax.set_title(f'{("Initial exposure", "Known +80 nm OPD probe", "Measured correction")[col]}\n'
                             f'WFE {metric:.2f} nm', fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f'{record["hidden_opd_nm"][0]:+g} nm defocus', fontsize=11)
    fig.suptitle("ProbeBench development fixtures: opposite defocus, indistinguishable initial intensity")
    fig.savefig(run_root / "ambiguity.png", dpi=160)
    plt.close(fig)


def write_development_report(run_root):
    summaries = json.loads((run_root / "summary.json").read_text())
    lines = [
        "# Two indistinguishable star images, two opposite corrections",
        "", "**Development demonstration, not final benchmark evidence.**", "",
        "For a symmetric pupil, positive and negative pure defocus have the same",
        "noiseless focal intensity. Noisy exposures therefore have the same distribution,",
        "although independent noise draws need not produce identical pixel arrays.", "",
        "A known positive OPD correction changes the two residual wavefronts differently.",
        "Here the phase-diversity policy uses an initial exposure, a known probe, then",
        "a third exposure to measure its inferred correction. The hidden wavefront never",
        "enters the policy. All measurements use the same fixed display transform.", "",
        "![Retained measurements](ambiguity.png)", "", "## Complete development outcomes", "",
        "Each row uses three exposures of 200,000 expected incident photons each, with",
        "Poisson photoelectrons and 1-electron Gaussian read noise. The table is generated",
        "from retained episode records. All four fixtures and all four policies are shown.", "",
        "| Fixture | Policy | Initial WFE (nm) | Final WFE (nm) | Sampled Strehl | Peak probe WFE increase (nm) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(f'| {row["fixture"]} | {row["method"]} | {row["initial_wfe_nm"]:.3f} | '
                     f'{row["residual_wfe_nm"]:.3f} | {row["strehl_peak_ratio"]:.6f} | '
                     f'{row["peak_probe_wfe_increase_nm"]:.3f} |')
    lines.extend(["", "## What this does and does not establish", "",
        "The development fixtures exercise information ambiguity, diagnostic intervention,",
        "noise accounting and retained evidence. The probe temporarily worsens the negative",
        "defocus case; its degradation is retained rather than hidden by a final-only plot.", "",
        "Four known fixtures do not establish statistical superiority or generalization.",
        "The phase-diversity reference uses multistart calibrated-intensity least squares;",
        "its noise likelihood and convergence still need qualification. A three-exposure",
        "coordinate search has only visited the defocus direction, so this is not a fair",
        "full six-mode optimizer comparison. No multimodal agent was run. The sampled scalar",
        "model is not telescope validation. Public development fixtures must never become",
        "reserved final evaluation under a different label.", "",
        "See configuration.json for fixtures, mode order, wavelength, limits and budgets;",
        "observations/ for controller-visible records; evaluator/ for private truth and",
        "per-exposure quality; source/ for the executed source snapshot and lockfile; and",
        "manifest.json for content hashes and runtime versions.", "",
        "Prior work: [pupil-diversity phase ambiguity](https://arxiv.org/abs/1607.01776) and",
        "[active perception by optical agents](https://arxiv.org/html/2609.32918v1).",
        "These sources motivate the task; this demonstration does not claim the first use",
        "of optical intervention or AI in laboratory optics.", ""])
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_development(output):
    output = Path(output)
    source_root, sources = source_inventory()
    # Exclusive creation prevents two writers from targeting the same experiment.
    output.mkdir(parents=True, exist_ok=False)
    for name in ("observations", "evaluator", "source"):
        (output / name).mkdir()
    for relative in sources:
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, dest)

    optics_config, episode_config = OpticsConfig(), EpisodeConfig()
    model = OpticalModel(optics_config)
    config = {"optics": asdict(optics_config), "episode": asdict(episode_config),
              "modes": MODES, "fixtures": FIXTURES, "methods": METHODS,
              "cohort": "public_development_fixtures_not_final_evaluation"}
    write_json(output / "configuration.json", config)
    summaries = []
    for case_index, (fixture, hidden) in enumerate(FIXTURES):
        for method_index, method in enumerate(METHODS):
            identity = f"episode_{len(summaries):04d}"
            # Separate random domains; identities/seeds are not passed to policies.
            detector_seed = 10_000_001 + case_index
            policy_seed = 20_000_001 + case_index * 100 + method_index
            episode = Episode(model, episode_config, hidden, detector_seed)
            policy = Policy(method, model, episode_config, policy_seed)
            history = []
            for _ in range(episode_config.max_exposures):
                history.append(episode.expose(policy.command(tuple(history))))
            result = episode.submit(policy.final_command(tuple(history)))
            result.update({"fixture": fixture, "method": method,
                           "detector_seed": detector_seed, "policy_seed": policy_seed,
                           "fit_diagnostics": policy.fit_records})
            write_json(output / "evaluator" / f"{identity}.json", result)
            write_json(output / "observations" / f"{identity}.json", {
                "public_instrument": {"optics": asdict(optics_config), "episode": asdict(episode_config)},
                "observations": [o.metadata() for o in history],
            })
            with (output / "observations" / f"{identity}.npz").open("xb") as f:
                np.savez_compressed(f,
                    image_electrons=np.stack([o.image_electrons for o in history]),
                    applied_opd_nm=np.array([o.applied_opd_nm for o in history]),
                    expected_photons=np.array([o.expected_photons for o in history], dtype=np.int64))
            summaries.append({"episode_id": identity, "fixture": fixture, "method": method,
                "initial_wfe_nm": result["initial"]["residual_wfe_nm"],
                **result["final"], "expected_photons_used": result["expected_photons_used"],
                "exposures": result["exposures"],
                "peak_probe_wfe_increase_nm": result["peak_probe_wfe_increase_nm"]})
    write_json(output / "summary.json", summaries)
    render_ambiguity(output)
    write_development_report(output)
    if source_inventory()[1] != sources:
        raise RuntimeError("Source changed during acquisition; output left incomplete")
    manifest = {
        "schema": 1, "purpose": "development_only", "episodes": len(summaries),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(),
                    "machine": platform.machine()},
        "source_sha256": sources,
        "files": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "limitations": ["development fixtures only", "no agent or API calls",
            "least-squares estimator not yet statistically qualified",
            "finite sampled scalar optics, no telescope validation",
            "no hostile-code sandbox", "no performance benchmark"],
    }
    write_json(output / "manifest.json", manifest)
    verify_run(output)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify_run(args.verify)
        print(f'Verified {manifest["episodes"]} development episodes and all artifact hashes.')
    else:
        rows = run_development(args.output)
        print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
