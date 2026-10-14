"""Resumable development comparison of estimators on identical acquired images."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy

from .environment import Episode, EpisodeConfig, Observation
from .estimation import estimate_phase_mle
from .optics import OpticalModel, OpticsConfig, coefficients
from .policies import estimate_phase
from .run import digest, source_inventory, verify_files, write_json


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def publish_text(path, content):
    """Publish a complete file under the caller's study lock; recover stale buffers."""
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def publish_json(path, value):
    publish_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def cases_for(config):
    if config.get("purpose") != "public_development_only":
        raise ValueError("This runner accepts development data only")
    if config["methods"] != ["least_squares", "poisson_gaussian_mle"]:
        raise ValueError("Unknown estimator comparison")
    if type(config["replicates"]) is not int or config["replicates"] < 1:
        raise ValueError("Positive integer replicates required")
    if type(config["detector_seed_root"]) is not int or config["detector_seed_root"] < 0:
        raise ValueError("A nonnegative integer detector seed is required")
    if type(config["max_mle_iterations"]) is not int or config["max_mle_iterations"] < 1:
        raise ValueError("Positive integer MLE iteration budget required")
    if type(config["operating_reserve_bytes"]) is not int or config["operating_reserve_bytes"] < 0:
        raise ValueError("Nonnegative integer operating reserve required")
    if len(set(config["photons_per_exposure"])) != len(config["photons_per_exposure"]):
        raise ValueError("Duplicate photon levels are not permitted; use replicates")
    OpticsConfig(**config["optics"])
    commands = np.asarray([coefficients(c) for c in config["acquisition_commands_nm"]])
    if len(commands) < 2 or np.all(commands == commands[0]):
        raise ValueError("At least two diverse commands are required")
    if np.max(np.abs(commands)) > config["command_limit_nm"]:
        raise ValueError("Acquisition exceeds the public command limits")
    cases = []
    for photons in config["photons_per_exposure"]:
        EpisodeConfig(max_exposures=len(commands), photon_budget=len(commands)*photons,
                      photons_per_exposure=photons, read_noise_electrons=config["read_noise_electrons"],
                      command_limit_nm=config["command_limit_nm"])
        for truth_index, truth in enumerate(config["truths_opd_nm"]):
            coefficients(truth)
            for replicate in range(config["replicates"]):
                index = len(cases)
                cases.append({"id": f"case_{index:04d}", "truth_index": truth_index,
                    "truth_opd_nm": truth, "photons_per_exposure": photons, "replicate": replicate,
                    "detector_seed": config["detector_seed_root"] + index})
    if not cases:
        raise ValueError("No development cases specified")
    return cases


def _acquire_or_load(root, case, config, model, binding):
    path = root / "acquisitions" / f'{case["id"]}.npz'
    commands = np.array(config["acquisition_commands_nm"], dtype=np.float64)
    case_binding = canonical_hash({"study": binding, "case": case})
    photons = case["photons_per_exposure"]
    if not path.exists():
        episode_config = EpisodeConfig(max_exposures=len(commands), photon_budget=len(commands)*photons,
            photons_per_exposure=photons, read_noise_electrons=config["read_noise_electrons"],
            command_limit_nm=config["command_limit_nm"])
        episode = Episode(model, episode_config, case["truth_opd_nm"], case["detector_seed"])
        observed = [episode.expose(command) for command in commands]
        images = np.stack([o.image_electrons for o in observed])
        temporary = path.with_suffix(".pending")
        with temporary.open("wb") as f:
            np.savez_compressed(f, images=images, commands=commands, binding=np.array(case_binding),
                images_sha256=np.array(hashlib.sha256(images.tobytes()).hexdigest()))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as arrays:
        expected_shape = (len(commands), model.config.pupil_pixels*model.config.oversampling,
                          model.config.pupil_pixels*model.config.oversampling)
        if (set(arrays.files) != {"images", "commands", "binding", "images_sha256"}
            or arrays["binding"].item() != case_binding
            or not np.array_equal(arrays["commands"], commands)
            or arrays["images"].shape != expected_shape
            or not np.isfinite(arrays["images"]).all()
            or arrays["images_sha256"].item() != hashlib.sha256(arrays["images"].tobytes()).hexdigest()):
            raise ValueError(f'Acquisition binding or numeric schema changed: {case["id"]}')
        images = arrays["images"].copy()
    return [Observation(i, tuple(command), photons, photons*(i+1), image)
            for i, (command, image) in enumerate(zip(commands, images))], digest(path)


def _read_case(root, case, binding):
    path = root / "cases" / f'{case["id"]}.json'
    row = json.loads(path.read_text())
    signed = dict(row)
    recorded_hash = signed.pop("record_sha256")
    if (canonical_hash(signed) != recorded_hash or row["study_binding"] != binding
        or row["case"] != case or row["acquisition_sha256"] != digest(
            root / "acquisitions" / f'{case["id"]}.npz')):
        raise ValueError(f'Changed completed case evidence: {case["id"]}')
    return row


def _report(root, records, config):
    rows = []
    for record in records:
        case = record["case"]
        for method, result in record["fits"].items():
            rows.append({"case_id": case["id"], "truth_index": case["truth_index"],
                "replicate": case["replicate"], "method": method,
                "expected_photons_used": case["photons_per_exposure"]*len(config["acquisition_commands_nm"]),
                "residual_wfe_nm": result.get("residual_wfe_nm"),
                "status": result["status"], "optimizer_success": result.get("optimizer_success", False)})
    if (root / "summary.json").exists():
        if json.loads((root / "summary.json").read_text()) != rows:
            raise ValueError("Existing derived summary disagrees with retained cases")
    else:
        publish_json(root / "summary.json", rows)
    text = ["# Noise-aware phase diversity: development qualification", "",
        "This comparison uses identical retained exposures for both estimators. It evaluates",
        "estimated corrections without taking a final corrected camera image; it is not a",
        "closed-loop controller benchmark. Fixtures are public development cases only.", "",
        "| Expected photons used | Estimator | Finite fits | Converged fits | Mean residual WFE (nm) |",
        "| ---: | --- | ---: | ---: | ---: |"]
    for budget in sorted({r["expected_photons_used"] for r in rows}):
        for method in config["methods"]:
            selected = [r for r in rows if r["expected_photons_used"] == budget and r["method"] == method]
            finite = [r["residual_wfe_nm"] for r in selected if r["status"] == "finite"]
            mean = f'{np.mean(finite):.4f}' if finite else "unavailable"
            text.append(f'| {budget} | {method} | {len(finite)}/{len(selected)} | '
                        f'{sum(r["optimizer_success"] for r in selected)}/{len(selected)} | {mean} |')
    text += ["", "Means describe finite fits; the adjacent failure counts must accompany them.",
        "All individual fits, all optimizer starts, measured images and known calibration",
        "commands are retained. Four related wavefronts and two noise draws per setting do",
        "not justify a population ranking or a significance claim. This pilot uses an exact",
        "plant/twin match; robustness to optical mismatch remains untested. The two solvers",
        "have different optimization procedures, so this does not isolate likelihood choice",
        "from every convergence effect. Read the complete per-start records.", ""]
    content = "\n".join(text)
    if (root / "report.md").exists():
        if (root / "report.md").read_text() != content:
            raise ValueError("Existing report disagrees with retained cases")
    else:
        publish_text(root / "report.md", content)
    return rows


def verify_qualification(root):
    root = Path(root)
    manifest = verify_files(root)
    if manifest.get("schema") != 1 or manifest.get("purpose") != "estimator_development_qualification":
        raise ValueError("Unknown qualification schema")
    study = json.loads((root / "study.json").read_text())
    for relative, expected in study["source_sha256"].items():
        if digest(root / "source" / relative) != expected:
            raise ValueError("Source snapshot disagrees with study binding")
    binding = canonical_hash(study)
    cases = cases_for(study["configuration"])
    if manifest["case_count"] != len(cases) or manifest["source_sha256"] != study["source_sha256"]:
        raise ValueError("Manifest does not describe the bound study")
    for case in cases:
        _read_case(root, case, binding)
    rows = json.loads((root / "summary.json").read_text())
    if len(rows) != len(cases)*len(study["configuration"]["methods"]) or len(rows) != manifest["fit_count"]:
        raise ValueError("Incomplete qualification report")
    return manifest


def run_qualification(root, config_path, resume=False, limit_cases=None):
    root = Path(root)
    if limit_cases is not None and (type(limit_cases) is not int or limit_cases < 1):
        raise ValueError("limit_cases must be a positive integer")
    config = json.loads(Path(config_path).read_text())
    cases = cases_for(config)
    source_root, sources = source_inventory()
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "platform": platform.platform(), "machine": platform.machine()}
    study = {"configuration": config, "source_sha256": sources, "runtime": runtime}
    root.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = len(cases)*len(config["acquisition_commands_nm"])*(config["optics"]["pupil_pixels"]
                 *config["optics"]["oversampling"])**2*8
    required_bytes = config["operating_reserve_bytes"] + 3*raw_bytes + 16*1024**2
    if not root.exists() and shutil.disk_usage(root.parent).free < required_bytes:
        raise OSError("Insufficient disk for retained development evidence plus operating reserve")
    if root.is_symlink():
        raise ValueError("Run root must not be a symbolic link")
    root.mkdir(exist_ok=resume)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "study.json").exists():
            if json.loads((root / "study.json").read_text()) != study:
                raise ValueError("Source, configuration or runtime changed; resume refused")
        else:
            if any(p.name not in {".lock", "study.json.pending"} for p in root.iterdir()):
                raise ValueError("Cannot adopt an existing directory without a study binding")
            publish_json(root / "study.json", study)
        for directory in ("cases", "acquisitions", "source"):
            folder = root / directory
            if folder.is_symlink():
                raise ValueError("Study directories must not be symbolic links")
            folder.mkdir(exist_ok=True)
        for relative, expected in sources.items():
            dest = root / "source" / relative
            if dest.is_symlink() or not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError("Source snapshot path escapes study directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if digest(dest) != expected:
                    raise ValueError("Source snapshot changed; resume refused")
            else:
                temporary = dest.with_suffix(dest.suffix + ".pending")
                shutil.copyfile(source_root / relative, temporary)
                os.replace(temporary, dest)
        if (root / "manifest.json").exists():
            return verify_qualification(root)
        if shutil.disk_usage(root.parent).free < required_bytes:
            raise OSError("Insufficient disk for continued acquisition and operating reserve")
        model = OpticalModel(OpticsConfig(**config["optics"]))
        binding = canonical_hash(study)
        records, acquired = [], 0
        for case in cases:
            path = root / "cases" / f'{case["id"]}.json'
            if path.exists():
                records.append(_read_case(root, case, binding))
                continue
            if limit_cases is not None and acquired >= limit_cases:
                return {"complete": False, "completed_cases": len(records)}
            observations, acquisition_hash = _acquire_or_load(root, case, config, model, binding)
            fits = {}
            for method in config["methods"]:
                try:
                    if method == "least_squares":
                        estimate, diagnostics = estimate_phase(model, observations, config["command_limit_nm"])
                    else:
                        estimate, diagnostics = estimate_phase_mle(model, observations,
                            config["read_noise_electrons"], config["command_limit_nm"],
                            max_iterations=config["max_mle_iterations"])
                    metrics = model.evaluate(np.asarray(case["truth_opd_nm"])-estimate)
                    if not np.isfinite(list(metrics.values())).all():
                        raise FloatingPointError("Nonfinite evaluator result")
                    fits[method] = {"status": "finite", "estimated_opd_nm": estimate.tolist(),
                                   **metrics, **diagnostics}
                except FloatingPointError as error:
                    fits[method] = {"status": "numerical_failure", "reason": str(error)}
            record = {"case": case, "study_binding": binding, "acquisition_sha256": acquisition_hash,
                      "fits": fits}
            record["record_sha256"] = canonical_hash(record)
            temporary = path.with_suffix(".pending")
            if temporary.exists():
                temporary.unlink()  # only this case's nonterminal publication buffer
            write_json(temporary, record)
            os.replace(temporary, path)
            records.append(record)
            acquired += 1
        if source_inventory()[1] != sources:
            raise RuntimeError("Source changed during acquisition; final publication refused")
        rows = _report(root, records, config)
        manifest = {"schema": 1, "purpose": "estimator_development_qualification",
            "case_count": len(cases), "fit_count": len(rows), "source_sha256": sources,
            "files": {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}}
        # A previous interrupted publication buffer is not scientific evidence.
        manifest["files"].pop("manifest.json.pending", None)
        publish_json(root / "manifest.json", manifest)
        return verify_qualification(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_qualification(args.verify)
    elif args.config and args.output:
        result = run_qualification(args.output, args.config, args.resume, args.limit_cases)
    else:
        parser.error("Use --verify, or provide --config and --output")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
