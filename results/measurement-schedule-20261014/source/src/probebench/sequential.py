"""Retained sequential-calibration experiments with per-exposure checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .budget_policies import ClassicalPolicy, METHODS, PolicyConfig
from .environment import Episode, EpisodeConfig, Observation
from .optics import OpticalModel, OpticsConfig, coefficients
from .qualification import canonical_hash, publish_json, publish_text
from .run import digest, source_inventory
from .study import publish_manifest, study_directory, verify_study

PURPOSE = "sequential_calibration_development"


def cases_for(config):
    if config.get("purpose") != "public_development_only":
        raise ValueError("This runner accepts development fixtures only")
    if not config["methods"] or len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("Methods must be nonempty and unique")
    if any(m not in METHODS for m in config["methods"]):
        raise ValueError("Unknown policy")
    for key in ("replicates", "detector_seed_root", "policy_seed_root"):
        if type(config[key]) is not int or config[key] < (1 if key == "replicates" else 0):
            raise ValueError(f"Invalid integer {key}")
    for key in ("exposure_budgets", "photons_per_exposure"):
        values = config[key]
        if (not values or any(type(v) is not int or v <= 0 for v in values)
                or values != sorted(set(values))):
            raise ValueError(f"{key} must contain increasing positive integers")
    OpticsConfig(**config["optics"])
    PolicyConfig(**config["policy"])
    cases = []
    for photons in config["photons_per_exposure"]:
        for budget in config["exposure_budgets"]:
            for truth_index, truth in enumerate(config["truths_opd_nm"]):
                coefficients(truth)
                for replicate in range(config["replicates"]):
                    # Pair wavefront/noise identities across methods and budget settings.
                    stream = truth_index*config["replicates"] + replicate
                    for method in config["methods"]:
                        case = {"id": f"episode_{len(cases):04d}", "method": method,
                            "truth_index": truth_index, "truth_opd_nm": truth,
                            "replicate": replicate, "exposure_budget": budget,
                            "photons_per_exposure": photons,
                            "detector_seed": config["detector_seed_root"]+stream,
                            "policy_seed": config["policy_seed_root"]+stream}
                        episode_config(config, case)
                        cases.append(case)
    if not cases:
        raise ValueError("No development cases")
    return cases


def episode_config(config, case):
    return EpisodeConfig(max_exposures=case["exposure_budget"],
        photon_budget=case["exposure_budget"]*case["photons_per_exposure"],
        photons_per_exposure=case["photons_per_exposure"],
        read_noise_electrons=config["read_noise_electrons"], command_limit_nm=config["command_limit_nm"])


def save_frame(path, observation, binding, fit_records, decision_seconds):
    if path.exists():
        raise FileExistsError("A retained exposure cannot be replaced")
    image = observation.image_electrons
    record = {"case_binding": binding, "observation": observation.metadata(),
        "fit_records": fit_records, "decision_seconds": decision_seconds,
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as output:
        np.savez_compressed(output, image_electrons=image,
            record_json=np.array(json.dumps(record, sort_keys=True, allow_nan=False)),
            record_sha256=np.array(canonical_hash(record)))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_frame(path, binding, index, case, image_shape):
    if path.is_symlink():
        raise ValueError("Exposure paths must not be symbolic links")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"image_electrons", "record_json", "record_sha256"}:
            raise ValueError("Unexpected exposure schema")
        record = json.loads(data["record_json"].item())
        image = data["image_electrons"].copy()
        if (canonical_hash(record) != data["record_sha256"].item()
            or record["image_sha256"] != hashlib.sha256(image.tobytes()).hexdigest()
            or record["case_binding"] != binding or image.shape != image_shape
            or image.dtype != np.dtype("float64") or not np.isfinite(image).all()):
            raise ValueError("Changed exposure evidence or binding")
    observed = record["observation"]
    if (observed["index"] != index or observed["expected_photons"] != case["photons_per_exposure"]
        or observed["cumulative_expected_photons"] != (index+1)*case["photons_per_exposure"]
        or not np.isfinite(record["decision_seconds"]) or record["decision_seconds"] < 0):
        raise ValueError("Invalid exposure accounting")
    coefficients(observed["applied_opd_nm"])
    return record, image


def run_episode(root, study, case):
    config = study["configuration"]
    model = OpticalModel(OpticsConfig(**config["optics"]))
    instrument = episode_config(config, case)
    episode = Episode(model, instrument, case["truth_opd_nm"], case["detector_seed"])
    policy = ClassicalPolicy(case["method"], model, instrument,
                             PolicyConfig(**config["policy"]), case["policy_seed"])
    binding = canonical_hash({"study": canonical_hash(study), "case": case})
    frames = root / "frames" / case["id"]
    if frames.is_symlink():
        raise ValueError("Exposure directory must not be a symbolic link")
    frames.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in frames.glob("*.npz"))
    if existing != [f"frame_{i:04d}.npz" for i in range(len(existing))] or len(existing) > instrument.max_exposures:
        raise ValueError("Saved exposures must form a contiguous prefix")
    history, decision_times = [], []
    pixels = model.config.pupil_pixels*model.config.oversampling
    for index in range(instrument.max_exposures):
        path = frames / f"frame_{index:04d}.npz"
        if path.exists():
            saved, measured = load_frame(path, binding, index, case, (pixels, pixels))
            policy.restore_fit_records(saved["fit_records"], history)
            # Deterministic simulation replay restores RNG/evaluator state. No stored
            # exposure is replaced and no additional observation is given to the policy.
            observation = episode.expose(saved["observation"]["applied_opd_nm"])
            if (observation.metadata() != saved["observation"]
                or not np.array_equal(observation.image_electrons, measured)):
                raise ValueError("Saved exposure disagrees with deterministic simulation replay")
            decision_times.append(saved["decision_seconds"])
        else:
            started = perf_counter()
            command = policy.command(tuple(history))
            elapsed = perf_counter()-started
            observation = episode.expose(command)
            save_frame(path, observation, binding, policy.fit_records, elapsed)
            decision_times.append(elapsed)
        history.append(observation)
    started = perf_counter()
    command = policy.final_command(tuple(history))
    final_seconds = perf_counter()-started
    result = episode.submit(command)
    result.update({"fit_records": policy.fit_records, "decision_seconds": decision_times,
                   "submission_seconds": final_seconds})
    record = {"case": case, "study_binding": canonical_hash(study), "result": result,
              "frame_sha256": {p.name: digest(p) for p in sorted(frames.glob("*.npz"))}}
    record["record_sha256"] = canonical_hash(record)
    if source_inventory()[1] != study["source_sha256"]:
        raise RuntimeError("Source changed during episode; publication refused")
    publish_json(root / "cases" / f'{case["id"]}.json', record)
    return record


def read_episode(root, study, case):
    record_path = root / "cases" / f'{case["id"]}.json'
    if record_path.is_symlink():
        raise ValueError("Episode records must not be symbolic links")
    record = json.loads(record_path.read_text())
    signed = dict(record)
    recorded_hash = signed.pop("record_sha256")
    if canonical_hash(signed) != recorded_hash or record["case"] != case or record["study_binding"] != canonical_hash(study):
        raise ValueError("Changed episode evidence")
    expected = {f"frame_{i:04d}.npz" for i in range(case["exposure_budget"])}
    if set(record["frame_sha256"]) != expected:
        raise ValueError("Incomplete episode exposures")
    for name, expected_hash in record["frame_sha256"].items():
        frame_path = root / "frames" / case["id"] / name
        if (frame_path.is_symlink() or not frame_path.resolve().is_relative_to(root.resolve())
            or digest(frame_path) != expected_hash):
            raise ValueError("Changed episode exposure")
    result = record["result"]
    if (result["exposures"] != case["exposure_budget"]
        or result["expected_photons_used"] != case["exposure_budget"]*case["photons_per_exposure"]):
        raise ValueError("Incorrect episode resource accounting")
    validate_physical_record(root, study, case, result)
    return record


def validate_physical_record(root, study, case, result):
    config = study["configuration"]
    model = OpticalModel(OpticsConfig(**config["optics"]))
    instrument = episode_config(config, case)
    policy = ClassicalPolicy(case["method"], model, instrument,
                             PolicyConfig(**config["policy"]), case["policy_seed"])
    binding = canonical_hash({"study": canonical_hash(study), "case": case})
    pixels = model.config.pupil_pixels*model.config.oversampling
    history, measured_indices, peak_wfe = [], [], []
    final = coefficients(result["final_applied_opd_nm"])
    if (np.max(np.abs(final)) > instrument.command_limit_nm
        or result["hidden_opd_nm"] != case["truth_opd_nm"]
        or len(result["trajectory"]) != case["exposure_budget"]
        or len(result["decision_seconds"]) != case["exposure_budget"]
        or not np.isfinite(result["submission_seconds"]) or result["submission_seconds"] < 0):
        raise ValueError("Invalid evaluator state")
    for index in range(case["exposure_budget"]):
        saved, image = load_frame(root/"frames"/case["id"]/f"frame_{index:04d}.npz",
                                  binding,index,case,(pixels,pixels))
        meta = saved["observation"]
        command = coefficients(meta["applied_opd_nm"])
        if np.max(np.abs(command)) > instrument.command_limit_nm:
            raise ValueError("Recorded command exceeds instrument limits")
        policy.restore_fit_records(saved["fit_records"], history)
        if (result["fit_records"][:len(saved["fit_records"])] != saved["fit_records"]
            or saved["decision_seconds"] != result["decision_seconds"][index]):
            raise ValueError("Inconsistent policy checkpoint")
        history.append(Observation(index,tuple(command),meta["expected_photons"],
                                   meta["cumulative_expected_photons"],image))
        trajectory = result["trajectory"][index]
        metrics = model.evaluate(np.array(case["truth_opd_nm"])-command)
        if (any(trajectory.get(key) != value for key,value in meta.items())
            or any(not np.isclose(trajectory[key],value,rtol=1e-12,atol=1e-12) for key,value in metrics.items())
            or type(trajectory["realized_photoelectrons"]) is not int
            or trajectory["realized_photoelectrons"] < 0):
            raise ValueError("Trajectory disagrees with physical state or exposure metadata")
        peak_wfe.append(metrics["residual_wfe_nm"])
        if np.array_equal(command,final):
            measured_indices.append(index)
    policy.restore_fit_records(result["fit_records"], history)
    for name, residual in (("initial", np.array(case["truth_opd_nm"])),
                           ("final", np.array(case["truth_opd_nm"])-final)):
        if any(not np.isclose(result[name][key],value,rtol=1e-12,atol=1e-12)
               for key,value in model.evaluate(residual).items()):
            raise ValueError("Evaluator metrics disagree with physical state")
    if (result["final_command_observation_indices"] != measured_indices
        or result["final_command_has_camera_observation"] != bool(measured_indices)
        or result["final_command_is_last_observation"] != (case["exposure_budget"]-1 in measured_indices)
        or not np.array_equal(result["final_mirror_surface_coefficients_nm"],final/2)
        or not np.isclose(result["peak_probe_wfe_increase_nm"],
                          max(0,max(peak_wfe)-result["initial"]["residual_wfe_nm"]),atol=1e-12)):
        raise ValueError("Final measurement status, OPD units or probe penalty is inconsistent")


def summary_row(record):
    case, result = record["case"], record["result"]
    fits = result["fit_records"]
    return {"episode_id": case["id"], "method": case["method"],
        "truth_index": case["truth_index"], "replicate": case["replicate"],
        "photons_per_exposure": case["photons_per_exposure"], "exposures": result["exposures"],
        "expected_photons_used": result["expected_photons_used"], **result["final"],
        "initial_wfe_nm": result["initial"]["residual_wfe_nm"],
        "peak_probe_wfe_increase_nm": result["peak_probe_wfe_increase_nm"],
        "final_command_has_camera_observation": result["final_command_has_camera_observation"],
        "fit_count": len(fits),
        "failed_fits": sum(f["fit_status"] == "numerical_failure" for f in fits),
        "unconverged_fits": sum(f["fit_status"] == "finite" and not f["optimizer_success"] for f in fits),
        "decision_seconds": sum(result["decision_seconds"])+result["submission_seconds"]}


def report_text(rows, config):
    lines = ["# Sequential optical calibration under exposure budgets", "",
        "Known development wavefronts; static, monochromatic, exactly matched plant and estimator.",
        "Final corrections may be submitted without another exposure. Evaluator WFE and Strehl",
        "for an unmeasured correction must not be presented as a measured corrected image.", "",
        "| Photons/exposure | Exposures | Method | Episodes | Mean WFE (nm) | Mean Strehl | Mean peak probe increase (nm) | Episodes with failed fits | Episodes with unconverged fits |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for photons in config["photons_per_exposure"]:
        for budget in config["exposure_budgets"]:
            for method in config["methods"]:
                selected = [r for r in rows if (r["photons_per_exposure"],r["exposures"],r["method"]) == (photons,budget,method)]
                mean = lambda key: np.mean([r[key] for r in selected])
                lines.append(f'| {photons} | {budget} | {method} | {len(selected)} | '
                    f'{mean("residual_wfe_nm"):.4f} | {mean("strehl_peak_ratio"):.6f} | '
                    f'{mean("peak_probe_wfe_increase_nm"):.4f} | '
                    f'{sum(r["failed_fits"] > 0 for r in selected)} | {sum(r["unconverged_fits"] > 0 for r in selected)} |')
    lines += ["", "Every episode is included. A numerical fit failure uses the best measured camera-score",
        "command as a declared fallback and remains counted above. Finite nonconvergence remains",
        "visible. Small related fixture sets and shared noise streams do not support population",
        "rankings, significance claims or treating budget points as independent replicates.", "",
        "Coordinate search completes its first six-mode sweep at 13 exposures. The phase-diversity",
        "reference submits zero after one exposure, then uses known probes and all available frames.",
        "Reported decision durations are recorded local execution costs, not real-time guarantees.", ""]
    if {"unprobed_mle_v1", "fixed_diversity_mle_v1"}.intersection(config["methods"]):
        lines += ["The schedule controls retain the same estimator. Unprobed estimation acquires",
            "only zero-command images; its selected minimum is not an identified wavefront.",
            "Fixed diversity repeats 0,+d,-d defocus commands without using camera feedback",
            "during acquisition. Both fit all available images once at final submission;",
            "all three estimator policies submit zero with only one exposure.", ""]
    return "\n".join(lines)


def verify_run(root):
    root = Path(root)
    manifest, study = verify_study(root, PURPOSE)
    cases = cases_for(study["configuration"])
    rows = [summary_row(read_episode(root, study, case)) for case in cases]
    if (manifest["episode_count"] != len(cases)
        or manifest["exposure_count"] != sum(c["exposure_budget"] for c in cases)
        or json.loads((root / "summary.json").read_text()) != rows
        or (root / "report.md").read_text() != report_text(rows, study["configuration"])):
        raise ValueError("Derived output disagrees with episode records")
    return manifest


def run(root, config_path, resume=False, limit_episodes=None):
    if limit_episodes is not None and (type(limit_episodes) is not int or limit_episodes <= 0):
        raise ValueError("limit_episodes must be a positive integer")
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    cases = cases_for(config)
    pixels = config["optics"]["pupil_pixels"]*config["optics"]["oversampling"]
    raw_bytes = sum(c["exposure_budget"] for c in cases)*pixels**2*8
    with study_directory(root, config, raw_bytes, resume) as study:
        if (root / "manifest.json").exists():
            return verify_run(root)
        for name in ("cases", "frames"):
            if (root/name).is_symlink():
                raise ValueError("Evidence directories must not be symbolic links")
            (root/name).mkdir(exist_ok=True)
        records, completed = [], 0
        for case in cases:
            if (root / "cases" / f'{case["id"]}.json').exists():
                records.append(read_episode(root, study, case))
                continue
            if limit_episodes is not None and completed >= limit_episodes:
                return {"complete": False, "completed_episodes": len(records)}
            records.append(run_episode(root, study, case))
            completed += 1
            print(f'Completed {case["id"]}: {case["method"]}, {case["exposure_budget"]} exposures', flush=True)
        rows = [summary_row(r) for r in records]
        for name, content in (("summary.json",json.dumps(rows,sort_keys=True,indent=2,allow_nan=False)+"\n"),
                              ("report.md",report_text(rows,config))):
            path = root/name
            if path.exists() and path.read_text() != content:
                raise ValueError("Existing derived artifact disagrees with retained episodes")
            if not path.exists():
                publish_text(path, content)
        publish_manifest(root, study, PURPOSE, episode_count=len(cases),
                         exposure_count=sum(r["exposures"] for r in rows))
        return verify_run(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_run(args.verify)
    elif args.config and args.output:
        result = run(args.output,args.config,args.resume,args.limit_episodes)
    else:
        parser.error("Provide --verify, or --config and --output")
    print(json.dumps({k:v for k,v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
