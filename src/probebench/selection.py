"""Source-bound global classical-parameter selection on separate wavefronts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from .budget_policies import ClassicalPolicy, METHODS, MLE_METHODS, PolicyConfig
from .cohorts import (generate_cohort, orbit_distances, pupil_gram, stream_seed,
                      IDENTIFIER)
from .environment import Episode, EpisodeConfig
from .optics import OpticalModel, OpticsConfig
from .policies import visible_score
from .qualification import canonical_hash, publish_json
from .selection_statistics import select_candidates
from .sequential import load_frame, read_episode, run_episode, summary_row
from .study import publish_manifest, study_directory, verify_study

PURPOSE = "classical_parameter_selection_v1"
TUNABLE = {"no_action_v2": (), "unprobed_mle_v1": (),
           "random_probe_v2": ("random_amplitude_nm",),
           "coordinate_search_v2": ("coordinate_step_nm",),
           "fixed_diversity_mle_v1": ("diversity_probe_nm",),
           "phase_diversity_mle_v1": ("diversity_probe_nm",)}
CASE_FIELDS = ("id", "family", "candidate_id", "base_wavefront_id", "stratum_id",
               "member", "replicate", "photons_per_exposure", "exposure_budget")
MAX_EPISODES = 100_000
MAX_EXPOSURES = 13


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"Unexpected {label} fields")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _numeric(value, name, *, zero=False):
    if (type(value) not in (int, float) or not np.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def policy_configuration(config, candidate):
    return {**config["common_policy"], **candidate["parameters"]}


def validate_candidates(config):
    common = config["common_policy"]
    _keys(common, asdict(PolicyConfig()), "common policy")
    for name, value in common.items():
        if name == "max_mle_iterations":
            _positive_int(value, name)
        else:
            _numeric(value, name)
    PolicyConfig(**common)
    candidates = config["candidates"]
    if type(candidates) is not list or not 1 <= len(candidates) <= 32:
        raise ValueError("Declare between one and 32 candidates")
    identifiers, effective = set(), set()
    limit = config["cohort"]["specification"]["command_limit_nm"]
    for candidate in candidates:
        _keys(candidate, {"id", "family", "parameters"}, "candidate")
        name, family = candidate["id"], candidate["family"]
        if (type(name) is not str or IDENTIFIER.fullmatch(name) is None
                or name in identifiers):
            raise ValueError("Candidate identifiers must be valid and unique")
        if type(family) is not str or family not in METHODS:
            raise ValueError("Unknown classical family")
        _keys(candidate["parameters"], TUNABLE[family], "candidate parameter")
        for key, value in candidate["parameters"].items():
            _numeric(value, key)
            if value > limit:
                raise ValueError("Candidate amplitude exceeds the command limit")
        policy = policy_configuration(config, candidate)
        PolicyConfig(**policy)
        # Only active dimensions count toward a family's search allowance. Numeric
        # representation (40 versus 40.0) cannot manufacture a second candidate.
        key = (family, tuple(float(policy[name]) for name in TUNABLE[family]))
        if key in effective:
            raise ValueError("Duplicate effective candidate within a family")
        identifiers.add(name)
        effective.add(key)


def prepare(config):
    _keys(config, {"schema", "purpose", "cohort", "excluded_cohorts", "candidates",
                   "common_policy", "replicates", "photons_per_exposure", "exposure_budgets",
                   "read_noise_electrons", "detector_seed_hex", "policy_seed_hex",
                   "operating_reserve_bytes"}, "selection configuration")
    if type(config["schema"]) is not int or config["schema"] != 1 or config["purpose"] != PURPOSE:
        raise ValueError("Unsupported selection protocol")
    spec = config["cohort"]["specification"]
    if spec["split"] != "selection":
        raise ValueError("Parameter selection requires an explicit selection cohort")
    cohort = generate_cohort(config["cohort"])
    validate_candidates(config)
    _positive_int(config["replicates"], "replicates")
    for name in ("photons_per_exposure", "exposure_budgets"):
        values = config[name]
        if type(values) is not list or not values:
            raise ValueError(f"A nonempty {name} list is required")
        for value in values:
            _positive_int(value, name)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {name}")
    if max(config["exposure_budgets"]) > MAX_EXPOSURES:
        raise ValueError(f"Selection v1 supports at most {MAX_EXPOSURES} exposures per episode")
    _numeric(config["read_noise_electrons"], "read noise", zero=True)
    reserve = config["operating_reserve_bytes"]
    if type(reserve) is not int or reserve < 0:
        raise ValueError("Operating reserve must be a nonnegative integer")
    for domain in ("detector", "policy"):
        stream_seed(config[f"{domain}_seed_hex"], domain, {})
    total = (cohort["member_count"] * config["replicates"] * len(config["candidates"])
             * len(config["photons_per_exposure"]) * len(config["exposure_budgets"]))
    if total > MAX_EPISODES:
        raise ValueError(f"Selection supports at most {MAX_EPISODES} episodes")
    exclusions = config["excluded_cohorts"]
    if type(exclusions) is not list or len(exclusions) > 32:
        raise ValueError("Declare at most 32 additional exclusion cohorts")
    model = OpticalModel(OpticsConfig(**spec["optics"]))
    gram = pupil_gram(model)
    exclusion_summary = []
    for source in exclusions:
        _keys(source, {"configuration", "clusters_sha256"}, "exclusion cohort source")
        excluded = source["configuration"]
        if excluded["specification"]["optics"] != spec["optics"]:
            raise ValueError("Exclusion cohort uses different optics")
        other = generate_cohort(excluded)
        if canonical_hash(other["clusters"]) != source["clusters_sha256"]:
            raise ValueError("Previously exposed cohort fingerprint changed")
        excluded_ids = {base["base_wavefront_id"] for base in other["clusters"]}
        if any(base["base_wavefront_id"] in excluded_ids for base in cohort["clusters"]):
            raise ValueError("Selection cohort reuses a previously exposed base identity")
        vectors = [r["members"][0]["truth_opd_nm"] for r in other["clusters"]]
        tolerance = max(spec["overlap_tolerance_nm"],
                        excluded["specification"]["overlap_tolerance_nm"])
        for base in cohort["clusters"]:
            if orbit_distances(base["members"][0]["truth_opd_nm"], vectors, gram).min() <= tolerance:
                raise ValueError("Selection cohort overlaps previously exposed wavefronts")
        exclusion_summary.append({"configuration_sha256": canonical_hash(excluded),
                                  "cohort_sha256": canonical_hash(other),
                                  "base_count": other["base_count"]})
    return {"cohort": cohort, "exclusion_cohorts": exclusion_summary,
            "expected_episodes": total}


def cases_for(config, prepared=None):
    prepared = prepare(config) if prepared is None else prepared
    cases = []
    truth_index = 0
    for base in prepared["cohort"]["clusters"]:
        for member in base["members"]:
            for photons in sorted(config["photons_per_exposure"]):
                for replicate in range(config["replicates"]):
                    stream_identity = {"protocol": PURPOSE,
                        "base_wavefront_id": base["base_wavefront_id"], "member": member["member"],
                        "replicate": replicate, "photons_per_exposure": photons}
                    seeds = {f"{domain}_seed": stream_seed(config[f"{domain}_seed_hex"], domain,
                                                           stream_identity)
                             for domain in ("detector", "policy")}
                    for budget in sorted(config["exposure_budgets"]):
                        for candidate in sorted(config["candidates"], key=lambda row: row["id"]):
                            case = {**stream_identity, **seeds, "stratum_id": base["stratum_id"],
                                    "truth_opd_nm": member["truth_opd_nm"], "truth_index": truth_index,
                                    "family": candidate["family"], "method": candidate["family"],
                                    "candidate_id": candidate["id"], "exposure_budget": budget}
                            # Full digests, rather than positions or truncated keys, identify files.
                            identity = {key: case[key] for key in CASE_FIELDS if key != "id"}
                            case["id"] = "case_" + canonical_hash(identity)
                            cases.append(case)
            truth_index += 1
    return cases


def episode_study(study, case):
    config = study["configuration"]
    candidate = next(c for c in config["candidates"] if c["id"] == case["candidate_id"])
    spec = config["cohort"]["specification"]
    # Reuse acquisition without changing the development runner's public purpose.
    # Every frame binds the entire selection study and the effective candidate policy.
    return {**study, "selection_study_sha256": canonical_hash(study),
            "candidate_id": candidate["id"], "configuration": {
                "optics": spec["optics"], "command_limit_nm": spec["command_limit_nm"],
                "read_noise_electrons": config["read_noise_electrons"],
                "policy": policy_configuration(config, candidate)}}


def restore_policy_fits(policy, records, history, *, final=False):
    """Require the exact fit schedule and the declared camera-only failure fallback."""
    count = len(history)
    if policy.name == "phase_diversity_mle_v1":
        expected = list(range(2, count + 1))
    elif final and policy.name in MLE_METHODS and count >= 2:
        expected = [count]
    else:
        expected = []
    if (type(records) is not list
            or [r.get("observations_used") for r in records] != expected):
        raise ValueError("Retained fits disagree with the declared policy schedule")
    policy.restore_fit_records(records, history)
    for record in records:
        _numeric(record["fit_elapsed_seconds"], "fit duration", zero=True)
        if type(record["optimizer_success"]) is not bool:
            raise ValueError("Retained optimizer success must be a boolean")
        if record["fit_status"] == "numerical_failure":
            best = max(history[:record["observations_used"]], key=visible_score)
            if (record.get("fallback") != "best_measured_camera_score"
                or type(record.get("reason")) is not str or not record["reason"].strip()
                or record["optimizer_success"]
                or not np.array_equal(record["estimated_opd_nm"], best.applied_opd_nm)):
                raise ValueError("Retained numerical failure disagrees with camera-score fallback")


def replay_frames(root, study, case, count):
    """Replay a contiguous retained prefix without acquiring or fitting new data."""
    config = study["configuration"]
    model = OpticalModel(OpticsConfig(**config["optics"]))
    instrument = EpisodeConfig(max_exposures=case["exposure_budget"],
        photon_budget=case["photons_per_exposure"] * case["exposure_budget"],
        photons_per_exposure=case["photons_per_exposure"],
        command_limit_nm=config["command_limit_nm"], read_noise_electrons=config["read_noise_electrons"])
    episode = Episode(model, instrument, case["truth_opd_nm"], case["detector_seed"])
    policy = ClassicalPolicy(case["method"], model, instrument,
                             PolicyConfig(**config["policy"]), case["policy_seed"])
    binding = canonical_hash({"study": canonical_hash(study), "case": case})
    history = []
    pixels = model.config.pupil_pixels * model.config.oversampling
    for index in range(count):
        saved, image = load_frame(root / "frames" / case["id"] / f"frame_{index:04d}.npz",
                                  binding, index, case, (pixels, pixels))
        restore_policy_fits(policy, saved["fit_records"], history)
        command = policy.command(tuple(history))
        if not np.array_equal(command, saved["observation"]["applied_opd_nm"]):
            raise ValueError("Retained command disagrees with declared candidate policy")
        measured = episode.expose(command)
        if measured.metadata() != saved["observation"] or not np.array_equal(measured.image_electrons, image):
            raise ValueError("Retained image disagrees with detector replay")
        history.append(measured)
    return episode, policy, history


def verify_policy_record(root, study, case, record):
    """Replay detector draws and commands from saved fits, without fitting again."""
    episode, policy, history = replay_frames(root, study, case, case["exposure_budget"])
    result = record["result"]
    restore_policy_fits(policy, result["fit_records"], history, final=True)
    replayed = episode.submit(policy.final_command(tuple(history)))
    if any(result.get(key) != value for key, value in replayed.items()):
        raise ValueError("Retained evaluator state disagrees with complete policy replay")


def read_selection_episode(root, study, case):
    derived = episode_study(study, case)
    record = read_episode(root, derived, case)
    verify_policy_record(root, derived, case, record)
    return record


def payload_size(path):
    """Bound credit by the canonical payload, excluding JSON/ZIP container padding."""
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        return len((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            record = json.loads(data["record_json"].item())
            payload = BytesIO()
            np.savez_compressed(payload, image_electrons=data["image_electrons"],
                record_json=np.array(json.dumps(record, sort_keys=True, allow_nan=False)),
                record_sha256=np.array(canonical_hash(record)))
        return payload.tell()
    return path.stat().st_size


def retained_credit(root, study, cases, prepared):
    """Credit only verified retained files; unfinished publication buffers earn none."""
    verified = {".lock", "study.json", *("source/" + name for name in study["source_sha256"])}
    pending = set()
    if (root / "preparation.json").exists():
        if json.loads((root / "preparation.json").read_text()) != prepared:
            raise ValueError("Preparation changed during selection resumption")
        verified.add("preparation.json")
    else:
        pending.add("preparation.json.pending")
    records = []
    for case in cases:
        record_name = f'cases/{case["id"]}.json'
        directory = root / "frames" / case["id"]
        frames = sorted(p.name for p in directory.glob("*.npz"))
        count = len(frames)
        if (count > case["exposure_budget"]
                or frames != [f"frame_{i:04d}.npz" for i in range(count)]):
            raise ValueError("Saved exposures must form a contiguous prefix")
        if (root / record_name).exists():
            records.append(read_selection_episode(root, study, case))
            verified.add(record_name)
        else:
            replay_frames(root, episode_study(study, case), case, count)
            if count == case["exposure_budget"]:
                pending.add(record_name + ".pending")
        verified.update(f'frames/{case["id"]}/{name}' for name in frames)
        if count < case["exposure_budget"]:
            pending.add(f'frames/{case["id"]}/frame_{count:04d}.pending')
    derived = selection_outputs(study, cases, records) if len(records) == len(cases) else {}
    for name in ("summary.json", "selection.json", "selected-settings.json"):
        if (root / name).exists():
            if name not in derived or json.loads((root / name).read_text()) != derived[name]:
                raise ValueError("Premature or changed selection output")
            verified.add(name)
        elif name in derived:
            pending.add(name + ".pending")
    if len(records) == len(cases):
        pending.add("manifest.json.pending")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if not actual <= verified | pending:
        raise ValueError("Unexpected file in selection study")
    credit, inodes = 0, set()
    for name in (actual & verified) - {".lock"}:
        stat = (root / name).stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in inodes:
            credit += min(stat.st_size, stat.st_blocks * 512, payload_size(root / name))
            inodes.add(identity)
    return credit


def selection_outputs(study, cases, records):
    rows = []
    for case, record in zip(cases, records, strict=True):
        metrics = summary_row(record)
        rows.append({**metrics, **{key: case[key] for key in CASE_FIELDS},
                     "outcome_status": "policy_failure" if metrics["failed_fits"] else "completed",
                     "failure_reason": "numerical_fit_fallback" if metrics["failed_fits"] else None})
    expected = [{key: case[key] for key in CASE_FIELDS} for case in cases]
    outcomes = [{"case_id": row["id"], **{key: row[key] for key in
                 ("residual_wfe_nm", "outcome_status", "failure_reason")}} for row in rows]
    selection = select_candidates(expected, outcomes)
    config = study["configuration"]
    selected = {"schema": 1, "purpose": PURPOSE, "study_sha256": canonical_hash(study),
                "selection_sha256": canonical_hash(selection), "families": {}}
    for family, identity in selection["selected_candidates"].items():
        candidate = next(c for c in config["candidates"] if c["id"] == identity)
        selected["families"][family] = {"candidate_id": identity,
                                        "policy": policy_configuration(config, candidate)}
    return {"summary.json": rows, "selection.json": selection, "selected-settings.json": selected}


def storage_estimate(config, prepared=None):
    prepared = prepare(config) if prepared is None else prepared
    spec = config["cohort"]["specification"]
    pixels = spec["optics"]["pupil_pixels"] * spec["optics"]["oversampling"]
    exposures = (prepared["cohort"]["member_count"] * config["replicates"] * len(config["candidates"])
                 * len(config["photons_per_exposure"]) * sum(config["exposure_budgets"]))
    raw_bytes = exposures * pixels**2 * 8
    # Per-case metadata and retained fit diagnostics need admission even at tiny grids.
    retained_bytes = raw_bytes + prepared["expected_episodes"] * 64 * 1024
    return {"episodes": prepared["expected_episodes"], "exposures": exposures,
            "raw_image_bytes": raw_bytes, "retained_bytes_for_admission": retained_bytes,
            "required_free_bytes": config["operating_reserve_bytes"] + 3 * retained_bytes + 16 * 1024**2}


def verify_run(root):
    root = Path(root)
    manifest, study = verify_study(root, PURPOSE)
    prepared = prepare(study["configuration"])
    cases = cases_for(study["configuration"], prepared)
    records = [read_selection_episode(root, study, case) for case in cases]
    expected_files = {".lock", "study.json", "preparation.json", "summary.json",
                      "selection.json", "selected-settings.json",
                      *("source/" + name for name in study["source_sha256"])}
    for case in cases:
        expected_files.add(f'cases/{case["id"]}.json')
        expected_files.update(f'frames/{case["id"]}/frame_{i:04d}.npz'
                              for i in range(case["exposure_budget"]))
    if set(manifest["files"]) != expected_files:
        raise ValueError("Unexpected file in completed selection study")
    artifacts = {"preparation.json": prepared, **selection_outputs(study, cases, records)}
    for filename, expected in artifacts.items():
        if json.loads((root / filename).read_text()) != expected:
            raise ValueError(f"Selection artifact disagrees with retained evidence: {filename}")
    estimate = storage_estimate(study["configuration"], prepared)
    if manifest.get("episode_count") != len(cases) or manifest.get("exposure_count") != estimate["exposures"]:
        raise ValueError("Selection manifest counts disagree")
    return manifest


def run(root, config_path, resume=False, limit_episodes=None):
    if limit_episodes is not None:
        _positive_int(limit_episodes, "limit_episodes")
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    prepared = prepare(config)
    cases = cases_for(config, prepared)
    estimate = storage_estimate(config, prepared)
    if (root / "manifest.json").exists():
        verify_run(root)  # Completed archives are never silently repaired.
    with study_directory(root, config, estimate["retained_bytes_for_admission"], resume,
                         resume_credit=lambda directory, bound:
                             retained_credit(directory, bound, cases, prepared)) as study:
        if (root / "manifest.json").exists():
            return verify_run(root)
        preparation_path = root / "preparation.json"
        if preparation_path.exists():
            if json.loads(preparation_path.read_text()) != prepared:
                raise ValueError("Preparation changed during selection resumption")
        else:
            publish_json(preparation_path, prepared)
        for name in ("cases", "frames"):
            (root / name).mkdir(exist_ok=True)
        records, completed = [], 0
        for case in cases:
            if (root / "cases" / f'{case["id"]}.json').exists():
                records.append(read_selection_episode(root, study, case))
                continue
            if limit_episodes is not None and completed >= limit_episodes:
                return {"complete": False, "completed_episodes": len(records),
                        "expected_episodes": len(cases)}
            run_episode(root, episode_study(study, case), case)
            # A returned acquisition is not yet verified evidence. Validate it before
            # it can contribute a loss or authorize publication of selected settings.
            records.append(read_selection_episode(root, study, case))
            completed += 1
            print(f'Completed {case["id"]}: {case["candidate_id"]}', flush=True)
        for filename, content in selection_outputs(study, cases, records).items():
            path = root / filename
            if path.exists():
                if json.loads(path.read_text()) != content:
                    raise ValueError("Existing selection output differs from retained evidence")
            else:
                publish_json(path, content)
        publish_manifest(root, study, PURPOSE, episode_count=len(cases), exposure_count=estimate["exposures"])
    return verify_run(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.config or args.output or args.resume or args.limit_episodes is not None or args.estimate:
            parser.error("--verify is exclusive")
        result = verify_run(args.verify)
    elif args.estimate:
        if not args.config or args.output or args.resume or args.limit_episodes is not None:
            parser.error("--estimate requires only --config")
        result = storage_estimate(json.loads(args.config.read_text()))
    elif args.config and args.output:
        result = run(args.output, args.config, args.resume, args.limit_episodes)
    else:
        parser.error("Provide --config with --output or --estimate, or --verify")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True))


if __name__ == "__main__":
    main()
