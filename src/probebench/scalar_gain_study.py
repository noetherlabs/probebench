"""Bounded, source-frozen scalar-response development study and retained-path verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import scipy

from .budget_policies import ClassicalPolicy, PolicyConfig
from .cohorts import (
    generate_cohort,
    orbit_distances,
    pupil_gram,
    stream_seed,
    validate_registry,
)
from .environment import EpisodeConfig, Observation
from .nominal_calibration import (
    GainEpisode,
    NumericalObservationError,
    run_classical_episode,
)
from .optics import OpticalModel, OpticsConfig, coefficients
from .qualification import canonical_hash, publish_json
from .run import digest, source_inventory
from .selection import restore_policy_fits
from .study import publish_manifest, study_directory, verify_study

PURPOSE = "scalar_gain_development_v1"
COHORT_ID = "nominal_calibration_development_v1"
HISTORICAL_CONFIG = "5c604249fd55fbe713c54ebf9f1e693e968f4b657821316ccfbca065ad44c402"
COMPLETED_CLUSTERS = "a4bbd6ba53379d308b88a2fda890fa8617725a0c71e06c6564a513c5c6929647"
SELECTED_SETTINGS = "c858028008125b277bb6f9dcf1696e9ada1e0b5bada1ab19a577688cb0e50cab"
ARMS = ("fixed_diversity_mle_v1", "phase_diversity_mle_v1")
GAINS = (0.9, 1.0, 1.1)
STRATA = (("low", 10.0, 25.0), ("medium", 25.0, 50.0), ("high", 50.0, 90.0))
RESERVE = 1024**3
JSON_LIMIT = 4 * 1024**2
CASE_JSON_LIMIT = 64 * 1024
CONTAINER_ALLOWANCE = 2048
GLOBAL_ALLOWANCE = 16 * 1024**2
RESAMPLES = 10_000
DOMAINS = ("wavefront", "detector", "policy", "bootstrap")


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"Unexpected {label} fields")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _parse(raw):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def _read_json(path, limit=JSON_LIMIT):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("Unsafe or oversized JSON artifact")
    return _parse(path.read_bytes())


def _hex(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "machine": platform.machine()}


def committed_source():
    """Require every executing source file to equal a file in the current commit."""
    root, inventory = source_inventory()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for name, expected in inventory.items():
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError("Executing source is not the committed source")
    return revision, inventory


def seed_commitments(config):
    seeds = {"wavefront": config["cohort"]["seed_hex"],
             **{d: config[f"{d}_seed_hex"] for d in DOMAINS[1:]}}
    if any(not _hex(value) for value in seeds.values()) or len(set(seeds.values())) != 4:
        raise ValueError("Four distinct 256-bit domain keys are required")
    return {domain: hashlib.sha256(bytes.fromhex(value)).hexdigest()
            for domain, value in seeds.items()}


def exclusion_inventory(config):
    historical = config["historical_selection_config"]
    return {"registry": config["cohort"]["exclusions"],
            "required_cohorts": [*historical["excluded_cohorts"],
                {"configuration": historical["cohort"], "clusters_sha256": COMPLETED_CLUSTERS}],
            "additional_cohorts": config["additional_prior_cohorts"],
            "cross_grid_fixtures": config["cross_grid_fixtures"],
            "reviewed_fixture_sources": config["reviewed_fixture_sources"]}


def map_pupil_coefficients(value, source_optics, target_optics):
    """Preserve the analytic OPD polynomial modulo piston across finite grids.

    The same six raw real Zernikes are normalized separately on each pupil.
    Consequently a_target = a_source * raw_RMS_target / raw_RMS_source.
    This is not interpolation of a measured wavefront or a changed aperture.
    """
    def scales(optics):
        specification = OpticsConfig(**optics)
        n = specification.pupil_pixels
        if n > 256 or n * specification.oversampling > 1024:
            raise ValueError("Cross-grid exclusion exceeds bounded optical dimensions")
        x = 2 * (np.arange(n) - (n - 1) / 2) / n
        xx, yy = np.meshgrid(x, x)
        r2 = xx**2 + yy**2
        mask = r2 <= 1
        raw = np.stack((np.sqrt(3) * (2*r2-1), np.sqrt(6)*(xx**2-yy**2),
            np.sqrt(6)*2*xx*yy, np.sqrt(8)*(3*r2-2)*xx,
            np.sqrt(8)*(3*r2-2)*yy, np.sqrt(5)*(6*r2**2-6*r2+1)))[:, mask]
        return np.sqrt(np.mean((raw - raw.mean(axis=1, keepdims=True))**2, axis=1))
    return coefficients(value) * scales(target_optics) / scales(source_optics)


def validate_configuration(config):
    """Inspect the fixed protocol without sampling any cohort or detector stream."""
    _keys(config, {"schema", "purpose", "cohort", "gains", "candidates", "exposure_budget",
        "photons_per_exposure", "read_noise_electrons", "historical_selection_config",
        "selected_settings_sha256", "additional_prior_cohorts", "cross_grid_fixtures", "reviewed_fixture_sources",
        "detector_seed_hex", "policy_seed_hex", "bootstrap_seed_hex", "seed_commitments",
        "operating_reserve_bytes"}, "scalar-gain configuration")
    if len(_json_bytes(config)) > JSON_LIMIT:
        raise ValueError("Configuration exceeds its byte allowance")
    if type(config["schema"]) is not int or config["schema"] != 1 or config["purpose"] != PURPOSE:
        raise ValueError("Unsupported scalar-gain protocol")
    expected_spec = {"schema": 1, "cohort_id": COHORT_ID, "split": "development",
        "optics": {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0},
        "strata": [{"id": n, "rms_min_nm": lo, "rms_max_nm": hi, "base_count": 8}
                    for n, lo, hi in STRATA],
        "command_limit_nm": 200.0, "overlap_tolerance_nm": 1e-6}
    cohort = config["cohort"]
    _keys(cohort, {"specification", "seed_hex", "exclusions", "operating_reserve_bytes"}, "cohort")
    if canonical_hash(cohort["specification"]) != canonical_hash(expected_spec):
        raise ValueError("This protocol requires exactly 24 declared development bases")
    expected_candidates = [{"id": identity, "family": family, "policy": asdict(PolicyConfig())}
        for identity, family in zip(("fixed_probe_080", "adaptive_probe_080"), ARMS, strict=True)]
    if canonical_hash(config["candidates"]) != canonical_hash(expected_candidates):
        raise ValueError("Frozen classical settings changed")
    for key, expected in (("gains", list(GAINS)), ("exposure_budget", 7),
                          ("photons_per_exposure", 2000), ("read_noise_electrons", 1.0)):
        if canonical_hash(config[key]) != canonical_hash(expected):
            raise ValueError(f"Fixed protocol field changed: {key}")
    if any(type(value) is not int or value < RESERVE for value in (
            config["operating_reserve_bytes"], cohort["operating_reserve_bytes"])):
        raise ValueError("The operating reserve cannot be below 1 GiB")
    historical = config["historical_selection_config"]
    if canonical_hash(historical) != HISTORICAL_CONFIG or config["selected_settings_sha256"] != SELECTED_SETTINGS:
        raise ValueError("Historical selection binding changed")
    model = OpticalModel(OpticsConfig(**expected_spec["optics"]))
    validate_registry(cohort["exclusions"], model)
    rows = cohort["exclusions"]["wavefronts"]
    hashes = {canonical_hash(row) for row in rows}
    if not {canonical_hash(row) for row in historical["cohort"]["exclusions"]["wavefronts"]} <= hashes:
        raise ValueError("Historical fixture exclusions are incomplete")
    for value in ([13.0, 0., 0., 0., 0., 0.], [31.0, 0., 0., 0., 0., 0.]):
        if not any(np.array_equal(row["truth_opd_nm"], value) for row in rows):
            raise ValueError("Export/replay fixtures must be excluded")
    additional = config["additional_prior_cohorts"]
    if type(additional) is not list or len(additional) > 32:
        raise ValueError("Additional cohort inventory must be a bounded list")
    for entry in additional:
        _keys(entry, {"configuration", "clusters_sha256"}, "prior cohort")
        if not _hex(entry["clusters_sha256"]):
            raise ValueError("Prior cohorts require a complete cluster digest")
    fixtures = config["cross_grid_fixtures"]
    if type(fixtures) is not list or len(fixtures) > 4096:
        raise ValueError("Cross-grid fixture inventory must be a bounded list")
    fixture_ids = set()
    for fixture in fixtures:
        _keys(fixture, {"id", "source_optics", "truth_opd_nm", "sources"}, "cross-grid fixture")
        if (type(fixture["id"]) is not str or not fixture["id"] or fixture["id"] in fixture_ids
                or type(fixture["sources"]) is not list or not fixture["sources"]
                or any(type(s) is not str or not s for s in fixture["sources"])):
            raise ValueError("Cross-grid fixtures require unique IDs and provenance")
        fixture_ids.add(fixture["id"])
        map_pupil_coefficients(fixture["truth_opd_nm"], fixture["source_optics"], expected_spec["optics"])
    reviewed = config["reviewed_fixture_sources"]
    if (type(reviewed) is not dict or not reviewed or len(reviewed) > 512
            or any(type(name) is not str or not name or not _hex(value)
                   for name, value in reviewed.items())):
        raise ValueError("A reviewed fixture-source inventory is required")
    for name in reviewed:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
            raise ValueError("Reviewed fixture sources must be repository-relative paths")
    commitments = seed_commitments(config)
    if config["seed_commitments"] != commitments:
        raise ValueError("Seed commitments disagree with domain keys")
    old_keys = {historical["cohort"]["seed_hex"], historical["detector_seed_hex"],
                historical["policy_seed_hex"],
                *(entry["configuration"]["seed_hex"] for entry in
                  [*historical["excluded_cohorts"], *additional])}
    new_keys = {cohort["seed_hex"], *(config[f"{d}_seed_hex"] for d in DOMAINS[1:])}
    if old_keys & new_keys:
        raise ValueError("New domain keys reuse a prior cohort or experiment key")


def _admission_bindings(config, admission, sources, runtime_value, revision=None):
    _keys(admission, {"schema", "purpose", "status", "configuration_sha256", "source_revision",
        "source_sha256", "runtime", "seed_commitments", "exclusion_inventory_sha256"}, "admission")
    if admission["schema"] != 1 or type(admission["schema"]) is not int or admission["status"] != "FROZEN":
        raise ValueError("Acquisition requires a frozen admission receipt")
    if type(admission["source_revision"]) is not str or re.fullmatch(r"[0-9a-f]{40}", admission["source_revision"]) is None:
        raise ValueError("Admission requires a full Git source revision")
    expected = {"schema": 1, "purpose": PURPOSE, "status": "FROZEN",
        "configuration_sha256": canonical_hash(config), "source_revision": revision or admission["source_revision"],
        "source_sha256": sources, "runtime": runtime_value, "seed_commitments": seed_commitments(config),
        "exclusion_inventory_sha256": canonical_hash(exclusion_inventory(config))}
    if admission != expected:
        raise ValueError("Admission does not bind the current configuration, source, runtime or exclusions")


def validate_admission(config, admission):
    revision, sources = committed_source()
    _admission_bindings(config, admission, sources, runtime(), revision)
    source_root = source_inventory()[0]
    for name, expected in config["reviewed_fixture_sources"].items():
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=source_root)
        if (hashlib.sha256(committed).hexdigest() != expected or
                (source_root / name).is_symlink() or digest(source_root / name) != expected):
            raise ValueError("Reviewed fixture source is not the pinned committed content")


def storage_estimate(config):
    """Conservative enforced payload allowances; no wavefront generation."""
    validate_configuration(config)
    raw = 2016 * 64 * 64 * 8
    retained = raw + 288 * (CASE_JSON_LIMIT + CONTAINER_ALLOWANCE) + GLOBAL_ALLOWANCE
    return {"episodes": 288, "maximum_exposures": 2016, "raw_image_bytes": raw,
            "retained_bytes_for_admission": retained,
            "required_free_bytes": config["operating_reserve_bytes"] + 3 * retained + 16 * 1024**2}


def _prepare(config):
    """Called only after frozen admission and storage checks, never by inspection."""
    inventory = exclusion_inventory(config)
    spec = config["cohort"]["specification"]
    gram = pupil_gram(OpticalModel(OpticsConfig(**spec["optics"])))
    prior = []
    for entry in inventory["required_cohorts"] + inventory["additional_cohorts"]:
        previous = entry["configuration"]
        regenerated = generate_cohort(previous)
        if canonical_hash(regenerated["clusters"]) != entry["clusters_sha256"]:
            raise ValueError("Prior cohort cluster fingerprint changed")
        prior.append((previous, regenerated))
    # All declared prior cohorts have been verified before drawing any new key.
    cohort = generate_cohort(config["cohort"])
    if cohort["base_count"] != 24 or cohort["member_count"] != 48:
        raise ValueError("Exactly 24 distinct two-member conjugate clusters are required")
    for previous, regenerated in prior:
        ids = {base["base_wavefront_id"] for base in regenerated["clusters"]}
        ratio = map_pupil_coefficients(np.ones(6), previous["specification"]["optics"], spec["optics"])
        vectors = [np.asarray(base["members"][0]["truth_opd_nm"]) * ratio for base in regenerated["clusters"]]
        tolerance = max(spec["overlap_tolerance_nm"], previous["specification"]["overlap_tolerance_nm"])
        for base in cohort["clusters"]:
            if (base["base_wavefront_id"] in ids or
                    orbit_distances(base["members"][0]["truth_opd_nm"], vectors, gram).min() <= tolerance):
                raise ValueError("New cohort overlaps a previously exposed identity or conjugate orbit")
    converted = [map_pupil_coefficients(row["truth_opd_nm"], row["source_optics"], spec["optics"])
                 for row in config["cross_grid_fixtures"]]
    for base in cohort["clusters"]:
        if converted and orbit_distances(base["members"][0]["truth_opd_nm"], converted, gram).min() <= spec["overlap_tolerance_nm"]:
            raise ValueError("New cohort overlaps a physical cross-grid fixture")
    return {"cohort": cohort, "exclusion_inventory_sha256": canonical_hash(inventory),
            "prior_cohorts": [{"configuration_sha256": canonical_hash(previous),
                               "clusters_sha256": canonical_hash(saved["clusters"]),
                               "base_count": saved["base_count"]} for previous, saved in prior]}


def cases_for(config, prepared):
    cases = []
    order = {name: index for index, (name, _, _) in enumerate(STRATA)}
    for base in sorted(prepared["cohort"]["clusters"],
                       key=lambda row: (order[row["stratum_id"]], row["base_index"])):
        for member in sorted(base["members"], key=lambda row: row["member"] != "primary"):
            stream = {"cohort_id": COHORT_ID, "base_index": base["base_index"],
                      "stratum_id": base["stratum_id"], "member": member["member"], "replicate": 0}
            seeds = {f"{domain}_seed": stream_seed(config[f"{domain}_seed_hex"], domain, stream)
                     for domain in ("detector", "policy")}
            for gain in GAINS:
                for family in ARMS:
                    candidate = next(c for c in config["candidates"] if c["family"] == family)
                    identity = {**stream, "base_wavefront_id": base["base_wavefront_id"],
                                "gain": gain, "candidate_id": candidate["id"], "family": family}
                    cases.append({**identity, **seeds, "truth_opd_nm": member["truth_opd_nm"],
                                  "id": "case_" + canonical_hash(identity)})
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identity")
    return cases


def _instrument(config):
    return EpisodeConfig(max_exposures=config["exposure_budget"],
        photon_budget=config["exposure_budget"] * config["photons_per_exposure"],
        photons_per_exposure=config["photons_per_exposure"],
        read_noise_electrons=config["read_noise_electrons"],
        command_limit_nm=config["cohort"]["specification"]["command_limit_nm"])


def _case_limit(config):
    optics = config["cohort"]["specification"]["optics"]
    pixels = optics["pupil_pixels"] * optics["oversampling"]
    return config["exposure_budget"] * pixels**2 * 8 + CASE_JSON_LIMIT + CONTAINER_ALLOWANCE


def _binding(study, case):
    return canonical_hash({"study_sha256": canonical_hash(study), "case": case})


def _run_case(study, case):
    config = study["configuration"]["protocol"]
    model = OpticalModel(OpticsConfig(**config["cohort"]["specification"]["optics"]))
    candidate = next(c for c in config["candidates"] if c["id"] == case["candidate_id"])
    started = perf_counter()
    result = run_classical_episode(model, _instrument(config), case["truth_opd_nm"], case["detector_seed"],
        command_gain=case["gain"], method=case["family"], policy_seed=case["policy_seed"],
        policy_config=PolicyConfig(**candidate["policy"]))
    elapsed = perf_counter() - started
    observations = result["observations"]
    failure = result["failed_observation"]
    all_frames = [o.image_electrons for o in observations]
    if failure is not None:
        all_frames.append(failure.image_electrons)
    images = np.ascontiguousarray(np.stack(all_frames), dtype="<f8")
    record = {"schema": 1, "purpose": PURPOSE, "case": case, "case_binding": _binding(study, case),
        "observations": [o.metadata() for o in observations],
        "failed_observation": None if failure is None else failure.metadata(),
        "evaluation": result["evaluation"], "fit_records": result["fit_records"],
        "elapsed_seconds": elapsed, "images_sha256": hashlib.sha256(images.tobytes()).hexdigest()}
    return record, images


def _save_case(path, config, record, images):
    if path.exists():
        raise FileExistsError("Terminal cases cannot be replaced")
    raw = _json_bytes(record)
    if len(raw) > CASE_JSON_LIMIT:
        raise ValueError("Case diagnostics exceed the enforced metadata allowance")
    temporary = path.with_suffix(".npz.pending")
    with temporary.open("xb") as output:
        np.savez(output, images=images, metadata=np.frombuffer(raw, dtype=np.uint8))
        output.flush()
        os.fsync(output.fileno())
    if temporary.stat().st_size > _case_limit(config):
        raise ValueError("Case archive exceeds the enforced retention allowance")
    os.replace(temporary, path)


def _load_case(path, config):
    if path.is_symlink() or path.stat().st_size > _case_limit(config):
        raise ValueError("Unsafe or oversized case archive")
    pixels = config["cohort"]["specification"]["optics"]["pupil_pixels"] * config["cohort"]["specification"]["optics"]["oversampling"]
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 2 or {m.filename for m in members} != {"images.npy", "metadata.npy"}:
            raise ValueError("Unexpected case archive members")
        for member in members:
            if member.compress_type != zipfile.ZIP_STORED or member.file_size > _case_limit(config):
                raise ValueError("Unsupported or oversized array member")
            with archive.open(member) as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError("Unsupported NPY header version")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                if member.filename == "images.npy":
                    valid = (len(shape) == 3 and 1 <= shape[0] <= config["exposure_budget"]
                             and shape[1:] == (pixels, pixels) and dtype == np.dtype("<f8"))
                else:
                    valid = len(shape) == 1 and 0 < shape[0] <= CASE_JSON_LIMIT and dtype == np.dtype("uint8")
                if not valid or fortran or stream.tell() + int(np.prod(shape)) * dtype.itemsize != member.file_size:
                    raise ValueError("Invalid bounded array header")
    with np.load(path, allow_pickle=False) as data:
        return _parse(data["metadata"].tobytes()), data["images"].copy()


def verify_case(study, case, record, images):
    """Reconstruct the saved policy path and physical camera, without refitting."""
    _keys(record, {"schema", "purpose", "case", "case_binding", "observations", "failed_observation",
                   "evaluation", "fit_records", "elapsed_seconds", "images_sha256"}, "case record")
    if (type(record["schema"]) is not int or record["schema"] != 1 or record["purpose"] != PURPOSE
            or record["case"] != case or record["case_binding"] != _binding(study, case)
            or record["images_sha256"] != hashlib.sha256(images.tobytes()).hexdigest()):
        raise ValueError("Case content or binding changed")
    if (type(record["elapsed_seconds"]) not in (int, float)
            or not np.isfinite(record["elapsed_seconds"]) or record["elapsed_seconds"] < 0):
        raise ValueError("Invalid elapsed duration")
    config = study["configuration"]["protocol"]
    model = OpticalModel(OpticsConfig(**config["cohort"]["specification"]["optics"]))
    instrument = _instrument(config)
    candidate = next(c for c in config["candidates"] if c["id"] == case["candidate_id"])
    policy = ClassicalPolicy(case["family"], model, instrument, PolicyConfig(**candidate["policy"]), case["policy_seed"])
    episode = GainEpisode(model, instrument, case["truth_opd_nm"], case["detector_seed"], command_gain=case["gain"])
    history = []
    fits = record["fit_records"]
    if type(fits) is not list or len(fits) > instrument.max_exposures:
        raise ValueError("Invalid bounded fit inventory")
    for index, image in enumerate(images):
        before = [fit for fit in fits if fit["observations_used"] <= len(history)] if case["family"] == ARMS[1] else []
        restore_policy_fits(policy, before, history)
        command = policy.command(tuple(history))
        try:
            observation = episode.expose(command)
        except NumericalObservationError:
            if (index != len(images) - 1 or episode.failed_observation.metadata() != record["failed_observation"]
                    or episode.failed_observation.image_electrons.tobytes() != image.tobytes()
                    or record["observations"] != [o.metadata() for o in episode.observations]
                    or before != fits):
                raise ValueError("Terminal numerical failure evidence disagrees") from None
            break
        if observation.image_electrons.tobytes() != image.tobytes():
            raise ValueError("Raw frame disagrees with deterministic camera replay")
        history.append(Observation(observation.index, observation.commanded_opd_nm,
            observation.expected_photons, observation.cumulative_expected_photons, observation.image_electrons))
    else:
        if len(images) != instrument.max_exposures or record["failed_observation"] is not None:
            raise ValueError("Submitted case lacks its complete exposure inventory")
        restore_policy_fits(policy, fits, history, final=True)
        episode.submit(policy.final_command(tuple(history)))
    if (record["observations"] != [o.metadata() for o in episode.observations]
            or record["evaluation"] != episode.evaluator_record()):
        raise ValueError("Retained evaluator state disagrees with the saved policy path")


def _case_path(root, case):
    return root / "cases" / (case["id"] + ".npz")


def _sources_unchanged(study):
    if source_inventory()[1] != study["source_sha256"] or runtime() != study["runtime"]:
        raise RuntimeError("Source or runtime changed during the study")


def _scan(root, study, cases, *, complete=False):
    config = study["configuration"]["protocol"]
    expected = {".lock", "study.json", "preparation.json", "cases.json",
                *("source/" + name for name in study["source_sha256"]),
                *("reviewed-sources/" + name for name in config["reviewed_fixture_sources"])}
    if complete:
        expected |= {"summary.json", "analysis.json", "manifest.json"}
    records, pending = [], []
    for case in cases:
        path = _case_path(root, case)
        if path.exists():
            record, images = _load_case(path, config)
            verify_case(study, case, record, images)
            records.append(record)
            expected.add(str(path.relative_to(root)))
        elif complete:
            raise ValueError("Completed study is missing a terminal case")
        buffer = path.with_suffix(".npz.pending")
        if buffer.exists():
            if path.exists() or complete or buffer.stat().st_size > _case_limit(config):
                raise ValueError("Invalid pending case buffer")
            record, images = _load_case(buffer, config)
            verify_case(study, case, record, images)
            pending.append((buffer, path, record))
    if len(pending) > 1:
        raise ValueError("At most one unpublished case buffer is permitted")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if not complete:
        for name in ("summary.json", "analysis.json"):
            if (root / name).exists():
                if len(records) != len(cases):
                    raise ValueError("Premature completed-study artifact")
                expected.add(name)
    if actual != expected | {str(buffer.relative_to(root)) for buffer, _, _ in pending}:
        raise ValueError("Unexpected or missing study artifact")
    # A complete fsynced terminal buffer is recovered, never reacquired. Corrupt
    # or incomplete buffers fail above and remain untouched for inspection.
    for buffer, path, record in pending:
        os.replace(buffer, path)
        records.append(record)
    return records


def _summary(records):
    return [{"case_id": r["case"]["id"], "base_wavefront_id": r["case"]["base_wavefront_id"],
        "stratum_id": r["case"]["stratum_id"], "member": r["case"]["member"],
        "gain": r["case"]["gain"], "family": r["case"]["family"],
        "status": r["evaluation"]["status"],
        "residual_wfe_nm": r["evaluation"]["physical"].get("final", {}).get("residual_wfe_nm"),
        "exposures": r["evaluation"]["physical"]["exposures"],
        "expected_photons_used": r["evaluation"]["physical"]["expected_photons_used"],
        "fit_count": len(r["fit_records"]),
        "numerical_fit_failures": sum(f["fit_status"] == "numerical_failure" for f in r["fit_records"]),
        "optimizer_unsuccessful_fits": sum(not f["optimizer_success"] for f in r["fit_records"]),
        "elapsed_seconds": r["elapsed_seconds"]} for r in records]


def analysis(config, cases, records):
    """Pointwise descriptive intervals, resampling only independent base curves."""
    if {r["case"]["id"] for r in records} != {c["id"] for c in cases} or len(records) != len(cases):
        raise ValueError("Analysis requires every declared terminal identity exactly once")
    rows = _summary(records)
    result = {"schema": 1, "purpose": PURPOSE, "scope": "development_only",
        "contrast": "adaptive_minus_fixed", "units": "nm", "interval_scope": "pointwise_descriptive",
        "method": "equal_stratum_base_cluster_percentile", "resamples": RESAMPLES,
        "configuration_sha256": canonical_hash(config), "case_inventory_sha256": canonical_hash(cases),
        "outcomes_sha256": canonical_hash(rows), "bootstrap_key_commitment": config["seed_commitments"]["bootstrap"]}
    failures = [row["case_id"] for row in rows if row["status"] != "submitted"
                or row["residual_wfe_nm"] is None or not np.isfinite(row["residual_wfe_nm"])]
    if failures:
        return {**result, "status": "withheld_missing_finite_outcomes", "failed_case_ids": failures}
    matrices, identities = {}, {}
    for name, _, _ in STRATA:
        if not any(row["stratum_id"] == name for row in rows):
            continue
        bases = sorted({row["base_wavefront_id"] for row in rows if row["stratum_id"] == name})
        if len(bases) < 2:
            raise ValueError("Uncertainty requires at least two independent bases per stratum")
        data = np.empty((len(bases), len(GAINS), len(ARMS)))
        for b, base in enumerate(bases):
            for g, gain in enumerate(GAINS):
                for a, arm in enumerate(ARMS):
                    selected = [row for row in rows if row["base_wavefront_id"] == base
                                and row["gain"] == gain and row["family"] == arm]
                    if len(selected) != 2 or {row["member"] for row in selected} != {"primary", "conjugate"}:
                        raise ValueError("Each base curve requires both conjugates at every gain and arm")
                    data[b, g, a] = np.mean([row["residual_wfe_nm"] for row in selected])
        matrices[name], identities[name] = data, bases
    draws = {}
    bootstrap_seed = stream_seed(config["bootstrap_seed_hex"], "bootstrap",
        {"cohort_id": COHORT_ID, "analysis": "paired_base_curve_v1", "replicates": RESAMPLES})
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    for name, data in matrices.items():
        draws[name] = data[rng.integers(0, len(data), size=(RESAMPLES, len(data)))].mean(axis=1)
    def curves(mean, resampled):
        def estimate(value, sample):
            return {"mean_nm": float(value), "interval_nm": np.quantile(sample, [.025, .975], method="linear").tolist()}
        return [{"gain": gain,
            "fixed_wfe": estimate(mean[g, 0], resampled[:, g, 0]),
            "adaptive_wfe": estimate(mean[g, 1], resampled[:, g, 1]),
            "adaptive_minus_fixed": estimate(mean[g, 1] - mean[g, 0], resampled[:, g, 1] - resampled[:, g, 0]),
            "fixed_degradation_from_unit_gain": estimate(mean[g, 0] - mean[1, 0], resampled[:, g, 0] - resampled[:, 1, 0]),
            "adaptive_degradation_from_unit_gain": estimate(mean[g, 1] - mean[1, 1], resampled[:, g, 1] - resampled[:, 1, 1]),
            "difference_in_differences": estimate((mean[g, 1] - mean[g, 0]) - (mean[1, 1] - mean[1, 0]),
                (resampled[:, g, 1] - resampled[:, g, 0]) - (resampled[:, 1, 1] - resampled[:, 1, 0]))}
                for g, gain in enumerate(GAINS)]
    return {**result, "status": "descriptive_complete", "independent_bases": sum(map(len, identities.values())),
        "bootstrap_seed": bootstrap_seed, "bootstrap_stratum_order": list(matrices),
        "base_curves": {name: {"base_ids": identities[name], "gain_arm_wfe_nm": data.tolist()}
                        for name, data in matrices.items()},
        "strata": {name: curves(data.mean(axis=0), draws[name]) for name, data in matrices.items()},
        "equal_stratum": curves(np.stack([data.mean(axis=0) for data in matrices.values()]).mean(axis=0),
                                np.stack(list(draws.values())).mean(axis=0))}


def _no_links(root):
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Study paths must not contain symbolic links")


def _global_budget(root, proposed=None):
    proposed = {} if proposed is None else proposed
    size = sum(path.stat().st_size for path in root.rglob("*")
               if path.is_file() and path.relative_to(root).parts[0] != "cases"
               and str(path.relative_to(root)) not in proposed)
    for value in proposed.values():
        encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if len(encoded) > JSON_LIMIT:
            raise ValueError("Global JSON artifact exceeds its byte allowance")
        size += len(encoded)
    if size > GLOBAL_ALLOWANCE:
        raise ValueError("Global artifact inventory exceeds the total retention allowance")


def _publish_global(root, name, value):
    _global_budget(root, {name: value})
    publish_json(root / name, value)


def verify_run(root):
    root = Path(root)
    _no_links(root)
    _global_budget(root)
    for name in ("manifest.json", "study.json", "preparation.json", "cases.json", "summary.json", "analysis.json"):
        _read_json(root / name)
    study = _read_json(root / "study.json")
    _keys(study, {"configuration", "source_sha256", "runtime"}, "study")
    _keys(study["configuration"], {"protocol", "admission", "operating_reserve_bytes"}, "study binding")
    config = study["configuration"]["protocol"]
    validate_configuration(config)
    _sources_unchanged(study)
    if study["configuration"]["operating_reserve_bytes"] != config["operating_reserve_bytes"]:
        raise ValueError("Operating reserve binding changed")
    _admission_bindings(config, study["configuration"]["admission"], study["source_sha256"], study["runtime"])
    saved_cases = _read_json(root / "cases.json")
    if type(saved_cases) is not list or len(saved_cases) > 288:
        raise ValueError("Invalid bounded case inventory")
    for case in saved_cases:
        if type(case) is not dict or type(case.get("id")) is not str or re.fullmatch(r"case_[0-9a-f]{64}", case["id"]) is None:
            raise ValueError("Unsafe case identity")
        path = _case_path(root, case)
        if path.stat().st_size > _case_limit(config):
            raise ValueError("Oversized terminal case")
    expected_files = {".lock", "study.json", "preparation.json", "cases.json", "summary.json", "analysis.json",
        *("source/" + name for name in study["source_sha256"]),
        *("reviewed-sources/" + name for name in config["reviewed_fixture_sources"]),
        *(str(_case_path(root, case).relative_to(root)) for case in saved_cases)}
    saved_manifest = _read_json(root / "manifest.json")
    if (set(saved_manifest["files"]) != expected_files or
            {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} != expected_files | {"manifest.json"}):
        raise ValueError("Unexpected or missing completed-study artifact")
    manifest, saved_study = verify_study(root, PURPOSE)
    for name, expected in config["reviewed_fixture_sources"].items():
        if digest(root / "reviewed-sources" / name) != expected:
            raise ValueError("Reviewed fixture-source snapshot changed")
    # Only the previously frozen and hash-verified cohort may now be regenerated.
    prepared = _prepare(config)
    cases = cases_for(config, prepared)
    records = _scan(root, study, cases, complete=True)
    if saved_study != study or _read_json(root / "preparation.json") != prepared or _read_json(root / "cases.json") != cases:
        raise ValueError("Frozen preparation or case inventory changed")
    if _read_json(root / "summary.json") != _summary(records) or _read_json(root / "analysis.json") != analysis(config, cases, records):
        raise ValueError("Derived report disagrees with complete retained evidence")
    if manifest.get("episode_count") != len(cases) or manifest.get("exposure_count") != sum(r["evaluation"]["physical"]["exposures"] for r in records):
        raise ValueError("Manifest terminal/exposure counts disagree")
    _sources_unchanged(study)
    return manifest


def run(root, config, admission, *, resume=False, limit_episodes=None):
    """Acquire only an admitted fixed cohort; terminal case publication is atomic."""
    validate_configuration(config)
    validate_admission(config, admission)
    if limit_episodes is not None and (type(limit_episodes) is not int or limit_episodes < 1):
        raise ValueError("Episode limit must be a positive integer")
    root = Path(root)
    _no_links(root)
    if (root / "manifest.json").exists():
        verify_run(root)
    # Once preparation exists, a missing source snapshot is corruption, not an
    # interrupted initial copy. Refuse before the shared helper could repair it.
    if (root / "preparation.json").exists():
        previous = _read_json(root / "study.json")
        for name, expected in previous["source_sha256"].items():
            if not (root / "source" / name).is_file() or digest(root / "source" / name) != expected:
                raise ValueError("Immutable source snapshot is missing or changed")
        for name, expected in config["reviewed_fixture_sources"].items():
            if not (root / "reviewed-sources" / name).is_file() or digest(root / "reviewed-sources" / name) != expected:
                raise ValueError("Immutable fixture-source snapshot is missing or changed")
    estimate = storage_estimate(config)
    bound = {"protocol": config, "admission": admission, "operating_reserve_bytes": config["operating_reserve_bytes"]}
    source_root, source_files = source_inventory()
    source_bytes = sum((source_root / name).stat().st_size for name in source_files)
    fixture_bytes = sum((source_root / name).stat().st_size for name in config["reviewed_fixture_sources"])
    initial_study = {"configuration": bound, "source_sha256": source_files, "runtime": runtime()}
    header_bytes = len((json.dumps(initial_study, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    if (source_bytes > JSON_LIMIT or fixture_bytes > JSON_LIMIT or header_bytes > JSON_LIMIT
            or source_bytes + fixture_bytes + header_bytes > GLOBAL_ALLOWANCE):
        raise ValueError("Initial source/header retention allowance exceeded")
    with study_directory(root, bound, estimate["retained_bytes_for_admission"], resume) as study:
        if (root / "manifest.json").exists():
            return verify_run(root)
        _admission_bindings(config, admission, study["source_sha256"], study["runtime"])
        _sources_unchanged(study)
        if sum((root / "source" / name).stat().st_size for name in study["source_sha256"]) > JSON_LIMIT:
            raise ValueError("Source snapshot exceeds the global retention allowance")
        for name, expected in config["reviewed_fixture_sources"].items():
            destination = root / "reviewed-sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if digest(destination) != expected:
                    raise ValueError("Reviewed fixture source changed")
            else:
                temporary = destination.with_suffix(destination.suffix + ".pending")
                shutil.copyfile(source_root / name, temporary)
                os.replace(temporary, destination)
            if digest(destination) != expected:
                raise ValueError("Newly copied fixture-source snapshot changed")
        _sources_unchanged(study)
        prepared = _prepare(config)
        cases = cases_for(config, prepared)
        for name, value in (("preparation.json", prepared), ("cases.json", cases)):
            path = root / name
            if path.exists():
                if _read_json(path) != value:
                    raise ValueError("Frozen preparation changed")
            else:
                _publish_global(root, name, value)
        records = _scan(root, study, cases)
        existing = {record["case"]["id"] for record in records}
        acquired = 0
        for case in cases:
            if case["id"] in existing:
                continue
            _sources_unchanged(study)
            if shutil.disk_usage(root).free < config["operating_reserve_bytes"] + 2 * _case_limit(config) + 16 * 1024**2:
                raise OSError("Insufficient reserve for the next terminal case")
            record, images = _run_case(study, case)
            _sources_unchanged(study)
            path = _case_path(root, case)
            path.parent.mkdir(exist_ok=True)
            _save_case(path, config, record, images)
            records.append(record)
            acquired += 1
            if limit_episodes is not None and acquired >= limit_episodes and len(records) < len(cases):
                return {"complete": False, "completed_episodes": len(records), "expected_episodes": len(cases)}
        records = sorted(records, key=lambda r: next(i for i, c in enumerate(cases) if c["id"] == r["case"]["id"]))
        for name, value in (("summary.json", _summary(records)), ("analysis.json", analysis(config, cases, records))):
            if len(_json_bytes(value)) > JSON_LIMIT:
                raise ValueError("Derived artifact exceeds the global retention allowance")
            path = root / name
            if path.exists() and _read_json(path) != value:
                raise ValueError("Retained derived artifact changed")
            if not path.exists():
                _publish_global(root, name, value)
        counts = {"episode_count": len(records),
                  "exposure_count": sum(r["evaluation"]["physical"]["exposures"] for r in records)}
        proposed_manifest = {"schema": 1, "purpose": PURPOSE, "source_sha256": study["source_sha256"],
            **counts, "files": {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
                if path.is_file() and path.name != "manifest.json.pending"}}
        _global_budget(root, {"manifest.json": proposed_manifest})
        publish_manifest(root, study, PURPOSE, episode_count=len(records),
                         exposure_count=sum(r["evaluation"]["physical"]["exposures"] for r in records))
    return verify_run(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        if any((args.config, args.admission, args.output, args.resume, args.limit_episodes, args.estimate)):
            parser.error("Verification cannot be combined with acquisition options")
        result = verify_run(args.verify)
    elif args.estimate:
        if not args.config or any((args.admission, args.output, args.resume, args.limit_episodes)):
            parser.error("Estimation requires only --config")
        result = storage_estimate(_read_json(args.config))
    else:
        if not all((args.config, args.admission, args.output)):
            parser.error("Acquisition requires --config, --admission and --output")
        result = run(args.output, _read_json(args.config), _read_json(args.admission),
                     resume=args.resume, limit_episodes=args.limit_episodes)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
