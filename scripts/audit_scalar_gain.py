#!/usr/bin/env python3
"""Independently audit the released scalar-gain development cohort.

Usage: python scripts/audit_scalar_gain.py --run EXTRACTED_RUN --out audit.json

The completed run is pinned by its manifest, configuration and source revision.
No ProbeBench modules are imported. The audit reconstructs finite-pupil WFE,
saved command paths, base/conjugate grouping and the specified bootstrap using
independent formulas. NumPy must match the acquisition version for PCG64 replay.

WFE uses simulator truth and physical commands, not an estimate from pixels.
This does not refit the estimator, replay the noisy camera, reconstruct Strehl,
or regenerate historical exclusion cohorts. Passing checks establish retained
evidence consistency, not estimator optimality, hardware reliability or safety.
Intervals remain pointwise and descriptive. The run must be an immutable local
copy; file hashes detect ordinary corruption, not a malicious concurrent writer.
The output receipt is created exclusively, outside the run, and hashes this file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import hmac
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import numpy as np

PURPOSE = "scalar_gain_development_v1"
COHORT = "nominal_calibration_development_v1"
SAMPLER = "finite_pupil_isotropic_uniform_rms_pcg64_v1"
CONFIG_SHA256 = "47145e03b9c96bc75256cd56d27ed55400ee5131c525fe57410fd76eaad5c5e8"
REVISION = "99e5e883c9e064fdf778d8cbea7e439e7f09e91b"
MANIFEST_SHA256 = "ee56441561fa50aeb80f1e33e516cf9151248445f75d997e0d46e9b5002c3322"
STRATA = (("low", 10.0, 25.0), ("medium", 25.0, 50.0), ("high", 50.0, 90.0))
GAINS = (0.9, 1.0, 1.1)
ARMS = ("fixed_diversity_mle_v1", "phase_diversity_mle_v1")
CANDIDATES = ("fixed_probe_080", "adaptive_probe_080")
MEMBERS = ("primary", "conjugate")
SIGN = np.array([-1., -1., -1., 1., 1., -1.])
RESAMPLES = 10000
ATOL, RTOL = 1e-9, 2e-12
CASE_LIMIT = 7 * 64 * 64 * 8 + 64 * 1024 + 2048


def require(condition, label):
    if not condition:
        raise ValueError(label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    def invalid(_value):
        raise ValueError("Nonfinite JSON literal")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path, limit=4 * 1024**2):
    require(path.is_file() and not path.is_symlink(), "Missing or linked JSON artifact")
    require(path.stat().st_size <= limit, "Oversized JSON artifact")
    return parse(path.read_bytes())


def close(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label)
    require(np.allclose(a, b, rtol=RTOL, atol=ATOL), label)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def seed(key, domain, identity):
    require(isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key), "Invalid frozen key")
    return int.from_bytes(hmac.digest(bytes.fromhex(key), canonical([SAMPLER, domain, identity]), "sha256"), "big")


def independent_gram():
    # Positive Zernike normalization constants cancel in finite-pupil RMS
    # normalization. Evaluate the raw polynomials point by point, and use fsum
    # reductions rather than the production OpticalModel/einsum implementation.
    points = []
    for j in range(32):
        y = (2 * j - 31) / 32
        for i in range(32):
            x = (2 * i - 31) / 32
            r2 = x*x + y*y
            if r2 <= 1:
                points.append((2*r2-1, x*x-y*y, 2*x*y, (3*r2-2)*x,
                               (3*r2-2)*y, 6*r2*r2-6*r2+1))
    centered = []
    for column in zip(*points):
        mean = math.fsum(column) / len(column)
        scale = math.sqrt(math.fsum((v-mean)**2 for v in column) / len(column))
        normalized = [(v-mean)/scale for v in column]
        piston = math.fsum(normalized) / len(normalized)
        centered.append([v-piston for v in normalized])
    return np.array([[math.fsum(x*y for x, y in zip(a, b)) / len(a)
                      for b in centered] for a in centered]), len(points)


def vector(value, label):
    result = np.asarray(value, dtype=float)
    require(result.shape == (6,) and np.isfinite(result).all(), label)
    return result


def wfe(coefficients, gram):
    v = vector(coefficients, "Invalid WFE coefficients")
    value = math.fsum(float(v[i])*float(gram[i, j])*float(v[j])
                      for i in range(6) for j in range(6))
    require(value >= -1e-12, "Negative physical variance")
    return math.sqrt(max(0., value))


def check_inventory(root, manifest, study, cases):
    require(manifest["schema"] == 1 and manifest["purpose"] == PURPOSE, "Wrong manifest protocol")
    require(manifest["source_sha256"] == study["source_sha256"], "Manifest source mismatch")
    config = study["configuration"]["protocol"]
    expected = {".lock", "study.json", "preparation.json", "cases.json", "summary.json", "analysis.json"}
    expected.update("source/" + name for name in study["source_sha256"])
    expected.update("reviewed-sources/" + name for name in config["reviewed_fixture_sources"])
    expected.update("cases/" + case["id"] + ".npz" for case in cases)
    require(set(manifest["files"]) == expected, "Manifest inventory differs from completed protocol")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Linked retained artifact")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == expected | {"manifest.json"}, "Extra, missing or pending retained artifact")
    for name, expected_hash in manifest["files"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
                and "\\" not in name, "Unsafe manifest path")
        require(re.fullmatch(r"[0-9a-f]{64}", expected_hash), "Invalid manifest digest")
        require(digest_file(root / name) == expected_hash, "Retained file hash mismatch: " + name)
    for prefix, sources in (("source/", study["source_sha256"]),
                            ("reviewed-sources/", config["reviewed_fixture_sources"])):
        for name, expected_hash in sources.items():
            require(manifest["files"][prefix + name] == expected_hash, "Source snapshot binding mismatch")


def check_design(config, preparation, cases, gram):
    require(digest_value(config) == CONFIG_SHA256, "Configuration differs from prospective freeze")
    spec = config["cohort"]["specification"]
    require(spec == {"schema": 1, "cohort_id": COHORT, "split": "development",
        "optics": {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0},
        "strata": [{"id": name, "rms_min_nm": lo, "rms_max_nm": hi, "base_count": 8}
                   for name, lo, hi in STRATA], "command_limit_nm": 200.0, "overlap_tolerance_nm": 1e-6},
        "Declared optical/base design changed")
    require(config["gains"] == list(GAINS) and config["exposure_budget"] == 7
            and config["photons_per_exposure"] == 2000 and config["read_noise_electrons"] == 1.0,
            "Gain or camera budget changed")
    require([(c["id"], c["family"]) for c in config["candidates"]] == list(zip(CANDIDATES, ARMS)),
            "Arm identities changed")
    for candidate in config["candidates"]:
        require(candidate["policy"] == {"random_amplitude_nm": 60.0, "coordinate_step_nm": 40.0,
            "coordinate_step_floor_nm": 5.0, "diversity_probe_nm": 80.0, "max_mle_iterations": 200},
            "Frozen policy settings changed")
    keys = {"wavefront": config["cohort"]["seed_hex"],
            **{d: config[d + "_seed_hex"] for d in ("detector", "policy", "bootstrap")}}
    require(len(set(keys.values())) == 4, "Random domains reuse a key")
    require(config["seed_commitments"] == {d: hashlib.sha256(bytes.fromhex(k)).hexdigest()
                                           for d, k in keys.items()}, "Key commitments changed")
    cohort = preparation["cohort"]
    require(cohort["configuration_sha256"] == digest_value(config["cohort"])
            and cohort["sampler"] == SAMPLER and cohort["base_count"] == 24
            and cohort["member_count"] == 48, "Cohort binding or counts changed")
    close(cohort["gram_nm2_per_nm2"], gram, "Finite-pupil Gram disagrees")
    clusters = cohort["clusters"]
    require(len(clusters) == 24, "Wrong number of base clusters")
    bases = {(b["stratum_id"], b["base_index"]): b for b in clusters}
    require(set(bases) == {(s, i) for s, _, _ in STRATA for i in range(8)}, "Base identity grid incomplete")
    expected_cases, truths = [], []
    for stratum, lo, hi in STRATA:
        for index in range(8):
            base = bases[stratum, index]
            base_id = f"{COHORT}:{stratum}:{index:06d}"
            require(base["base_wavefront_id"] == base_id, "Base naming mismatch")
            require([m["member"] for m in base["members"]] == list(MEMBERS), "Conjugate inventory changed")
            primary = vector(base["members"][0]["truth_opd_nm"], "Invalid primary wavefront")
            close(base["members"][1]["truth_opd_nm"], primary * SIGN, "Conjugate polynomial mismatch")
            require(lo <= wfe(primary, gram) <= hi, "Base outside declared physical-RMS stratum")
            close(base["physical_rms_nm"], wfe(primary, gram), "Recorded base physical RMS disagrees")
            identity = {"cohort_id": COHORT, "split": "development", "stratum_id": stratum, "base_index": index}
            radial = np.random.Generator(np.random.PCG64(seed(keys["wavefront"], "wavefront", {**identity, "component": "radius"})))
            angular = np.random.Generator(np.random.PCG64(seed(keys["wavefront"], "wavefront", {**identity, "component": "direction"})))
            radius, direction = float(radial.uniform(lo, hi)), angular.normal(size=6)
            expected = radius * np.linalg.solve(np.linalg.cholesky(gram).T, direction / np.linalg.norm(direction))
            close(primary, expected, "Frozen seed-to-wavefront reconstruction disagrees")
            close(base["sampled_rms_nm"], radius, "Frozen radial draw disagrees")
            require(np.max(np.abs(primary)) <= 200 and wfe(primary-primary*SIGN, gram) > 1e-6,
                    "Base command envelope or nondegenerate conjugacy fails")
            for prior in truths:
                require(min(wfe(primary-prior, gram), wfe(primary-prior*SIGN, gram)) > 1e-6,
                        "Independent base grid repeats a conjugate orbit")
            truths.append(primary)
            for member in base["members"]:
                stream = {"cohort_id": COHORT, "base_index": index, "stratum_id": stratum,
                          "member": member["member"], "replicate": 0}
                random_fields = {d + "_seed": seed(keys[d], d, stream) for d in ("detector", "policy")}
                for gain in GAINS:
                    for candidate, arm in zip(CANDIDATES, ARMS):
                        identity = {**stream, "base_wavefront_id": base_id, "gain": gain,
                                    "candidate_id": candidate, "family": arm}
                        expected_cases.append({**identity, **random_fields, "truth_opd_nm": member["truth_opd_nm"],
                                               "id": "case_" + digest_value(identity)})
    require(cases == expected_cases and len(cases) == 288, "Complete ordered episode/stream grid disagrees")
    require(len({case["id"] for case in cases}) == 288, "Duplicate case ID")


def load_case(path):
    require(path.stat().st_size <= CASE_LIMIT, "Oversized case archive")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == 2 and {m.filename for m in members} == {"metadata.npy", "images.npy"},
                "Unexpected NPZ members")
        for member in members:
            require(member.compress_type == zipfile.ZIP_STORED and member.file_size <= CASE_LIMIT,
                    "Compressed or oversized NPZ member")
            with archive.open(member) as stream:
                require(np.lib.format.read_magic(stream) == (1, 0), "Unexpected NPY version")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                if member.filename == "images.npy":
                    valid = len(shape) == 3 and 1 <= shape[0] <= 7 and shape[1:] == (64, 64) and dtype == np.dtype("<f8")
                else:
                    valid = len(shape) == 1 and 0 < shape[0] <= 65536 and dtype == np.dtype("uint8")
                require(valid and not fortran and stream.tell() + math.prod(shape)*dtype.itemsize == member.file_size,
                        "Invalid bounded NPY layout")
        # Headers and bounded byte counts are verified before allocation.
    with np.load(path, allow_pickle=False) as data:
        return parse(data["metadata"].tobytes()), data["images"].copy()


def public_history_hash(observations, images):
    result = hashlib.sha256()
    for observation, image in zip(observations, images):
        legacy = {"index": observation["index"], "applied_opd_nm": observation["commanded_opd_nm"],
                  "expected_photons": observation["expected_photons"],
                  "cumulative_expected_photons": observation["cumulative_expected_photons"]}
        result.update(json.dumps(legacy, sort_keys=True, allow_nan=False).encode())
        result.update(json.dumps(image.shape).encode())
        result.update(image.tobytes())
    return result.hexdigest()


def check_record(study, case, record, images, gram):
    require(record["schema"] == 1 and record["purpose"] == PURPOSE and record["case"] == case,
            "Case metadata differs from frozen identity")
    require(record["case_binding"] == digest_value({"study_sha256": digest_value(study), "case": case}),
            "Case/study binding disagrees")
    require(record["images_sha256"] == hashlib.sha256(images.tobytes()).hexdigest(), "Raw image digest mismatch")
    evaluation, observations, fits = record["evaluation"], record["observations"], record["fit_records"]
    physical = evaluation["physical"]
    require(evaluation["schema"] == "probebench.scalar-gain-evaluation/1" and evaluation["command_gain"] == case["gain"],
            "Physical gain or schema mismatch")
    require(evaluation["nominal_command_limit_nm"] == 200.0 and evaluation["nominal_observations"] == observations,
            "Nominal command/observation contract mismatch")
    require(evaluation["nominal_calibration"] == {"response": "identity", "command_units": "nm_nominal_opd",
                                                "prediction_units": "nm_opd"}, "Wrong nominal calibration")
    truth, gain = vector(case["truth_opd_nm"], "Invalid truth vector"), case["gain"]
    close(physical["hidden_opd_nm"], truth, "Evaluator truth mismatch")
    count = images.shape[0]
    require(physical["exposures"] == count and physical["expected_photons_used"] == 2000*count,
            "Consumed exposure/photon counters disagree")
    require(len(physical["trajectory"]) == count, "Missing physical trajectory element")
    require(math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] >= 0, "Invalid episode duration")
    submitted = evaluation["status"] == "submitted"
    require(submitted or evaluation["status"] == "numerical_failure", "Undeclared terminal status")
    require(len(observations) == count - (not submitted), "Public/produced observation counts disagree")
    require(np.isfinite(images[:len(observations)]).all(), "Nonfinite public observation")
    expected_prefixes = ([7] if case["family"] == ARMS[0] else list(range(2, 8))) if submitted else (
        [] if case["family"] == ARMS[0] else list(range(2, count)))
    require([f["observations_used"] for f in fits] == expected_prefixes, "Frozen fit-prefix inventory disagrees")
    estimates = {}
    for fit in fits:
        n = fit["observations_used"]
        require(fit["visible_history_sha256"] == public_history_hash(observations[:n], images[:n]),
                "Fit does not bind its public observation prefix")
        estimate = vector(fit["estimated_opd_nm"], "Nonfinite saved estimate")
        require(np.max(np.abs(estimate)) <= 200 and fit["fit_status"] in {"finite", "numerical_failure"}
                and type(fit["optimizer_success"]) is bool, "Invalid fit status or bounded estimate")
        estimates[n] = estimate
    error = 0.0
    physical_commands = []
    for i, trajectory in enumerate(physical["trajectory"]):
        if i == 0:
            command = np.zeros(6)
        elif case["family"] == ARMS[0]:
            command = np.array([(0, 80, -80)[i % 3], 0, 0, 0, 0, 0.])
        elif i == 1:
            command = np.array([80., 0, 0, 0, 0, 0])
        else:
            command = estimates[i].copy()
            if i % 2:
                command[0] += -80 if ((i-3)//2) % 2 == 0 else 80
            command = np.clip(command, -200, 200)
        applied = gain * command
        physical_commands.append(applied)
        close(trajectory["applied_opd_nm"], applied, "Physical trajectory differs from frozen policy path")
        require(trajectory["index"] == i and trajectory["expected_photons"] == 2000
                and trajectory["cumulative_expected_photons"] == 2000*(i+1), "Trajectory accounting mismatch")
        error = max(error, close(trajectory["residual_wfe_nm"], wfe(truth-applied, gram), "Trajectory physical WFE mismatch"))
        if i < len(observations):
            public = observations[i]
            require(public["schema"] == "probebench.nominal-observation/1" and public["index"] == i
                    and public["expected_photons"] == 2000 and public["cumulative_expected_photons"] == 2000*(i+1)
                    and public["nominal_calibration"] == evaluation["nominal_calibration"], "Public observation accounting mismatch")
            close(public["commanded_opd_nm"], command, "Public nominal policy command mismatch")
            close(public["nominal_applied_opd_nm"], command, "Nominal calibration semantics mismatch")
    if submitted:
        require(count == 7 and record["failed_observation"] is None, "Submitted episode lacks complete exposure evidence")
        final_command = vector(evaluation["final_commanded_opd_nm"], "Invalid final command")
        close(final_command, estimates[7], "Final command differs from saved terminal estimate")
        close(physical["final_applied_opd_nm"], gain*final_command, "Final physical actuation mismatch")
        close(physical["final_mirror_surface_coefficients_nm"], gain*final_command/2, "OPD-to-surface conversion mismatch")
        error = max(error, close(physical["initial"]["residual_wfe_nm"], wfe(truth, gram), "Initial WFE mismatch"),
                    close(physical["final"]["residual_wfe_nm"], wfe(truth-gain*final_command, gram), "Final WFE mismatch"))
        measured = [i for i, command in enumerate(physical_commands) if np.array_equal(command, gain*final_command)]
        require(physical["final_command_observation_indices"] == measured
                and physical["final_command_has_camera_observation"] == bool(measured)
                and physical["final_command_is_last_observation"] == bool(measured and measured[-1] == 6),
                "Final command camera-support annotation disagrees")
        close(physical["peak_probe_wfe_increase_nm"], max(0., max(wfe(truth-command, gram)
              for command in physical_commands)-wfe(truth, gram)), "Peak probe WFE penalty mismatch")
    else:
        failure = evaluation["failure"]
        require(evaluation["final_commanded_opd_nm"] is None and "final" not in physical,
                "Terminal camera failure acquired a final score")
        require(record["failed_observation"] == failure["failed_physical_observation"]
                and failure["public_observation_count"] == count-1 and failure["produced_observation_count"] == count
                and failure["nonfinite_pixel_count"] == int(np.count_nonzero(~np.isfinite(images[-1])))
                and failure["nonfinite_pixel_count"] > 0, "Terminal failure evidence mismatch")
    return error


def summary_row(record):
    c, e = record["case"], record["evaluation"]
    return {"case_id": c["id"], "base_wavefront_id": c["base_wavefront_id"], "stratum_id": c["stratum_id"],
            "member": c["member"], "gain": c["gain"], "family": c["family"], "status": e["status"],
            "residual_wfe_nm": e["physical"].get("final", {}).get("residual_wfe_nm"),
            "exposures": e["physical"]["exposures"], "expected_photons_used": e["physical"]["expected_photons_used"],
            "fit_count": len(record["fit_records"]),
            "numerical_fit_failures": sum(f["fit_status"] == "numerical_failure" for f in record["fit_records"]),
            "optimizer_unsuccessful_fits": sum(f["optimizer_success"] is False for f in record["fit_records"]),
            "elapsed_seconds": record["elapsed_seconds"]}


def percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered)-1) * probability
    lower = math.floor(position)
    fraction = position-lower
    return ordered[lower] + fraction*(ordered[min(lower+1, len(ordered)-1)]-ordered[lower])


def estimates(mean, draws):
    flat, sampled = mean.reshape(6), draws.reshape(RESAMPLES, 6)
    rows = []
    for gain_index, gain in enumerate(GAINS):
        fixed, adaptive = np.eye(6)[2*gain_index:2*gain_index+2]
        fixed_unit, adaptive_unit = np.eye(6)[2:4]
        contrasts = {"fixed_wfe": fixed, "adaptive_wfe": adaptive,
            "adaptive_minus_fixed": adaptive-fixed,
            "fixed_degradation_from_unit_gain": fixed-fixed_unit,
            "adaptive_degradation_from_unit_gain": adaptive-adaptive_unit,
            "difference_in_differences": adaptive-fixed-adaptive_unit+fixed_unit}
        result = {"gain": gain}
        for name, weights in contrasts.items():
            value = math.fsum(float(v)*float(w) for v, w in zip(flat, weights))
            sample = sampled @ weights
            result[name] = {"mean_nm": value, "interval_nm": [percentile(sample, p) for p in (.025, .975)]}
        rows.append(result)
    return rows


def independent_analysis(config, cases, rows):
    result = {"schema": 1, "purpose": PURPOSE, "scope": "development_only", "contrast": "adaptive_minus_fixed",
        "units": "nm", "interval_scope": "pointwise_descriptive", "method": "equal_stratum_base_cluster_percentile",
        "resamples": RESAMPLES, "configuration_sha256": digest_value(config), "case_inventory_sha256": digest_value(cases),
        "outcomes_sha256": digest_value(rows), "bootstrap_key_commitment": config["seed_commitments"]["bootstrap"]}
    failed = [r["case_id"] for r in rows if r["status"] != "submitted" or r["residual_wfe_nm"] is None
              or not math.isfinite(r["residual_wfe_nm"])]
    if failed:
        return {**result, "status": "withheld_missing_finite_outcomes", "failed_case_ids": failed}
    stream = seed(config["bootstrap_seed_hex"], "bootstrap",
                  {"cohort_id": COHORT, "analysis": "paired_base_curve_v1", "replicates": RESAMPLES})
    rng = np.random.Generator(np.random.PCG64(stream))
    curves, group_means, bootstrap_means, strata = {}, [], [], {}
    for name, _, _ in STRATA:
        ids = sorted({r["base_wavefront_id"] for r in rows if r["stratum_id"] == name})
        require(len(ids) == 8, "Bootstrap stratum lacks eight bases")
        cube = np.empty((8, 3, 2))
        for b, base in enumerate(ids):
            for g, gain in enumerate(GAINS):
                for a, arm in enumerate(ARMS):
                    pair = [r for r in rows if r["base_wavefront_id"] == base and r["gain"] == gain and r["family"] == arm]
                    require(len(pair) == 2 and {r["member"] for r in pair} == set(MEMBERS), "Conjugate pair missing or repeated")
                    cube[b, g, a] = math.fsum(r["residual_wfe_nm"] for r in pair)/2
        # Resample bases jointly across all gains and arms. Convert sampled
        # indices to multiplicities, independently of production gather/mean.
        indices = rng.integers(0, 8, size=(RESAMPLES, 8))
        multiplicities = np.zeros((RESAMPLES, 8), dtype=np.int64)
        np.add.at(multiplicities, (np.repeat(np.arange(RESAMPLES), 8), indices.reshape(-1)), 1)
        draws = (multiplicities @ cube.reshape(8, 6)).reshape(RESAMPLES, 3, 2)/8
        mean = np.array([math.fsum(cube[:, g, a])/8 for g in range(3) for a in range(2)]).reshape(3, 2)
        curves[name] = {"base_ids": ids, "gain_arm_wfe_nm": cube.tolist()}
        strata[name] = estimates(mean, draws)
        group_means.append(mean)
        bootstrap_means.append(draws)
    pooled_mean = sum(group_means)/3
    pooled_draws = sum(bootstrap_means)/3
    return {**result, "status": "descriptive_complete", "independent_bases": 24, "bootstrap_seed": stream,
            "bootstrap_stratum_order": [s for s, _, _ in STRATA], "base_curves": curves, "strata": strata,
            "equal_stratum": estimates(pooled_mean, pooled_draws)}


def compare_report(saved, reconstructed, path="analysis"):
    if isinstance(reconstructed, dict):
        require(isinstance(saved, dict) and saved.keys() == reconstructed.keys(), "Report keys disagree: " + path)
        return max((compare_report(saved[k], v, path + "." + k) for k, v in reconstructed.items()), default=0.)
    if isinstance(reconstructed, list):
        require(isinstance(saved, list) and len(saved) == len(reconstructed), "Report list shape disagrees: " + path)
        return max((compare_report(a, b, path + f"[{i}]") for i, (a, b) in enumerate(zip(saved, reconstructed))), default=0.)
    if isinstance(reconstructed, float):
        return close(saved, reconstructed, "Report numerical mismatch: " + path)
    require(type(saved) is type(reconstructed) and saved == reconstructed, "Report metadata mismatch: " + path)
    return 0.


def audit(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Missing or linked run directory")
    require((root / "manifest.json").is_file(), "Complete manifest is absent")
    require(not (root / "manifest.json").is_symlink(), "Linked manifest")
    manifest_hash = digest_file(root / "manifest.json")
    require(manifest_hash == MANIFEST_SHA256, "Manifest differs from the released complete cohort")
    manifest, study = read_json(root / "manifest.json"), read_json(root / "study.json")
    admission = study["configuration"]["admission"]
    require(admission["status"] == "FROZEN" and admission["source_revision"] == REVISION
            and admission["configuration_sha256"] == CONFIG_SHA256, "Frozen admission changed")
    require(study["source_sha256"] == admission["source_sha256"]
            and study["runtime"] == admission["runtime"], "Study/admission/source/runtime binding mismatch")
    require(np.__version__ == admission["runtime"]["numpy"], "Use the frozen NumPy runtime for PCG64 compatibility")
    config = study["configuration"]["protocol"]
    cases, preparation = read_json(root / "cases.json"), read_json(root / "preparation.json")
    require(type(cases) is list and len(cases) == 288, "Incomplete episode inventory")
    for case in cases:
        require(isinstance(case, dict) and isinstance(case.get("id"), str)
                and re.fullmatch(r"case_[0-9a-f]{64}", case["id"]), "Unsafe case identity")
    check_inventory(root, manifest, study, cases)
    gram, pupil_samples = independent_gram()
    check_design(config, preparation, cases, gram)
    rows, maximum_wfe_error = [], 0.
    for case in cases:
        record, images = load_case(root / "cases" / (case["id"] + ".npz"))
        maximum_wfe_error = max(maximum_wfe_error, check_record(study, case, record, images, gram))
        rows.append(summary_row(record))
    require(read_json(root / "summary.json") == rows, "Per-episode summary differs from retained metadata")
    exposures = sum(row["exposures"] for row in rows)
    require(manifest["episode_count"] == 288 and manifest["exposure_count"] == exposures <= 2016,
            "Manifest terminal or exposure count disagrees")
    reconstructed = independent_analysis(config, cases, rows)
    maximum_report_error = compare_report(read_json(root / "analysis.json"), reconstructed)
    # Recheck inventory and content before declaring success. The input is an
    # immutable local copy; this is not a concurrent-filesystem security boundary.
    check_inventory(root, manifest, study, cases)
    require(digest_file(root / "manifest.json") == manifest_hash, "Manifest changed during audit")
    return {"schema": 1, "audit_status": "PASS", "purpose": PURPOSE, "source_revision": REVISION,
        "configuration_sha256": CONFIG_SHA256, "manifest_sha256": manifest_hash,
        "audit_script_sha256": digest_file(Path(__file__)), "numpy_version": np.__version__,
        "episode_count": 288, "independent_base_count": 24, "conjugate_member_count": 48,
        "episodes_per_stratum": dict(Counter(r["stratum_id"] for r in rows)),
        "episodes_per_gain": {str(g): sum(r["gain"] == g for r in rows) for g in GAINS},
        "episodes_per_arm": dict(Counter(r["family"] for r in rows)),
        "terminal_statuses": dict(Counter(r["status"] for r in rows)), "exposure_count": exposures,
        "expected_photons_consumed": sum(r["expected_photons_used"] for r in rows),
        "fit_count": sum(r["fit_count"] for r in rows),
        "numerical_fit_failures": sum(r["numerical_fit_failures"] for r in rows),
        "optimizer_unsuccessful_fits": sum(r["optimizer_unsuccessful_fits"] for r in rows),
        "finite_pupil_samples": pupil_samples, "max_wfe_absolute_difference_nm": maximum_wfe_error,
        "max_report_absolute_difference_nm": maximum_report_error,
        "comparison_absolute_tolerance_nm": ATOL, "comparison_relative_tolerance": RTOL,
        "analysis_status": reconstructed["status"],
        "independent_equal_stratum": reconstructed.get("equal_stratum"),
        "limits": ["Development-only descriptive result; no controller, hardware or safety qualification.",
                   "WFE reconstructed from simulator truth and retained physical commands, not inferred from camera pixels.",
                   "No estimator refitting or optimality claim; retained fit estimates determine the checked policy path.",
                   "No independent Poisson/Gaussian camera replay, Strehl reconstruction or historical exclusion regeneration.",
                   "Bootstrap preserves full base curves; pointwise percentile intervals are not simultaneous or confirmatory.",
                   "Hash checks do not protect against a malicious concurrent filesystem writer."]}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Extracted complete retained run.")
    parser.add_argument("--out", type=Path, required=True, help="New audit receipt outside the run.")
    args = parser.parse_args(argv)
    require(not args.out.exists() and not args.out.is_symlink(), "Audit receipt already exists")
    require(not args.out.resolve().is_relative_to(args.run.resolve()), "Receipt must remain outside the retained run")
    receipt = audit(args.run)
    with args.out.open("x") as output:
        json.dump(receipt, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"audit_status": receipt["audit_status"], "out": str(args.out),
                      "manifest_sha256": receipt["manifest_sha256"], "analysis_status": receipt["analysis_status"]}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        print("AUDIT FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from None
