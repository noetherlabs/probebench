"""Fresh-only qualification of frozen classical policies on a public fixture.

This is not a held-out comparison. There is no cohort sampler, private key,
choice of incident wavefront, policy reselection or automatic retry interface.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import stat
import time

import numpy as np

from .budget_policies import ClassicalPolicy, PolicyConfig
from .comparison_contract import FAMILIES, SELECTED_SOURCE_SHA256, _selected, canonical_bytes, parse_json
from .environment import EpisodeConfig
from .nominal_calibration import GainEpisode, NominalObservation, NumericalObservationError
from .optics import OpticalModel, OpticsConfig
from .run import FIXTURES, digest, source_inventory
from .study import publish_manifest, study_directory, verify_study

PURPOSE = "classical_public_fixture_pilot_v1"
EXPOSURES = 13
PHOTONS = 2000
# Reuse the existing public measurement_schedule_development.json streams.
DETECTOR_SEED = 60000001
POLICY_SEED = 70000001
MIN_RESERVE_BYTES = 1024**3
MAX_PACKET_BYTES = 128 * 1024**2
MAX_FILE_BYTES = 1_048_576
MAX_PACKET_ENTRIES = 2048


def configuration(selected_settings_bytes, *, operating_reserve_bytes):
    """Metadata only; exact selected bytes are required before any optical work."""
    selected = _selected(selected_settings_bytes)
    if type(operating_reserve_bytes) is not int or operating_reserve_bytes < MIN_RESERVE_BYTES:
        raise ValueError("Pilot operating reserve must be an integer of at least 1 GiB")
    return {"schema": 1, "purpose": PURPOSE, "split": "public_development_only",
            "fixture": "mixed_a", "truth_opd_nm": list(dict(FIXTURES)["mixed_a"]),
            "optics": {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0},
            "episode": asdict(EpisodeConfig(max_exposures=EXPOSURES,
                photon_budget=EXPOSURES*PHOTONS, photons_per_exposure=PHOTONS)),
            "selected_settings_sha256": hashlib.sha256(selected_settings_bytes).hexdigest(),
            "families": selected["families"], "detector_seed": DETECTOR_SEED,
            "policy_seed": POLICY_SEED, "command_gain": 1.0,
            "operating_reserve_bytes": operating_reserve_bytes,
            "inference": "none_public_fixture_reuse", "resume": "refused"}


def _write(path, payload):
    """Exclusive durable publication; an interrupted partial file is preserved."""
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path, value):
    _write(path, canonical_bytes(_plain(value)) + b"\n")


def _plain(value, *, retain_nonfinite=False):
    if isinstance(value, np.generic):
        value = value.item()
    if retain_nonfinite and type(value) is float and not math.isfinite(value):
        return {"nonfinite_float": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
    if type(value) is list:
        return [_plain(item, retain_nonfinite=retain_nonfinite) for item in value]
    if type(value) is dict:
        return {key: _plain(item, retain_nonfinite=retain_nonfinite) for key, item in value.items()}
    return value


def _array(path, value):
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays are not permitted in pilot artifacts")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": path.name, "sha256": digest(path), "shape": list(array.shape),
            "dtype": array.dtype.str}


def _fit_snapshot(path, records):
    # Normal native records are JSON values. A nonfinite diagnostic remains
    # visible as a tagged value, rather than being dropped or scored as finite.
    _json(path, _plain(records, retain_nonfinite=True))


def _episode(root, method, settings, model, config):
    """Persist intent before each callback; return only after terminal evidence."""
    root.mkdir()
    episode = policy = None
    stage, index, camera_calls = "initialization", 0, 0
    started = time.monotonic()
    _json(root / "identity.json", {"method": method, "fixture": "mixed_a",
                                  "policy": asdict(settings)})
    try:
        # Hidden state and detector randomness remain in the driver. The policy
        # receives only calibration, bounds, selected settings and its RNG seed.
        episode = GainEpisode(model, config, dict(FIXTURES)["mixed_a"], DETECTOR_SEED,
                              command_gain=1.0)
        policy = ClassicalPolicy(method, model, config, settings, POLICY_SEED)
        for index in range(EXPOSURES + 1):
            stage = "final_command" if index == EXPOSURES else "command"
            stem = f"{index:02d}"
            _json(root / f"{stem}-policy-intent.json", {"stage": stage,
                  "observations_available": len(episode.observations)})
            command_started = time.monotonic()
            callback = policy.final_command if index == EXPOSURES else policy.command
            command = callback(episode._policy_history())
            command_artifact = _array(root / f"{stem}-command.npy", command)
            _fit_snapshot(root / f"{stem}-fits.json", policy.fit_records)
            _json(root / f"{stem}-policy-result.json", {"command": command_artifact,
                  "wall_seconds": time.monotonic() - command_started})
            stage = "submit" if index == EXPOSURES else "exposure"
            _json(root / f"{stem}-camera-intent.json", {"stage": stage,
                  "command_sha256": command_artifact["sha256"]})
            if index == EXPOSURES:
                evaluation = episode.submit(command)
                _fit_snapshot(root / "evaluation.json", evaluation)
                # This refusal is after retaining all commands, images and fits.
                canonical_bytes(_plain(evaluation))
                if evaluation["status"] != "submitted":
                    raise ValueError("Pilot evaluator did not submit a finite result")
            else:
                camera_calls += 1
                observation = episode.expose(command)
                artifact = _array(root / f"{stem}-image.npy", observation.image_electrons)
                _json(root / f"{stem}-observation.json", {"public": observation.metadata(),
                                                        "image": artifact})
        terminal = {"method": method, "status": "complete", "camera_calls_started": camera_calls,
                    "observations_retained": len(episode.observations),
                    "expected_photons_used": evaluation["physical"]["expected_photons_used"],
                    "fit_count": len(policy.fit_records), "wall_seconds": time.monotonic()-started}
        _json(root / "terminal.json", terminal)
        return terminal
    except BaseException as error:
        # Leave every existing intent/frame untouched. In particular, a killed
        # writer may leave an intent with an unknown outcome; it is never retried.
        _json(root / "failure.json", {"method": method, "status": "stopped", "stage": stage,
              "index": index, "error_type": type(error).__name__, "error": str(error)[:2048],
              "camera_calls_started": camera_calls,
              "observations_retained": 0 if episode is None else len(episode.observations),
              "wall_seconds": time.monotonic()-started})
        # Episode can fail while evaluating an already produced camera frame,
        # before GainEpisode creates its public snapshot. Preserve that private
        # evidence too. This access is confined to the trusted driver.
        if episode is not None:
            for produced in episode._episode.observations:
                _array(root / f"failed-produced-{produced.index:02d}.npy", produced.image_electrons)
                _json(root / f"failed-produced-{produced.index:02d}.json", produced.metadata())
        if episode is not None and episode.failed_observation is not None:
            failed = episode.failed_observation
            _array(root / "failed-image.npy", failed.image_electrons)
            _json(root / "failed-observation.json", failed.metadata())
            if isinstance(error, NumericalObservationError):
                _fit_snapshot(root / "failed-evaluation.json", episode.evaluator_record())
        if policy is not None:
            _fit_snapshot(root / "failed-fits.json", policy.fit_records)
        raise


def run_pilot(output, selected_settings_bytes, *, operating_reserve_bytes):
    """Execute the one fixed public fixture in a fresh native study directory.

    External admission must enforce duration, process-group memory and disk
    bounds. This function does not grant admission or lower the caller's reserve.
    """
    config = configuration(selected_settings_bytes, operating_reserve_bytes=operating_reserve_bytes)
    _, sources = source_inventory()
    if any(sources.get(path) != expected for path, expected in SELECTED_SOURCE_SHA256.items()):
        raise ValueError("Selected optical or policy implementation changed")
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Pilot output must be fresh; partial evidence cannot be resumed")
    raw_bytes = len(FAMILIES)*EXPOSURES*64*64*8 + 16*1024**2
    with study_directory(root, config, raw_bytes, resume=False) as study:
        try:
            _write(root / "selected-settings.json", selected_settings_bytes)
            model = OpticalModel(OpticsConfig(**config["optics"]))
            episode_config = EpisodeConfig(**config["episode"])
            summaries = []
            for method, selected in config["families"].items():
                summaries.append(_episode(root / method, method, PolicyConfig(**selected["policy"]),
                                          model, episode_config))
            _json(root / "summary.json", {"purpose": PURPOSE, "inference": config["inference"],
                                         "episodes": summaries})
            return publish_manifest(root, study, PURPOSE, episodes=len(summaries),
                                    observations=len(summaries)*EXPOSURES)
        except BaseException as error:
            _json(root / "stop.json", {"status": "incomplete", "error_type": type(error).__name__,
                                      "error": str(error)[:2048]})
            raise


def _preflight_packet(root):
    """Bound a stable retained packet before native helpers read/hash its bytes."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Pilot packet must be a regular directory")
    total, count = 0, 0
    def scan_error(error):
        raise error
    for _, directories, files, descriptor in os.fwalk(root, follow_symlinks=False, onerror=scan_error):
        count += len(directories) + len(files)
        if count > MAX_PACKET_ENTRIES:
            raise ValueError("Pilot packet entry count exceeds its allowance")
        for name in directories:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Pilot packet contains an unsafe directory")
        for name in files:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                raise ValueError("Pilot packet file exceeds its allowance or is unsafe")
            total += info.st_size
            if total > MAX_PACKET_BYTES:
                raise ValueError("Pilot packet exceeds its byte allowance")


def _read_selected(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("Selected settings exceed their file allowance")
        payload = stream.read(MAX_FILE_BYTES+1)
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("Selected settings exceed their file allowance")
    return payload


def verify_pilot(output):
    """Read retained bytes without constructing optics, invoking policies or refitting."""
    root = Path(output)
    _preflight_packet(root)
    manifest, study = verify_study(root, PURPOSE)
    expected = configuration(_read_selected(root / "selected-settings.json"),
                            operating_reserve_bytes=study["configuration"]["operating_reserve_bytes"])
    if canonical_bytes(study["configuration"]) != canonical_bytes(expected):
        raise ValueError("Pilot configuration changed")
    if any(study["source_sha256"].get(path) != expected_hash
           for path, expected_hash in SELECTED_SOURCE_SHA256.items()):
        raise ValueError("Selected optical or policy source binding changed")
    if manifest["episodes"] != len(FAMILIES) or manifest["observations"] != len(FAMILIES)*EXPOSURES:
        raise ValueError("Incomplete pilot inventory")
    if "stop.json" in manifest["files"] or any(name.endswith("failure.json") for name in manifest["files"]):
        raise ValueError("Stopped pilot cannot have a completion manifest")
    expected_files = {".lock", "study.json", "selected-settings.json", "summary.json"}
    expected_files.update(f"source/{path}" for path in study["source_sha256"])
    for method in expected["families"]:
        expected_files.update(f"{method}/{name}" for name in ("identity.json", "terminal.json", "evaluation.json"))
        for index in range(EXPOSURES + 1):
            expected_files.update(f"{method}/{index:02d}-{suffix}" for suffix in
                                  ("policy-intent.json", "command.npy", "fits.json",
                                   "policy-result.json", "camera-intent.json"))
            if index < EXPOSURES:
                expected_files.update((f"{method}/{index:02d}-image.npy",
                                       f"{method}/{index:02d}-observation.json"))
    if set(manifest["files"]) != expected_files:
        raise ValueError("Unexpected or missing pilot artifact")
    def read_json(path):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Pilot JSON exceeds its byte allowance")
        return parse_json(path.read_bytes())
    def read_array(path, shape):
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Pilot array exceeds its byte allowance")
        with path.open("rb") as stream:
            if np.lib.format.read_magic(stream) != (1, 0):
                raise ValueError("Unexpected pilot NPY format")
            actual_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
            if (actual_shape != shape or fortran or dtype != np.dtype("float64")
                    or path.stat().st_size != stream.tell() + math.prod(shape)*8):
                raise ValueError("Invalid pilot NPY header or length")
        array = np.load(path, allow_pickle=False, max_header_size=4096)
        if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("Invalid retained pilot array")
        return array
    summary = read_json(root / "summary.json")
    if summary["purpose"] != PURPOSE or summary["inference"] != expected["inference"]:
        raise ValueError("Changed pilot interpretation")
    if [row["method"] for row in summary["episodes"]] != list(expected["families"]):
        raise ValueError("Pilot method inventory mismatch")
    for row in summary["episodes"]:
        method = row["method"]
        terminal = read_json(root / method / "terminal.json")
        if (terminal != row or row["status"] != "complete" or row["camera_calls_started"] != EXPOSURES
                or row["observations_retained"] != EXPOSURES
                or row["expected_photons_used"] != EXPOSURES*PHOTONS):
            raise ValueError("Incomplete pilot episode")
        identity = read_json(root / method / "identity.json")
        if identity != {"method": method, "fixture": "mixed_a", "policy": expected["families"][method]["policy"]}:
            raise ValueError("Changed pilot policy identity")
        previous_fits, observations = [], []
        for index in range(EXPOSURES + 1):
            folder, stem = root / method, f"{index:02d}"
            command = read_array(folder / f"{stem}-command.npy", (6,))
            if np.any(np.abs(command) > expected["episode"]["command_limit_nm"]):
                raise ValueError("Retained command exceeds the instrument envelope")
            policy_intent = read_json(folder / f"{stem}-policy-intent.json")
            if policy_intent != {"stage": "final_command" if index == EXPOSURES else "command",
                                 "observations_available": index}:
                raise ValueError("Invalid pilot policy dispatch")
            camera_intent = read_json(folder / f"{stem}-camera-intent.json")
            if camera_intent != {"stage": "submit" if index == EXPOSURES else "exposure",
                                 "command_sha256": digest(folder / f"{stem}-command.npy")}:
                raise ValueError("Invalid pilot camera dispatch")
            policy_result = read_json(folder / f"{stem}-policy-result.json")
            if (policy_result["command"] != {"file": f"{stem}-command.npy",
                    "sha256": digest(folder / f"{stem}-command.npy"), "shape": [6], "dtype": "<f8"}
                    or type(policy_result["wall_seconds"]) not in (int, float)
                    or policy_result["wall_seconds"] < 0):
                raise ValueError("Invalid pilot command artifact binding")
            fits = read_json(folder / f"{stem}-fits.json")
            if type(fits) is not list or fits[:len(previous_fits)] != previous_fits:
                raise ValueError("Retained fit history changed")
            previous_fits = fits
            if index < EXPOSURES:
                observation = read_json(folder / f"{stem}-observation.json")
                image = read_array(folder / f"{stem}-image.npy", (64, 64))
                public = observation["public"]
                if observation["image"] != {"file": f"{stem}-image.npy",
                        "sha256": digest(folder / f"{stem}-image.npy"), "shape": [64, 64], "dtype": "<f8"}:
                    raise ValueError("Invalid pilot image artifact binding")
                reconstructed = NominalObservation(index, tuple(command), PHOTONS,
                                                    (index+1)*PHOTONS, image)
                if canonical_bytes(public) != canonical_bytes(_plain(reconstructed.metadata())):
                    raise ValueError("Invalid retained pilot observation")
                observations.append(public)
        evaluation = read_json(root / method / "evaluation.json")
        if (evaluation["status"] != "submitted" or evaluation["command_gain"] != 1.0
                or evaluation["nominal_observations"] != observations
                or evaluation["final_commanded_opd_nm"] != command.tolist()
                or evaluation["physical"]["hidden_opd_nm"] != expected["truth_opd_nm"]
                or evaluation["physical"]["exposures"] != EXPOSURES
                or evaluation["physical"]["expected_photons_used"] != EXPOSURES*PHOTONS
                or row["fit_count"] != len(previous_fits)):
            raise ValueError("Invalid retained pilot evaluation")
        physical = evaluation["physical"]
        metric_keys = {"residual_wfe_nm", "strehl_peak_ratio"}
        for metrics in (physical.get("initial"), physical.get("final")):
            if (type(metrics) is not dict or set(metrics) != metric_keys
                    or any(type(value) not in (int, float) or value < 0 for value in metrics.values())):
                raise ValueError("Missing or invalid finite physical metrics")
        final_indices = [o["index"] for o in observations if o["commanded_opd_nm"] == command.tolist()]
        if (physical.get("final_applied_opd_nm") != command.tolist()
                or physical.get("final_mirror_surface_coefficients_nm") != (command/2).tolist()
                or physical.get("final_command_observation_indices") != final_indices
                or physical.get("final_command_has_camera_observation") is not bool(final_indices)
                or physical.get("final_command_is_last_observation") is not (EXPOSURES-1 in final_indices)):
            raise ValueError("Physical final command disagrees with retained observations")
        trajectory = physical.get("trajectory")
        if type(trajectory) is not list or len(trajectory) != EXPOSURES:
            raise ValueError("Incomplete physical observation trajectory")
        for public, produced in zip(observations, trajectory):
            fields = {"index": public["index"], "applied_opd_nm": public["commanded_opd_nm"],
                      "expected_photons": PHOTONS,
                      "cumulative_expected_photons": public["cumulative_expected_photons"]}
            if type(produced) is not dict or any(produced.get(key) != value for key, value in fields.items()):
                raise ValueError("Physical trajectory disagrees with retained observations")
            if (not metric_keys <= set(produced)
                    or any(type(produced[key]) not in (int, float) or produced[key] < 0 for key in metric_keys)
                    or type(produced.get("realized_photoelectrons")) is not int
                    or produced["realized_photoelectrons"] < 0):
                raise ValueError("Missing or invalid produced-frame metrics")
        expected_peak = max(0.0, max(item["residual_wfe_nm"] for item in trajectory)
                            - physical["initial"]["residual_wfe_nm"])
        if physical.get("peak_probe_wfe_increase_nm") != expected_peak:
            raise ValueError("Physical peak probe summary disagrees with trajectory")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selected-settings", type=Path)
    parser.add_argument("--operating-reserve-bytes", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.selected_settings is not None or args.operating_reserve_bytes is not None:
            parser.error("Verification takes only an existing output directory")
        verify_pilot(args.output)
    else:
        if args.selected_settings is None or args.operating_reserve_bytes is None:
            parser.error("Acquisition requires selected settings and an explicit operating reserve")
        run_pilot(args.output, _read_selected(args.selected_settings),
                  operating_reserve_bytes=args.operating_reserve_bytes)


if __name__ == "__main__":
    main()
