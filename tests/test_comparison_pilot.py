"""Retained-public-fixture and injected-failure checks, not benchmark results."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import probebench.comparison_pilot as pilot
from probebench.environment import Episode, Observation
from probebench.optics import OpticalModel

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "results/classical-selection-v1-20261016/selected-settings.json"
RESERVE = 1024**3


class CameraOnlyPolicy:
    """An injected accounting fixture; never a replacement scientific policy."""
    instances = []

    def __init__(self, name, model, config, settings, seed):
        assert seed == 70000001
        assert set(vars(model)) == {"config", "pupil", "basis", "amplitude", "ideal_peak"}
        assert config.max_exposures == 13 and config.photons_per_exposure == 2000
        self.name, self.fit_records, self.counts = name, [], []
        self.instances.append(self)

    def command(self, history):
        assert type(history) is tuple
        for observation in history:
            assert type(observation) is Observation
            assert set(vars(observation)) == {"index", "applied_opd_nm", "expected_photons",
                                             "cumulative_expected_photons", "image_electrons"}
            assert not observation.image_electrons.flags.writeable
        self.counts.append(len(history))
        return np.zeros(6, dtype=np.float64)

    def final_command(self, history):
        return self.command(history)


@pytest.fixture
def injected(monkeypatch):
    CameraOnlyPolicy.instances = []
    monkeypatch.setattr(pilot, "ClassicalPolicy", CameraOnlyPolicy)
    return CameraOnlyPolicy


def run(path):
    return pilot.run_pilot(path, SELECTED.read_bytes(), operating_reserve_bytes=RESERVE)


def rehash(root):
    # Integrity alone is insufficient when an artifact and its unsigned
    # manifest are coherently rewritten by this explicit adversarial fixture.
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in root.rglob("*") if p.is_file() and p != path}
    path.write_text(json.dumps(manifest))


def test_configuration_is_metadata_only_and_exact(monkeypatch):
    monkeypatch.setattr(pilot, "OpticalModel", lambda *_: pytest.fail("Optics constructed"))
    plan = pilot.configuration(SELECTED.read_bytes(), operating_reserve_bytes=RESERVE)
    assert plan["split"] == "public_development_only"
    assert plan["truth_opd_nm"] == [30, 12, -10, 8, -6, 15]
    assert len(plan["families"]) == 6 and plan["resume"] == "refused"
    with pytest.raises(ValueError, match="trusted release"):
        pilot.configuration(SELECTED.read_bytes()+b" ", operating_reserve_bytes=RESERVE)
    with pytest.raises(ValueError, match="reserve"):
        pilot.configuration(SELECTED.read_bytes(), operating_reserve_bytes=0)


def test_configuration_metadata_cannot_mutate_the_public_fixture():
    first = pilot.configuration(SELECTED.read_bytes(), operating_reserve_bytes=RESERVE)
    first["truth_opd_nm"][0] = 999
    first["families"].clear()
    second = pilot.configuration(SELECTED.read_bytes(), operating_reserve_bytes=RESERVE)
    assert second["truth_opd_nm"] == [30, 12, -10, 8, -6, 15]
    assert dict(pilot.FIXTURES)["mixed_a"] == [30, 12, -10, 8, -6, 15]
    assert len(second["families"]) == 6


def test_changed_selected_source_refuses_before_optics_or_output(tmp_path, monkeypatch):
    root, sources = pilot.source_inventory()
    changed = dict(sources)
    changed[next(iter(pilot.SELECTED_SOURCE_SHA256))] = "0"*64
    monkeypatch.setattr(pilot, "source_inventory", lambda: (root, changed))
    monkeypatch.setattr(pilot, "OpticalModel", lambda *_: pytest.fail("Optics constructed"))
    output = tmp_path / "pilot"
    with pytest.raises(ValueError, match="implementation changed"):
        run(output)
    assert not output.exists()


def test_complete_injected_pipeline_pairing_isolation_and_readonly_verification(tmp_path, monkeypatch, injected):
    root = tmp_path / "pilot"
    manifest = run(root)
    assert manifest["episodes"] == 6 and manifest["observations"] == 78
    assert len(injected.instances) == 6
    assert all(policy.counts == list(range(14)) for policy in injected.instances)
    methods = list(pilot.FAMILIES)
    first = np.load(root / methods[0] / "00-image.npy", allow_pickle=False)
    for method in methods[1:]:
        np.testing.assert_array_equal(first, np.load(root / method / "00-image.npy", allow_pickle=False))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(pilot, "OpticalModel", lambda *_: pytest.fail("Verification constructed optics"))
    monkeypatch.setattr(pilot, "ClassicalPolicy", lambda *_: pytest.fail("Verification invoked policy"))
    assert pilot.verify_pilot(root) == manifest
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(FileExistsError, match="fresh"):
        run(root)


@pytest.mark.parametrize("failure", ["exception", "interrupt", "invalid_command", "nonfinite_command"])
def test_policy_failure_preserves_intent_prefix_and_refuses_retry(tmp_path, monkeypatch, injected, failure):
    base = CameraOnlyPolicy.command
    def broken(self, history):
        if len(history) == 2:
            self.fit_records.append({"observations_used": 2, "status": "retained_failed_start"})
            if failure == "exception":
                raise RuntimeError("injected policy failure")
            if failure == "interrupt":
                raise KeyboardInterrupt("injected interruption")
            return np.array([201 if failure == "invalid_command" else np.nan, 0, 0, 0, 0, 0], dtype=float)
        return base(self, history)
    monkeypatch.setattr(CameraOnlyPolicy, "command", broken)
    root = tmp_path / "pilot"
    with pytest.raises((RuntimeError, KeyboardInterrupt, ValueError)):
        run(root)
    folder = root / sorted(pilot.FAMILIES)[0]
    assert not (root / "manifest.json").exists() and (root / "stop.json").is_file()
    assert (folder / "02-policy-intent.json").is_file()
    assert (folder / "00-image.npy").is_file() and (folder / "01-image.npy").is_file()
    assert json.loads((folder / "failed-fits.json").read_text())[0]["status"] == "retained_failed_start"
    failure_record = json.loads((folder / "failure.json").read_text())
    assert failure_record["observations_retained"] == 2
    assert failure_record["camera_calls_started"] == (2 if failure in {"exception", "interrupt"} else 3)
    with pytest.raises(FileExistsError):
        run(root)


def test_nonfinite_produced_image_is_retained(tmp_path, monkeypatch, injected):
    expose = Episode.expose
    def nonfinite(self, *args, **kwargs):
        observation = expose(self, *args, **kwargs)
        image = observation.image_electrons.copy()
        image[0, 0] = np.nan
        changed = Observation(observation.index, observation.applied_opd_nm,
                              observation.expected_photons, observation.cumulative_expected_photons, image)
        self.observations[-1] = changed
        return changed
    monkeypatch.setattr(Episode, "expose", nonfinite)
    root = tmp_path / "pilot"
    with pytest.raises(pilot.NumericalObservationError):
        run(root)
    folder = root / sorted(pilot.FAMILIES)[0]
    assert np.isnan(np.load(folder / "failed-image.npy", allow_pickle=False)[0, 0])
    assert np.isnan(np.load(folder / "failed-produced-00.npy", allow_pickle=False)[0, 0])
    assert json.loads((folder / "failed-evaluation.json").read_text())["failure"]["nonfinite_pixel_count"] == 1
    assert not (root / "manifest.json").exists()


def test_post_camera_evaluator_exception_preserves_produced_raw(tmp_path, monkeypatch, injected):
    def broken(*_):
        raise FloatingPointError("injected evaluator failure after image creation")
    monkeypatch.setattr(OpticalModel, "evaluate", broken)
    root = tmp_path / "pilot"
    with pytest.raises(FloatingPointError):
        run(root)
    folder = root / sorted(pilot.FAMILIES)[0]
    assert np.load(folder / "failed-produced-00.npy", allow_pickle=False).shape == (64, 64)
    assert not (folder / "00-observation.json").exists()
    assert not (root / "manifest.json").exists()


def test_finite_nonconvergence_and_fallback_records_are_not_excluded(tmp_path, monkeypatch, injected):
    base = CameraOnlyPolicy.final_command
    def fallback(self, history):
        self.fit_records.extend([{"status": "nonconverged", "estimate_opd_nm": [0.0]*6},
                                 {"status": "camera_only_fallback", "all_starts": [{"success": False}]}])
        return base(self, history)
    monkeypatch.setattr(CameraOnlyPolicy, "final_command", fallback)
    root = tmp_path / "pilot"
    run(root)
    pilot.verify_pilot(root)
    for method in pilot.FAMILIES:
        assert len(json.loads((root / method / "13-fits.json").read_text())) == 2


@pytest.mark.parametrize("tamper", ["extra", "missing", "command", "history", "photons", "huge_header",
    "physical_command", "mirror_surface", "missing_trajectory", "trajectory_command", "trajectory_photons",
    "final_indices", "observed_flag", "last_flag", "nominal_schema", "public_evaluator_field",
    "missing_final_metrics", "missing_frame_metrics", "peak_probe"])
def test_verifier_rejects_coherent_manifest_tampering(tmp_path, injected, tamper):
    root = tmp_path / "pilot"
    run(root)
    folder = root / sorted(pilot.FAMILIES)[0]
    if tamper == "extra":
        (root / "unexpected.json").write_text("{}")
    elif tamper == "missing":
        (folder / "13-policy-intent.json").unlink()
    elif tamper == "command":
        np.save(folder / "00-command.npy", np.ones(6))
    elif tamper == "history":
        (folder / "13-policy-intent.json").write_text('{"stage":"final_command","observations_available":12}')
    elif tamper == "huge_header":
        with (folder / "00-image.npy").open("wb") as stream:
            np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False,
                                                         "shape": (10**12, 10**12)})
    elif tamper == "photons":
        path = folder / "00-observation.json"
        record = json.loads(path.read_text())
        record["public"]["cumulative_expected_photons"] += 1
        path.write_text(json.dumps(record))
    else:
        path = folder / "evaluation.json"
        evaluation = json.loads(path.read_text())
        physical = evaluation["physical"]
        if tamper == "physical_command":
            physical["final_applied_opd_nm"][0] = 5.0
        elif tamper == "mirror_surface":
            physical["final_mirror_surface_coefficients_nm"][0] = 5.0
        elif tamper == "missing_trajectory":
            del physical["trajectory"]
        elif tamper == "trajectory_command":
            physical["trajectory"][0]["applied_opd_nm"][0] = 5.0
        elif tamper == "trajectory_photons":
            physical["trajectory"][0]["expected_photons"] += 1
        elif tamper == "final_indices":
            physical["final_command_observation_indices"] = []
        elif tamper == "observed_flag":
            physical["final_command_has_camera_observation"] = False
        elif tamper == "last_flag":
            physical["final_command_is_last_observation"] = False
        elif tamper == "missing_final_metrics":
            del physical["final"]
        elif tamper == "missing_frame_metrics":
            del physical["trajectory"][0]["residual_wfe_nm"]
        elif tamper == "peak_probe":
            physical["peak_probe_wfe_increase_nm"] += 1
        else:
            observation_path = folder / "00-observation.json"
            observation = json.loads(observation_path.read_text())
            if tamper == "nominal_schema":
                observation["public"]["schema"] = "changed"
            else:
                observation["public"]["evaluator_truth"] = [30, 12, -10, 8, -6, 15]
            # Update both views to bypass the simple cross-file equality check.
            evaluation["nominal_observations"][0] = observation["public"]
            observation_path.write_text(json.dumps(observation))
        path.write_text(json.dumps(evaluation))
    rehash(root)
    with pytest.raises(ValueError):
        pilot.verify_pilot(root)


@pytest.mark.parametrize("kind", ["oversize", "symlink", "aggregate", "entry_count", "permission"])
def test_packet_bounds_precede_native_hash_or_array_reads(tmp_path, monkeypatch, kind):
    root = tmp_path / "pilot"
    root.mkdir()
    monkeypatch.setattr(pilot, "verify_study", lambda *_: pytest.fail("Native unbounded reader reached"))
    monkeypatch.setattr(pilot.np, "load", lambda *_args, **_kwargs: pytest.fail("Array load reached"))
    if kind == "oversize":
        # A sparse fixture checks the stat guard without allocating large bytes.
        with (root / "manifest.json").open("wb") as stream:
            stream.seek(pilot.MAX_FILE_BYTES)
            stream.write(b"x")
    elif kind == "symlink":
        (root / "link").symlink_to(tmp_path / "missing")
    elif kind == "aggregate":
        monkeypatch.setattr(pilot, "MAX_PACKET_BYTES", 4)
        (root / "a").write_bytes(b"123")
        (root / "b").write_bytes(b"456")
    elif kind == "entry_count":
        monkeypatch.setattr(pilot, "MAX_PACKET_ENTRIES", 1)
        (root / "a").mkdir()
        (root / "b").mkdir()
    else:
        def denied(*_args, **kwargs):
            kwargs["onerror"](PermissionError("injected unreadable directory"))
        monkeypatch.setattr(pilot.os, "fwalk", denied)
    with pytest.raises((ValueError, PermissionError)):
        pilot.verify_pilot(root)


def test_cli_bounds_selected_bytes_before_native_runner(tmp_path, monkeypatch):
    selected = tmp_path / "selected.json"
    with selected.open("wb") as stream:
        stream.seek(pilot.MAX_FILE_BYTES)
        stream.write(b"x")
    monkeypatch.setattr(pilot, "run_pilot", lambda *_args, **_kwargs: pytest.fail("Runner reached"))
    monkeypatch.setattr("sys.argv", ["comparison_pilot", str(tmp_path / "output"),
        "--selected-settings", str(selected), "--operating-reserve-bytes", str(RESERVE)])
    with pytest.raises(ValueError, match="file allowance"):
        pilot.main()
