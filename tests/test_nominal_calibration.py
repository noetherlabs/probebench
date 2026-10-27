import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from io import BytesIO

import numpy as np
import pytest

import probebench.budget_policies as policies
import probebench.nominal_calibration as nominal
from probebench.environment import Episode, EpisodeConfig, Observation
from probebench.optics import OpticalModel, OpticsConfig


@pytest.fixture
def model():
    return OpticalModel(OpticsConfig(pupil_pixels=12, oversampling=1))


def config(count=3, photons=100, limit=200):
    return EpisodeConfig(max_exposures=count, photon_budget=count*photons,
                         photons_per_exposure=photons, command_limit_nm=limit)


TRUTH = [30, 12, -10, 8, -6, 15]


def test_public_schema_is_nominal_and_legacy_schema_is_unchanged(model):
    episode = nominal.GainEpisode(model, config(), TRUTH, 42, command_gain=0.9)
    observed = episode.expose([80, 0, 0, 0, 0, 0])
    assert {f.name for f in fields(observed)} == {
        "index", "commanded_opd_nm", "expected_photons", "cumulative_expected_photons", "image_electrons"}
    assert observed.metadata() == {
        "schema": "probebench.nominal-observation/1", "index": 0,
        "commanded_opd_nm": [80, 0, 0, 0, 0, 0],
        "nominal_applied_opd_nm": [80, 0, 0, 0, 0, 0],
        "nominal_calibration": {"response": "identity", "command_units": "nm_nominal_opd",
                                "prediction_units": "nm_opd"},
        "expected_photons": 100, "cumulative_expected_photons": 100}
    json.dumps(observed.metadata(), allow_nan=False)
    assert {f.name for f in fields(Observation)} == {
        "index", "applied_opd_nm", "expected_photons", "cumulative_expected_photons", "image_electrons"}
    assumed = episode._policy_history()[0]
    assert type(assumed) is Observation
    assert assumed.applied_opd_nm == (80, 0, 0, 0, 0, 0)
    assert set(assumed.metadata()) == {
        "index", "applied_opd_nm", "expected_photons", "cumulative_expected_photons"}


@pytest.mark.parametrize("gain", [0.5, 0.9, 1.0, 1.1, 2.0])
def test_camera_and_final_metrics_use_actual_scaled_opd(model, gain):
    cfg = config()
    gained = nominal.GainEpisode(model, cfg, TRUTH, 42, command_gain=gain)
    physical = Episode(model, replace(cfg, command_limit_nm=gain*cfg.command_limit_nm), TRUTH, 42)
    commands = [[0]*6, [80, 0, 0, 0, 0, 0], [-20, 10, 4, 6, -7, 3]]
    for command in commands:
        observed = gained.expose(command)
        independent = physical.expose(gain*np.asarray(command))
        np.testing.assert_array_equal(observed.image_electrons, independent.image_electrons)
        assert observed.commanded_opd_nm == tuple(command)
        assert not np.shares_memory(observed.image_electrons, independent.image_electrons)
    result = gained.submit(commands[1])
    expected = physical.submit(gain*np.asarray(commands[1]))
    assert result["physical"] == expected
    assert result["command_gain"] == gain
    assert result["final_commanded_opd_nm"] == commands[1]
    assert result["physical"]["final_applied_opd_nm"] == (gain*np.asarray(commands[1])).tolist()
    assert result["physical"]["final_command_observation_indices"] == [1]
    assert not result["physical"]["final_command_is_last_observation"]
    assert result["physical"]["exposures"] == 3


def test_zero_command_public_output_cannot_reveal_gain(model):
    histories = []
    for gain in (0.5, 1.0, 2.0):
        episode = nominal.GainEpisode(model, config(), TRUTH, 42, command_gain=gain)
        histories.append(tuple(episode.expose([0]*6) for _ in range(3)))
    for history in histories[1:]:
        for baseline, compared in zip(histories[0], history):
            assert baseline.metadata() == compared.metadata()
            np.testing.assert_array_equal(baseline.image_electrons, compared.image_electrons)


def test_observations_are_immutable_and_detached_from_camera_and_inputs(model):
    episode = nominal.GainEpisode(model, config(), TRUTH, 42, command_gain=1.1)
    command = np.array([80., 0, 0, 0, 0, 0])
    public = episode.expose(command)
    original = public.image_electrons.copy()
    command[:] = 0
    with pytest.raises(FrozenInstanceError):
        public.commanded_opd_nm = (0,)*6
    with pytest.raises(ValueError):
        public.image_electrons[0, 0] = 0
    with pytest.raises(ValueError):
        public.image_electrons.setflags(write=True)
    metadata = public.metadata()
    metadata["commanded_opd_nm"][0] = -1
    metadata["nominal_calibration"]["response"] = "other"
    private_image = episode._episode.observations[0].image_electrons
    private_image.setflags(write=True)
    private_image[:] = 999
    assert public.commanded_opd_nm[0] == 80
    assert public.metadata()["nominal_calibration"]["response"] == "identity"
    np.testing.assert_array_equal(public.image_electrons, original)
    assert not np.shares_memory(public.image_electrons, private_image)
    assert (public.image_electrons < 0).any()


@pytest.mark.parametrize("changes", [
    {"index": -1}, {"index": True}, {"expected_photons": 0},
    {"expected_photons": True}, {"cumulative_expected_photons": 99},
    {"commanded_opd_nm": [np.inf]*6}, {"commanded_opd_nm": np.ones(6, dtype=complex)},
    {"image_electrons": np.zeros(3)}, {"image_electrons": np.zeros((0, 3))},
    {"image_electrons": np.full((2, 2), np.nan)},
    {"image_electrons": np.ones((2, 2), dtype=complex)},
    {"image_electrons": np.array([["0"]])},
])
def test_malformed_public_observation_is_rejected(changes):
    fields = {"index": 0, "commanded_opd_nm": [0]*6, "expected_photons": 100,
              "cumulative_expected_photons": 100, "image_electrons": np.ones((2, 2))}
    fields.update(changes)
    with pytest.raises(ValueError):
        nominal.NominalObservation(**fields)


@pytest.mark.parametrize("gain", [0, -1, np.inf, -np.inf, np.nan, True, np.bool_(True), "1", 1j,
                                  10**500, np.finfo(float).max])
def test_invalid_or_unrepresentable_gain_refuses_before_constructing_camera(model, monkeypatch, gain):
    def forbidden(*args, **kwargs):
        pytest.fail("Camera constructed before gain validation")
    monkeypatch.setattr(nominal, "Episode", forbidden)
    with pytest.raises(ValueError):
        nominal.GainEpisode(model, config(), TRUTH, 42, command_gain=gain)


def test_underflowed_physical_limit_is_rejected(model):
    with pytest.raises(ValueError, match="representable"):
        nominal.GainEpisode(model, config(limit=1e-200), TRUTH, 42, command_gain=1e-200)


@pytest.mark.parametrize("command,photons", [([201, 0, 0, 0, 0, 0], None), ([np.nan]*6, None),
    ([0]*5, None), ([1j, 0, 0, 0, 0, 0], None), ([0]*6, 301), ([0]*6, True), ([0]*6, -1)])
def test_invalid_action_preserves_detector_stream_and_accounting(model, command, photons):
    episode = nominal.GainEpisode(model, config(), TRUTH, 42, command_gain=1.1)
    control = nominal.GainEpisode(model, config(), TRUTH, 42, command_gain=1.1)
    with pytest.raises((ValueError, TypeError)):
        episode.expose(command, photons)
    assert episode.observations == ()
    measured = episode.expose([80, 0, 0, 0, 0, 0])
    expected = control.expose([80, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(measured.image_electrons, expected.image_electrons)
    assert measured.cumulative_expected_photons == 100 and measured.index == 0


def test_tiny_nonzero_command_cannot_silently_underflow(model):
    episode = nominal.GainEpisode(model, config(), TRUTH, 42, command_gain=0.5)
    with pytest.raises(ValueError, match="representable"):
        episode.expose([np.nextafter(0., 1.), 0, 0, 0, 0, 0])
    assert episode.observations == ()


def test_produced_nonfinite_image_is_retained_as_terminal_failure(model):
    cfg = replace(config(), read_noise_electrons=np.finfo(float).max)
    episode = nominal.GainEpisode(model, cfg, TRUTH, 42, command_gain=1)
    assert episode.failed_observation is None
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(nominal.NumericalObservationError, match="terminal"):
            episode.expose([0]*6)
    assert episode.observations == ()
    failed = episode.failed_observation
    assert failed.index == 0 and failed.expected_photons == 100
    assert not np.isfinite(failed.image_electrons).all()
    assert not failed.image_electrons.flags.writeable
    with pytest.raises(ValueError):
        failed.image_electrons.setflags(write=True)
    np.testing.assert_array_equal(failed.image_electrons,
                                  episode._episode.observations[0].image_electrons)
    payload = BytesIO()
    np.save(payload, failed.image_electrons, allow_pickle=False)
    payload.seek(0)
    np.testing.assert_array_equal(np.load(payload, allow_pickle=False), failed.image_electrons)
    record = episode.evaluator_record()
    json.dumps(record, allow_nan=False)
    assert record["status"] == "numerical_failure"
    assert record["final_commanded_opd_nm"] is None
    assert "final" not in record["physical"]
    assert record["physical"]["exposures"] == 1
    assert record["physical"]["expected_photons_used"] == 100
    assert len(record["physical"]["trajectory"]) == 1
    assert record["failure"]["public_observation_count"] == 0
    assert record["failure"]["produced_observation_count"] == 1
    assert record["failure"]["nonfinite_pixel_count"] > 0
    assert record["failure"]["failed_physical_observation"] == failed.metadata()
    rng_state = deepcopy(episode._episode._rng.bit_generator.state)
    for action in (episode.expose, episode.submit):
        with pytest.raises(RuntimeError, match="terminal"):
            action([0]*6)
    assert episode._episode._rng.bit_generator.state == rng_state
    assert episode.evaluator_record() == record
    record["failure"]["commanded_opd_nm"][0] = 99
    assert episode.evaluator_record()["failure"]["commanded_opd_nm"][0] == 0


def test_runner_returns_failed_raw_evidence_without_final_policy_callback(model, monkeypatch):
    cfg = replace(config(), read_noise_electrons=np.finfo(float).max)
    def forbidden(*args):
        pytest.fail("A terminal camera failure cannot produce a policy final command")
    monkeypatch.setattr(nominal.ClassicalPolicy, "final_command", forbidden)
    with np.errstate(over="ignore", invalid="ignore"):
        result = nominal.run_classical_episode(model, cfg, TRUTH, 42, command_gain=1,
                                              method="no_action_v2")
    assert result["evaluation"]["status"] == "numerical_failure"
    assert result["observations"] == ()
    assert result["fit_records"] == []
    assert result["failed_observation"].index == 0
    assert not np.isfinite(result["failed_observation"].image_electrons).all()
    assert result["evaluation"]["physical"]["expected_photons_used"] == 100


def test_nominal_bound_and_submission_state_do_not_leak_scaled_limit(model):
    episode = nominal.GainEpisode(model, config(count=1), TRUTH, 42, command_gain=2)
    with pytest.raises(ValueError, match="nominal"):
        episode.expose([201, 0, 0, 0, 0, 0])
    episode.expose([200, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="budget"):
        episode.expose([0]*6)
    result = episode.submit([200, 0, 0, 0, 0, 0])
    assert result["physical"]["final_applied_opd_nm"][0] == 400
    for operation in (episode.expose, episode.submit):
        with pytest.raises(RuntimeError, match="submitted"):
            operation([0]*6)
    result["physical"]["trajectory"][0]["applied_opd_nm"][0] = 0
    assert episode.evaluator_record()["physical"]["trajectory"][0]["applied_opd_nm"][0] == 400


def test_final_submission_before_observation_does_not_close_episode(model):
    episode = nominal.GainEpisode(model, config(count=1), TRUTH, 42, command_gain=1.1)
    with pytest.raises(ValueError, match="observation"):
        episode.submit([0]*6)
    with pytest.raises(RuntimeError, match="submission"):
        episode.evaluator_record()
    episode.expose([0]*6)
    assert episode.submit([0]*6)["physical"]["final_command_is_last_observation"]


def test_estimator_receives_nominal_history_and_original_public_instrument(model, monkeypatch):
    cfg = config(count=4)
    seen = []
    original = nominal.ClassicalPolicy

    def construct(method, public_model, public_config, settings, seed):
        assert public_model is model and public_config is cfg
        assert public_config.command_limit_nm == 200
        return original(method, public_model, public_config, settings, seed)

    def estimate(public_model, history, *args, **kwargs):
        assert public_model is model
        assert all(type(o) is Observation for o in history)
        seen.append([o.applied_opd_nm for o in history])
        return np.array([30., 0, 0, 0, 0, 0]), {"optimizer_success": True, "starts": []}

    monkeypatch.setattr(nominal, "ClassicalPolicy", construct)
    monkeypatch.setattr(policies, "estimate_phase_mle", estimate)
    result = nominal.run_classical_episode(model, cfg, TRUTH, 42, command_gain=0.5,
                                           method="phase_diversity_mle_v1")
    assert [len(h) for h in seen] == [2, 3, 4]
    assert [o.commanded_opd_nm[0] for o in result["observations"]] == [0, 80, 30, -50]
    assert [o["applied_opd_nm"][0] for o in result["evaluation"]["physical"]["trajectory"]] == [0, 40, 15, -25]
    for history in seen:
        assert [o[0] for o in history] == [0, 80, 30, -50][:len(history)]
    assert result["evaluation"]["physical"]["final_applied_opd_nm"][0] == 15


def without_timings(records):
    result = deepcopy(records)
    for record in result:
        record.pop("fit_elapsed_seconds")
    return result


@pytest.mark.parametrize("method", policies.METHODS)
def test_unit_gain_exactly_matches_complete_legacy_episode(model, method):
    cfg = config(count=3)
    settings = policies.PolicyConfig(random_amplitude_nm=20, max_mle_iterations=10)
    legacy = Episode(model, cfg, TRUTH, 42)
    controller = policies.ClassicalPolicy(method, model, cfg, settings, 17)
    history = []
    for _ in range(cfg.max_exposures):
        history.append(legacy.expose(controller.command(tuple(history))))
    expected = legacy.submit(controller.final_command(tuple(history)))
    result = nominal.run_classical_episode(model, cfg, TRUTH, 42, command_gain=1,
                                           method=method, policy_config=settings, policy_seed=17)
    for old, new in zip(history, result["observations"]):
        np.testing.assert_array_equal(old.image_electrons, new.image_electrons)
        assert old.applied_opd_nm == new.commanded_opd_nm
        assert (old.index, old.expected_photons, old.cumulative_expected_photons) == (
            new.index, new.expected_photons, new.cumulative_expected_photons)
    assert result["evaluation"]["physical"] == expected
    assert without_timings(result["fit_records"]) == without_timings(controller.fit_records)


def test_numerical_fallback_is_retained_by_runner(model, monkeypatch):
    def failed(*args, **kwargs):
        raise FloatingPointError("synthetic solver failure")
    monkeypatch.setattr(policies, "estimate_phase_mle", failed)
    result = nominal.run_classical_episode(model, config(), TRUTH, 42, command_gain=1.1,
                                           method="fixed_diversity_mle_v1")
    assert result["fit_records"][0]["fit_status"] == "numerical_failure"
    assert result["fit_records"][0]["fallback"] == "best_measured_camera_score"
    assert result["evaluation"]["physical"]["exposures"] == 3


def test_incomplete_photon_allowance_refuses_before_camera_or_policy(model, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Construction before complete budget validation")
    monkeypatch.setattr(nominal, "GainEpisode", forbidden)
    monkeypatch.setattr(nominal, "ClassicalPolicy", forbidden)
    cfg = replace(config(), photon_budget=299)
    with pytest.raises(ValueError, match="budget"):
        nominal.run_classical_episode(model, cfg, TRUTH, 42, command_gain=1.1,
                                       method="fixed_diversity_mle_v1")


def test_invalid_policy_config_refuses_before_camera(model, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Camera constructed with invalid policy configuration")
    monkeypatch.setattr(nominal, "GainEpisode", forbidden)
    with pytest.raises(ValueError, match="PolicyConfig"):
        nominal.run_classical_episode(model, config(), TRUTH, 42, command_gain=1.1,
                                       method="fixed_diversity_mle_v1", policy_config={})
