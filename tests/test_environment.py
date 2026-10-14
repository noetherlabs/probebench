from dataclasses import fields
import numpy as np
import pytest

from probebench.environment import Episode, EpisodeConfig, Observation, display_image
from probebench.optics import OpticalModel, OpticsConfig


def make_episode(seed=42, config=None):
    return Episode(OpticalModel(OpticsConfig()), config or EpisodeConfig(),
                   [45, 0, 0, 0, 0, 0], seed)


def test_invalid_actions_and_budget_refusal_do_not_advance_noise_or_accounting():
    config = EpisodeConfig(max_exposures=1, photon_budget=20, photons_per_exposure=20)
    episode, control = make_episode(config=config), make_episode(config=config)
    for action, photons in (([201, 0, 0, 0, 0, 0], 20), ([0]*6, 21), ([0]*6, -1)):
        with pytest.raises(ValueError): episode.expose(action, photons)
    observed = episode.expose([0]*6)
    np.testing.assert_array_equal(observed.image_electrons, control.expose([0]*6).image_electrons)
    with pytest.raises(ValueError): episode.expose([0]*6)
    result = episode.submit([0]*6)
    assert result["exposures"] == 1 and result["expected_photons_used"] == 20
    with pytest.raises(RuntimeError): episode.expose([0]*6)
    with pytest.raises(RuntimeError): episode.submit([0]*6)


def test_controller_observation_contains_no_hidden_state_or_evaluator_metric():
    observed = make_episode().expose([0]*6)
    assert {f.name for f in fields(observed)} == {
        "index", "applied_opd_nm", "expected_photons", "cumulative_expected_photons", "image_electrons"}
    assert set(observed.metadata()) == {"index", "applied_opd_nm", "expected_photons", "cumulative_expected_photons"}
    assert not observed.image_electrons.flags.writeable


def test_read_noise_retained_display_fixed_and_final_sign_correct():
    episode = make_episode()
    first = episode.expose([0]*6)
    assert (first.image_electrons < 0).any()
    copied = Observation(0, tuple([0]*6), first.expected_photons, first.expected_photons,
                         first.image_electrons * 0.5)
    assert display_image(copied).max() < display_image(first).max()
    episode.expose([45, 0, 0, 0, 0, 0])
    result = episode.submit([45, 0, 0, 0, 0, 0])
    assert result["final"]["residual_wfe_nm"] == 0
    assert result["final"]["strehl_peak_ratio"] == pytest.approx(1)
    assert result["final_mirror_surface_coefficients_nm"][0] == 22.5
    assert result["final_command_has_camera_observation"]


@pytest.mark.parametrize("kw", [{"max_exposures": 0}, {"photon_budget": 2.3},
    {"read_noise_electrons": -1}, {"command_limit_nm": float("nan")}])
def test_reject_invalid_episode_config(kw):
    with pytest.raises(ValueError): EpisodeConfig(**kw)
