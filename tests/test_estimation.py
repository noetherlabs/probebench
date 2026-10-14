import numpy as np
import pytest

from probebench.environment import Episode, EpisodeConfig
from probebench.estimation import phase_objective, estimate_phase_mle
from probebench.optics import OpticalModel, OpticsConfig


def measurements(truth, photons=200_000, pixels=16):
    model = OpticalModel(OpticsConfig(pupil_pixels=pixels))
    config = EpisodeConfig(max_exposures=3, photon_budget=3*photons,
                           photons_per_exposure=photons)
    episode = Episode(model, config, truth, detector_seed=41)
    observed = [episode.expose(command) for command in ([0]*6, [80,0,0,0,0,0], [-80,0,0,0,0,0])]
    return model, observed


@pytest.mark.parametrize("pixels,oversampling", [(12,1), (13,2)])
def test_optical_jacobian_matches_six_directional_differences(pixels, oversampling):
    model = OpticalModel(OpticsConfig(pupil_pixels=pixels, oversampling=oversampling))
    wave = np.array([30., 12., -10., 8., -6., 15.])
    image, jacobian = model.intensity_and_jacobian(wave)
    np.testing.assert_allclose(image, model.intensity(wave), atol=2e-16)
    np.testing.assert_allclose(jacobian.sum(axis=(1,2)), 0, atol=5e-18)
    for index in range(6):
        delta = np.eye(6)[index] * 1e-4
        numerical = (model.intensity(wave+delta)-model.intensity(wave-delta))/(2e-4)
        np.testing.assert_allclose(jacobian[index], numerical, rtol=2e-6, atol=2e-12)


def test_whole_phase_objective_gradient_matches_independent_differences():
    model, observed = measurements([30,12,-10,8,-6,15], photons=5000, pixels=12)
    point = np.array([.22,-.13,.08,.04,.05,-.10])
    _, gradient = phase_objective(model, observed, 1, point)
    for index in range(6):
        step = np.eye(6)[index]*1e-5
        expected = (phase_objective(model, observed, 1, point+step)[0]
                    - phase_objective(model, observed, 1, point-step)[0])/(2e-5)
        assert gradient[index] == pytest.approx(expected, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("truth", [[-45,0,0,0,0,0], [30,12,-10,8,-6,15]])
def test_bright_noisy_phase_diversity_recovers_opposite_sign_and_six_modes(truth):
    model, observed = measurements(truth)
    estimated, diagnostics = estimate_phase_mle(model, observed, 1)
    assert model.evaluate(np.array(truth)-estimated)["residual_wfe_nm"] < 2.0
    assert diagnostics["optimizer_success"]
    assert len(diagnostics["starts"]) == 3
    assert all("status" in start and "cost" in start for start in diagnostics["starts"])


def test_iteration_limit_is_retained_as_nonconvergence():
    model, observed = measurements([30,12,-10,8,-6,15], photons=5000, pixels=12)
    estimated, diagnostics = estimate_phase_mle(model, observed, 1, max_iterations=1)
    assert np.isfinite(estimated).all()
    assert not diagnostics["optimizer_success"]
    assert all(not start["success"] for start in diagnostics["starts"])
