import numpy as np
import pytest
import json
from types import SimpleNamespace

import probebench.estimation as estimation
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


def test_numerical_start_failure_does_not_skip_remaining_starts(monkeypatch):
    model, observed = measurements([30,12,-10,8,-6,15], photons=5000, pixels=12)
    attempts=[]
    def minimize(objective,start,**kwargs):
        attempts.append(start.copy())
        if len(attempts)==1:
            raise FloatingPointError("synthetic numerical failure")
        return SimpleNamespace(x=start,fun=float(len(attempts)),success=True,status=0,
                               message="converged",nit=1,nfev=2)
    monkeypatch.setattr(estimation,"minimize",minimize)
    estimate,diagnostics=estimate_phase_mle(model,observed,1)
    assert len(attempts)==3 and diagnostics["chosen_start"]==1
    assert diagnostics["starts"][0]["failure_kind"]=="floating_point_exception"
    assert diagnostics["starts"][0]["function_evaluations"] is None
    assert diagnostics["reported_function_evaluations"] is None
    np.testing.assert_array_equal(estimate,[50,0,0,0,0,0])
    json.dumps(diagnostics,allow_nan=False)


def test_all_nonfinite_starts_are_carried_by_failure(monkeypatch):
    model, observed = measurements([30,12,-10,8,-6,15], photons=5000, pixels=12)
    def minimize(*args,**kwargs):
        return SimpleNamespace(x=np.full(6,np.nan),fun=np.inf,success=False,status=2,
                               message="nonfinite",nit=2,nfev=3)
    monkeypatch.setattr(estimation,"minimize",minimize)
    with pytest.raises(estimation.PhaseEstimationError) as caught:
        estimate_phase_mle(model,observed,1)
    diagnostics=caught.value.diagnostics
    assert len(diagnostics["starts"])==3
    assert all(not start["finite"] for start in diagnostics["starts"])
    assert all(start["solution_opd_nm"]==[None]*6 for start in diagnostics["starts"])
    assert diagnostics["reported_function_evaluations"]==9
    assert diagnostics["chosen_start"] is None
    json.dumps(diagnostics,allow_nan=False)


def test_repeated_images_require_explicit_ambiguous_control_opt_in():
    model, observed = measurements([30,12,-10,8,-6,15], photons=5000, pixels=12)
    repeated = [observed[0], observed[0]]
    with pytest.raises(ValueError, match="Known diverse commands"):
        estimate_phase_mle(model, repeated, 1)
    estimate, diagnostics = estimate_phase_mle(model, repeated, 1, require_diversity=False)
    assert np.isfinite(estimate).all()
    assert diagnostics["has_command_diversity"] is False
    assert len(diagnostics["starts"]) == 3
    with pytest.raises(ValueError, match="boolean"):
        estimate_phase_mle(model, repeated, 1, require_diversity="false")


def test_unprobed_likelihood_and_gradient_preserve_full_conjugate_symmetry():
    model = OpticalModel(OpticsConfig(pupil_pixels=16))
    truth = np.array([30.,12,-10,8,-6,15])
    config = EpisodeConfig(max_exposures=3, photon_budget=6000, photons_per_exposure=2000)
    episode = Episode(model, config, truth, detector_seed=87)
    observations = [episode.expose(np.zeros(6)) for _ in range(3)]
    point = np.array([.24,-.11,.07,.09,-.05,.13])
    conjugacy = np.array([-1,-1,-1,1,1,-1])
    objective, gradient = phase_objective(model, observations, 1, point)
    other, other_gradient = phase_objective(model, observations, 1, point*conjugacy)
    assert objective == pytest.approx(other, rel=0, abs=1e-12)
    np.testing.assert_allclose(other_gradient, gradient*conjugacy, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(model.intensity(truth), model.intensity(truth*conjugacy), atol=1e-15)
    probe = np.array([80,0,0,0,0,0])
    assert np.linalg.norm(model.intensity(truth-probe)-model.intensity(truth*conjugacy-probe)) > .01
