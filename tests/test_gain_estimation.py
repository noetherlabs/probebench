import numpy as np
import pytest
from probebench.optics import OpticalModel, OpticsConfig
from probebench.gain_estimation import (NominalObservation, joint_objective,
                                       estimate_phase_and_gain, compensated_command)
from probebench.estimation import phase_objective


def fixture(read_noise=1.0):
    model = OpticalModel(OpticsConfig(pupil_pixels=12))
    phase = np.array([15., 10., -3., 4., -5., 2.])
    rng = np.random.default_rng(104)
    obs = []
    for d in (0., 80., -80.):
        command = np.array([d, 0, 0, 0, 0, 0])
        mean = 20000*model.intensity(phase-0.9*command)
        image = rng.poisson(mean).astype(float) + rng.normal(0, read_noise, mean.shape)
        obs.append(NominalObservation(command, image, 20000))
    return model, obs


@pytest.mark.parametrize('index', range(7))
def test_joint_gradient_matches_central_difference(index):
    model, obs = fixture()
    x = np.array([.12,.08,-.02,.03,-.04,.01,-.5])
    _, grad = joint_objective(model, obs, 1., x)
    step = np.eye(7)[index]*1e-5
    numerical = (joint_objective(model, obs, 1., x+step)[0]
                 - joint_objective(model, obs, 1., x-step)[0])/2e-5
    assert grad[index] == pytest.approx(numerical, rel=3e-5, abs=1e-7)


def test_at_unit_gain_exactly_matches_existing_nominal_objective():
    model, obs = fixture()
    x = np.array([.12,.08,-.02,.03,-.04,.01,0.])
    new, grad = joint_objective(model, obs, 1., x)
    old, oldgrad = phase_objective(model, obs, 1., x[:6])
    assert new == old
    np.testing.assert_allclose(grad[:6], oldgrad, rtol=2e-14, atol=2e-15)


def test_no_diversity_refused():
    model, obs = fixture()
    with pytest.raises(ValueError, match='Unchanging'):
        estimate_phase_and_gain(model, [obs[0], obs[0]], 1.)


def test_snapshot_does_not_mutate_or_accept_truth():
    model, obs = fixture()
    assert not hasattr(obs[0], 'true_gain')
    assert not obs[0].image_electrons.flags.writeable
    with pytest.raises(ValueError):
        obs[0].nominal_command_nm[0] = 20
    with pytest.raises(TypeError):
        NominalObservation(np.zeros(6), np.ones((12,12)), 1., true_gain=.9)


@pytest.mark.parametrize('gain', [0., -1., np.nan])
def test_invalid_compensation_refused(gain):
    with pytest.raises(ValueError):
        compensated_command(np.ones(6), gain)


def test_gain_compensation_cancels_ideal_plant_and_respects_limit():
    phase = np.array([15., 10., -3., 4., -5., 2.])
    np.testing.assert_allclose(phase-.9*compensated_command(phase,.9), 0., atol=1e-14)
    assert np.max(np.abs(compensated_command(phase,.01,10))) <= 10


def test_power_and_finite_pupil_wfe():
    model, _ = fixture()
    phase = np.array([15., 10., -3., 4., -5., 2.])
    assert model.intensity(phase).sum() == pytest.approx(1., abs=2e-14)
    wf = np.einsum('m,mij->ij', phase, model.basis)[model.pupil]
    assert model.evaluate(phase)['residual_wfe_nm'] == pytest.approx(np.std(wf))


def test_joint_fit_recovers_response_in_bright_development_fixture():
    model, obs = fixture()
    phase, gain, diag = estimate_phase_and_gain(model, obs, 1.)
    assert diag['optimizer_success']
    assert gain == pytest.approx(.9, abs=.025)
    assert model.evaluate(np.array([15.,10.,-3.,4.,-5.,2.])-.9*compensated_command(phase,gain))['residual_wfe_nm'] < 2.
