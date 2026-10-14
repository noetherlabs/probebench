import numpy as np
import pytest

from probebench.environment import Observation, EpisodeConfig
from probebench.optics import OpticalModel, OpticsConfig
from probebench.policies import estimate_phase, Policy


@pytest.mark.parametrize("truth", [[45, 0, 0, 0, 0, 0], [-45, 0, 0, 0, 0, 0],
                                   [30, 12, -10, 8, -6, 15]])
def test_known_phase_diversity_recovers_noiseless_six_mode_truth(truth):
    model = OpticalModel(OpticsConfig(pupil_pixels=20))
    photons = 1_000_000
    commands = [np.zeros(6), np.array([80., 0, 0, 0, 0, 0])]
    observations = [Observation(i, tuple(command), photons, (i+1)*photons,
                      photons*model.intensity(np.array(truth)-command))
                    for i, command in enumerate(commands)]
    estimate, diagnostics = estimate_phase(model, observations)
    np.testing.assert_allclose(estimate, truth, atol=0.002, rtol=0)
    assert diagnostics["optimizer_success"]


def test_estimator_rejects_missing_diversity():
    model = OpticalModel(OpticsConfig())
    observation = Observation(0, tuple([0]*6), 1000, 1000, 1000*model.intensity([0]*6))
    with pytest.raises(ValueError): estimate_phase(model, [observation])
    with pytest.raises(ValueError): estimate_phase(model, [observation, observation])


@pytest.mark.parametrize("method", ["no_action", "random_probe", "coordinate_search", "phase_diversity"])
def test_policy_respects_smaller_public_command_limit(method):
    model = OpticalModel(OpticsConfig())
    config = EpisodeConfig(command_limit_nm=20)
    policy = Policy(method, model, config, 42)
    first = Observation(0, tuple([0]*6), 1000, 1000, 1000*model.intensity([0]*6))
    assert np.max(np.abs(policy.command([first]))) <= 20
