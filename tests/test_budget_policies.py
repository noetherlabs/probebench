import numpy as np
import pytest

import probebench.budget_policies as policies
from probebench.environment import EpisodeConfig, Observation
from probebench.optics import OpticalModel, OpticsConfig


def observed(command, index, score=1.0):
    image = np.zeros((8, 8))
    image[3:6, 3:6] = score * 100 / 9
    return Observation(index, tuple(command), 100, 100*(index+1), image)


def policy(name, limit=200, budget=25, seed=17):
    return policies.ClassicalPolicy(name, OpticalModel(OpticsConfig()),
        EpisodeConfig(max_exposures=budget, photon_budget=budget*100,
                      photons_per_exposure=100, command_limit_nm=limit), policy_seed=seed)


def test_coordinate_pairs_share_anchor_and_visit_all_six_modes():
    controller = policy("coordinate_search_v2")
    target = np.array([40, -40, 40, -40, 40, -40])
    history = []
    for index in range(13):
        command = controller.command(history)
        score = 1 - np.square(command-target).sum()/100_000
        history.append(observed(command, index, score))
    for mode in range(6):
        plus, minus = history[1+2*mode], history[2+2*mode]
        difference = np.array(plus.applied_opd_nm)-minus.applied_opd_nm
        np.testing.assert_array_equal(difference, np.eye(6)[mode]*80)
    np.testing.assert_array_equal(controller.final_command(history), target)
    np.testing.assert_array_equal(controller.command(history), target+np.array([20,0,0,0,0,0]))


def test_partial_pair_can_submit_its_best_measured_command_and_bound_costs_a_frame():
    controller = policy("coordinate_search_v2", limit=10)
    history = [observed([0]*6, 0, 1)]
    plus = controller.command(history)
    np.testing.assert_array_equal(plus, [10,0,0,0,0,0])
    history.append(observed(plus, 1, 2))
    np.testing.assert_array_equal(controller.final_command(history), plus)
    np.testing.assert_array_equal(controller.command(history), [-10,0,0,0,0,0])


def test_random_action_is_independent_of_budget_and_extra_diagnostic_calls():
    first = policy("random_probe_v2", budget=3)
    second = policy("random_probe_v2", budget=25)
    history = [observed([0]*6, 0)]
    for index in range(1, 3):
        command = first.command(history)
        np.testing.assert_array_equal(first.command(history), command)
        first.final_command(history)
        np.testing.assert_array_equal(second.command(history), command)
        assert np.max(np.abs(command)) <= 60
        history.append(observed(command, index))


def test_diversity_uses_only_available_frames_cache_and_declared_probe_order(monkeypatch):
    controller = policy("phase_diversity_mle_v1")
    counts = []
    def fit(model, observations, *args, **kwargs):
        counts.append(len(observations))
        return np.array([30.,12,-10,8,-6,15]), {"optimizer_success": True, "starts": []}
    monkeypatch.setattr(policies, "estimate_phase_mle", fit)
    history = [observed([0]*6, 0)]
    np.testing.assert_array_equal(controller.final_command(history), np.zeros(6))
    np.testing.assert_array_equal(controller.command(history), [80,0,0,0,0,0])
    history.append(observed([80,0,0,0,0,0], 1))
    correction = controller.final_command(history)
    np.testing.assert_array_equal(controller.command(history), correction)
    assert counts == [2]  # Repeated use of the same public history is cached.
    history.append(observed(correction, 2))
    np.testing.assert_array_equal(controller.command(history), correction+[-80,0,0,0,0,0])
    assert counts == [2,3]
    history[-1] = observed(correction, 2, score=2)
    controller.final_command(history)
    assert counts == [2,3,3]  # Changed measured data must invalidate the cache.
    assert [r["observations_used"] for r in controller.fit_records] == counts


def test_numerical_failure_is_retained_and_fallback_uses_camera_only(monkeypatch):
    controller = policy("phase_diversity_mle_v1")
    history = [observed([0]*6, 0, 2), observed([80,0,0,0,0,0], 1, 1)]
    def failed(*args, **kwargs):
        raise FloatingPointError("synthetic solver failure")
    monkeypatch.setattr(policies, "estimate_phase_mle", failed)
    np.testing.assert_array_equal(controller.final_command(history), [0]*6)
    assert controller.fit_records[0]["fit_status"] == "numerical_failure"
    assert controller.fit_records[0]["fallback"] == "best_measured_camera_score"
    assert not controller.fit_records[0]["optimizer_success"]


def test_all_failed_optimizer_starts_remain_in_policy_evidence(monkeypatch):
    from probebench.estimation import PhaseEstimationError
    controller = policy("phase_diversity_mle_v1")
    history = [observed([0]*6,0,2),observed([80,0,0,0,0,0],1,1)]
    diagnostics={"starts":[{"finite":False,"message":str(i)} for i in range(3)]}
    def failed(*args,**kwargs):
        raise PhaseEstimationError("all starts failed",diagnostics)
    monkeypatch.setattr(policies,"estimate_phase_mle",failed)
    np.testing.assert_array_equal(controller.final_command(history),[0]*6)
    assert controller.fit_records[0]["starts"]==diagnostics["starts"]
    assert controller.fit_records[0]["fit_status"]=="numerical_failure"


@pytest.mark.parametrize("name,offsets", [
    ("unprobed_mle_v1", [0]*7), ("fixed_diversity_mle_v1", [0,80,-80,0,80,-80,0])])
def test_schedule_controls_do_not_consult_images_or_fit_during_acquisition(monkeypatch,name,offsets):
    first, second = policy(name, budget=7), policy(name, budget=25)
    calls = []
    def fit(model, history, *args, **kwargs):
        calls.append((len(history), kwargs["require_diversity"]))
        return np.arange(6, dtype=float), {"optimizer_success": True, "starts": []}
    monkeypatch.setattr(policies, "estimate_phase_mle", fit)
    histories = [[],[]]
    for index, offset in enumerate(offsets):
        for controller, history, score in ((first,histories[0],1.), (second,histories[1],9.)):
            command = controller.command(history)
            np.testing.assert_array_equal(command,[offset,0,0,0,0,0])
            history.append(observed(command,index,score))
    assert calls == []
    np.testing.assert_array_equal(first.final_command(histories[0]), np.arange(6))
    assert calls == [(7,name != "unprobed_mle_v1")]
    # A checkpoint before submission has no fits. A saved final fit is reusable.
    resumed = policy(name,budget=7)
    resumed.restore_fit_records(first.fit_records,histories[0])
    np.testing.assert_array_equal(resumed.final_command(histories[0]),np.arange(6))
    assert len(calls) == 1


def test_fixed_and_adaptive_diversity_have_identical_two_frame_estimates():
    from probebench.environment import Episode
    instrument = EpisodeConfig(max_exposures=2, photon_budget=10000, photons_per_exposure=5000)
    model = OpticalModel(OpticsConfig(pupil_pixels=12))
    outputs = []
    for name in ("fixed_diversity_mle_v1", "phase_diversity_mle_v1"):
        controller = policies.ClassicalPolicy(name,model,instrument)
        episode = Episode(model,instrument,[30,12,-10,8,-6,15],detector_seed=19)
        history = []
        for _ in range(2):
            history.append(episode.expose(controller.command(history)))
        outputs.append((history,controller.final_command(history)))
    for first, second in zip(outputs[0][0],outputs[1][0]):
        np.testing.assert_array_equal(first.image_electrons,second.image_electrons)
    np.testing.assert_array_equal(outputs[0][1],outputs[1][1])


@pytest.mark.parametrize("settings", [{"coordinate_step_nm": 0}, {"diversity_probe_nm": np.nan},
    {"coordinate_step_floor_nm": 41}, {"max_mle_iterations": 0}])
def test_invalid_policy_settings_rejected(settings):
    with pytest.raises(ValueError): policies.PolicyConfig(**settings)
