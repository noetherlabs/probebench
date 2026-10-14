import base64
from copy import deepcopy
import io
import json

import numpy as np
from PIL import Image
import pytest

from probebench.agent_protocol import AgentSession, PROTOCOL, MAX_ACTION_BYTES, encode_message
from probebench.environment import Episode, EpisodeConfig
from probebench.optics import OpticalModel, OpticsConfig


def make_session(track="direct_action", **kwargs):
    model = OpticalModel(OpticsConfig(pupil_pixels=12))
    episode = Episode(model,EpisodeConfig(max_exposures=3,photon_budget=15000,photons_per_exposure=5000),
                      [30,12,-10,8,-6,15],detector_seed=91)
    return AgentSession(episode,track=track,max_tool_calls=3 if track == "tool_assisted" else 0,
                        nonce="a"*32,**kwargs)


def action(session,operation,command=None):
    result = {"protocol":PROTOCOL,"request_id":session.request_id,"action":operation}
    if command is not None:
        result["command_opd_nm"] = command
    return encode_message(result)


def keys(value):
    if isinstance(value,dict):
        return set(value).union(*(keys(v) for v in value.values()))
    if isinstance(value,list):
        return set().union(*(keys(v) for v in value))
    return set()


def test_public_schema_contains_only_calibration_measurements_and_declared_resources():
    session = make_session()
    message = session.request()
    assert set(message) == {"protocol","kind","request_id","track","instrument","resources",
        "available_actions","latest_observation","tool_result","error","status","failure_reason",
        "submitted_command_opd_nm","submitted_command_observation_indices","estimator"}
    assert message["estimator"] is None
    assert not keys(message).intersection({"hidden_opd_nm","detector_seed","trajectory",
        "residual_wfe_nm","strehl_peak_ratio","realized_photoelectrons","truth_index","policy_seed"})
    measured = message["latest_observation"]
    raw = np.array(measured["image_electrons"])
    assert (raw < 0).any()
    np.testing.assert_array_equal(raw,session._episode.observations[0].image_electrons)
    assert Image.open(io.BytesIO(base64.b64decode(measured["image_png_base64"]))).size == (24,24)
    assert message["resources"]["exposures_used"] == 1
    assert message["resources"]["expected_photons_used"] == 5000
    assert message["instrument"]["command_units"] == "opd_nm"
    assert message["instrument"]["mirror_surface_nm_per_opd_nm"] == .5
    # Private-state changes with the same public observations cannot alter the payload.
    session._episode._hidden[:] = [91,82,73,64,55,46]
    session._episode._rng = np.random.default_rng(1234)
    session._episode._records[0]["residual_wfe_nm"] = 987654.0
    assert session.request() == message
    message["latest_observation"]["image_electrons"][0][0] = 987654.0
    assert session.request()["latest_observation"]["image_electrons"][0][0] != 987654.0


@pytest.mark.parametrize("corruption", ["duplicate","nan","infinity","overflow","bool","string",
    "short","bounds","extra","stale","array","utf8","oversized","deep"])
def test_invalid_actions_preserve_camera_state_and_future_noise(corruption):
    session,control = make_session(),make_session()
    command = json.loads(action(session,"expose",[80,0,0,0,0,0]))
    if corruption in {"nan","infinity","overflow"}:
        raw = encode_message(command).replace("[80,", "["+{"nan":"NaN","infinity":"Infinity","overflow":"1e999"}[corruption]+",")
    elif corruption == "duplicate":
        raw = '{"action":"submit",'+encode_message(command)[1:]
    elif corruption == "array": raw = "[]"
    elif corruption == "utf8": raw = b"\xff"
    elif corruption == "oversized": raw = b" "*(MAX_ACTION_BYTES+1)
    elif corruption == "deep": raw = "["*1100+"0"+"]"*1100
    else:
        if corruption == "bool": command["command_opd_nm"][0] = True
        if corruption == "string": command["command_opd_nm"][0] = "80"
        if corruption == "short": command["command_opd_nm"].pop()
        if corruption == "bounds": command["command_opd_nm"][0] = 201
        if corruption == "extra": command["hidden_opd_nm"] = [0]*6
        if corruption == "stale": command["request_id"] = "wrong"
        raw = encode_message(command)
    state = deepcopy(session._episode._rng.bit_generator.state)
    token = session.request_id
    response = session.handle(raw)
    assert response["error"] is not None and response["request_id"] == token
    assert response["resources"]["requests_used"] == 1
    assert response["resources"]["exposures_used"] == 1
    assert response["resources"]["expected_photons_used"] == 5000
    assert session._episode._rng.bit_generator.state == state
    first = session.handle(action(session,"expose",[80,0,0,0,0,0]))
    second = control.handle(action(control,"expose",[80,0,0,0,0,0]))
    assert first["latest_observation"] == second["latest_observation"]


def test_stale_action_and_exposure_limit_do_not_charge_again():
    session = make_session()
    raw = action(session,"expose",[80,0,0,0,0,0])
    session.handle(raw)
    assert session.handle(raw)["error"] == "stale_request"
    session.handle(action(session,"expose",[-80,0,0,0,0,0]))
    before = deepcopy(session._episode._rng.bit_generator.state)
    assert session.handle(action(session,"expose",[0]*6))["error"] == "action_unavailable"
    final = session.handle(action(session,"submit",[12,0,0,0,0,0]))
    assert final["status"] == "submitted" and final["kind"] == "final"
    assert final["submitted_command_observation_indices"] == []
    assert final["resources"]["expected_photons_used"] == 15000
    assert session._episode._rng.bit_generator.state == before
    assert session.evaluator_record()["exposures"] == 3
    with pytest.raises(RuntimeError,match="closed"):
        session.handle(raw)


def test_estimator_requires_assisted_track_diversity_and_separate_call_allowance():
    direct = make_session()
    assert "estimate" not in direct.request()["available_actions"]
    assert direct.handle(action(direct,"estimate"))["error"] == "action_unavailable"
    session = make_session("tool_assisted")
    assert "estimate" not in session.request()["available_actions"]
    assert session.request()["estimator"]["initial_defocus_opd_nm"] == [0,50,-50]
    assert session.handle(action(session,"estimate"))["error"] == "diverse_measurements_required"
    assert session.tool_calls_used == 0
    session.handle(action(session,"expose",[80,0,0,0,0,0]))
    assert "estimate" in session.request()["available_actions"]
    first = session.handle(action(session,"estimate"))
    assert not first["tool_result"]["cached"]
    assert first["tool_result"]["diagnostics"]["observations_used"] == 2
    second = session.handle(action(session,"estimate"))
    assert second["tool_result"]["cached"]
    assert len(session.retained_fits()) == 1
    session.handle(action(session,"estimate"))
    assert session.handle(action(session,"estimate"))["error"] == "action_unavailable"
    assert session.request()["resources"]["exposures_used"] == 2
    final = session.handle(action(session,"submit",first["tool_result"]["estimated_opd_nm"]))
    assert not keys(final).intersection({"hidden_opd_nm","detector_seed","residual_wfe_nm","strehl_peak_ratio"})
    record = session.evaluator_record()
    assert record["tool_calls_used"] == 3 and record["exposures"] == 2
    assert record["final"]["residual_wfe_nm"] < record["initial"]["residual_wfe_nm"]/2


def test_invalid_request_limit_retains_a_failed_outcome_at_last_measured_command():
    session = make_session(max_requests=1)
    final = session.handle("not json")
    assert final["status"] == "failed" and final["failure_reason"] == "request_limit"
    assert final["submitted_command_opd_nm"] == [0]*6
    assert final["resources"]["exposures_used"] == 1
    assert final["resources"]["requests_remaining"] == 0
    assert session.evaluator_record()["status"] == "failed"


def test_photon_allowance_can_expire_before_exposure_allowance():
    model = OpticalModel(OpticsConfig(pupil_pixels=12))
    episode = Episode(model,EpisodeConfig(max_exposures=3,photon_budget=5000,photons_per_exposure=5000),
                      [30,12,-10,8,-6,15],detector_seed=91)
    session = AgentSession(episode,track="direct_action")
    before = deepcopy(episode._rng.bit_generator.state)
    assert "expose" not in session.request()["available_actions"]
    response = session.handle(action(session,"expose",[80,0,0,0,0,0]))
    assert response["error"] == "action_unavailable"
    assert response["resources"]["exposures_remaining"] == 2
    assert response["resources"]["expected_photons_remaining"] == 0
    assert episode._rng.bit_generator.state == before


def test_failed_estimator_call_remains_visible_and_consumes_its_tool_allowance(monkeypatch):
    import probebench.budget_policies as policies
    from probebench.estimation import PhaseEstimationError
    session = make_session("tool_assisted")
    session.handle(action(session,"expose",[80,0,0,0,0,0]))
    def failed(*args,**kwargs):
        raise PhaseEstimationError("synthetic failure",{"starts":[{"success":False,"finite":False}]*3})
    monkeypatch.setattr(policies,"estimate_phase_mle",failed)
    response = session.handle(action(session,"estimate"))
    result = response["tool_result"]
    assert result["diagnostics"]["fit_status"] == "numerical_failure"
    assert len(result["diagnostics"]["starts"]) == 3
    assert response["resources"]["tool_calls_used"] == 1
    assert result["estimated_opd_nm"] in [list(o.applied_opd_nm) for o in session._episode.observations]
    session.handle(action(session,"submit",result["estimated_opd_nm"]))
    assert session.evaluator_record()["fit_records"][0]["fit_status"] == "numerical_failure"
