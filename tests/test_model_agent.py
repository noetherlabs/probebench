import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from probebench import agent_session
from probebench.agent_protocol import PROTOCOL, encode_message
import probebench.model_agent as adapter
from probebench.qualification import canonical_hash
from probebench.run import digest


@pytest.fixture
def configs():
    root = Path(__file__).parents[1]
    model = json.loads((root/"configs/local_vision_model.json").read_text())
    model.update(model="fixture-vlm:1",http_timeout_seconds=5,operating_reserve_bytes=16*1024**2,
                 max_request_bytes=256*1024,max_response_bytes=128*1024,max_public_message_bytes=512*1024)
    session = json.loads((root/"configs/agent_protocol_development.json").read_text())
    session["optics"]["pupil_pixels"] = 12
    session.update(operating_reserve_bytes=16*1024**2,max_requests=5)
    return model,session


class FixtureModel:
    """Deterministic protocol fixture, not model-performance evidence."""
    def __init__(self,model,mode="normal"):
        self.model,self.mode = model,mode
        self.calls = []
        self.identity = {"version":{"version":"fixture-1"},"tag":{"digest":"1"*64},
                         "show":{"template":"fixture template"}}
    def inspect(self,**kwargs):
        return {"identity":deepcopy(self.identity),"loaded_models":[]}
    def request(self,path,payload):
        assert path == "/api/chat"
        self.calls.append(deepcopy(payload))
        visible = json.loads(payload["messages"][-1]["content"])
        action = {"protocol":PROTOCOL,"request_id":visible["request_id"],"action":"submit","command_opd_nm":[0]*6}
        if visible["observation"]["index"] == 0 and "expose" in visible["available_actions"]:
            action.update(action="expose",command_opd_nm=[80,0,0,0,0,0])
        elif visible["track"] == "tool_assisted":
            if visible["tool_result"] is not None:
                action["command_opd_nm"] = visible["tool_result"]["estimated_opd_nm"]
            elif "estimate" in visible["available_actions"]:
                action["action"] = "estimate"
                del action["command_opd_nm"]
        content = json.dumps(action,indent=2)
        if self.mode == "malformed" and len(self.calls) == 1:
            content = "This is not an action.\nIt must be rejected."
        if self.mode == "timeout":
            return {"failure":"http_deadline","status":None,"body_base64":"","elapsed_seconds":5.0}
        response = {"model":self.model,"done":True,"done_reason":"stop",
                    "message":{"role":"assistant","content":content,"thinking":"fixture-only reasoning"},
                    "prompt_eval_count":100,"eval_count":30,"total_duration":1000}
        return {"failure":None,"status":200,"body_base64":base64.b64encode(encode_message(response).encode()).decode(),
                "elapsed_seconds":0.01}


@pytest.mark.parametrize("track,expected_calls",[("direct_action",2),("tool_assisted",3)])
def test_real_driver_both_tracks_and_completed_resume_without_model_calls(configs,tmp_path,track,expected_calls):
    model,session = configs
    session.update(track=track,max_tool_calls=0 if track == "direct_action" else 3)
    api = FixtureModel(model["model"])
    root = tmp_path/"run"
    manifest = adapter.run(root,model,session,api=api)
    assert manifest["status"] == "submitted" and len(api.calls) == expected_calls
    assert adapter.verify(root) == manifest
    saved = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    class ForbiddenAPI:
        def inspect(self,**kwargs):
            raise AssertionError("Completed results must not contact a model service")
    assert adapter.run(root,model,session,resume=True,api=ForbiddenAPI()) == manifest
    assert all((root/name).read_bytes() == content for name,content in saved.items())
    assert json.loads((root/"optical/evaluator.json").read_text())["exposures"] == 2
    assert len(api.calls[-1]["messages"]) == 2*expected_calls
    assert all("thinking" not in message for call in api.calls for message in call["messages"])


def test_projection_sends_exact_png_and_no_private_or_raw_numeric_fields(configs):
    _,config = configs
    message = agent_session.create_session(config).request()
    normal = adapter.visual_message(message)
    message["hidden_opd_nm"] = "private marker"
    message["instrument"]["detector_seed"] = "private marker"
    message["instrument"]["optics"]["private_state"] = "private marker"
    message["resources"]["residual_wfe_nm"] = "private marker"
    message["latest_observation"]["image_electrons"] = "private marker"
    assert adapter.visual_message(message) == normal
    assert normal["images"] == [message["latest_observation"]["image_png_base64"]]
    assert not any(term in normal["content"] for term in ("private marker","hidden_opd","detector_seed","image_electrons","residual_wfe"))


def test_malformed_model_output_is_charged_without_repair(configs,tmp_path):
    model,session = configs
    api = FixtureModel(model["model"],"malformed")
    root = tmp_path/"run"
    adapter.run(root,model,session,api=api)
    first = agent_session.read_signed(root/"decisions/decision_000000.json")
    assert json.loads(base64.b64decode(first["action_base64"])) == "This is not an action.\nIt must be rejected."
    second = agent_session.read_signed(root/"decisions/decision_000001.json")["request"]["public_request"]
    assert second["error"] == "invalid_schema" and second["resources"]["exposures_used"] == 1
    assert second["resources"]["requests_used"] == 1
    assert adapter.verify(root)["status"] == "submitted"


def test_model_timeout_retains_explicit_failed_optical_outcome(configs,tmp_path):
    model,session = configs
    api = FixtureModel(model["model"],"timeout")
    root = tmp_path/"run"
    manifest = adapter.run(root,model,session,api=api)
    assert manifest["status"] == "failed" and len(api.calls) == 1
    result = json.loads((root/"result.json").read_text())
    assert result["model_failure"] == "http_deadline"
    assert result["failure_reason"] == "input_eof"
    assert result["final_public_response"]["resources"]["exposures_used"] == 1


def test_saved_response_recovers_without_a_second_generation(configs,tmp_path,monkeypatch):
    model,session = configs
    api = FixtureModel(model["model"])
    root = tmp_path/"run"
    publish = adapter.publish_json
    def interrupt(path,value):
        if path.name == "decision_000000.json":
            raise RuntimeError("interrupted after retained model response")
        publish(path,value)
    monkeypatch.setattr(adapter,"publish_json",interrupt)
    with pytest.raises(RuntimeError,match="interrupted"):
        adapter.run(root,model,session,api=api)
    assert len(api.calls) == 1 and (root/"response.pending.json").exists()
    monkeypatch.setattr(adapter,"publish_json",publish)
    result = adapter.run(root,model,session,resume=True,api=api)
    assert result["status"] == "submitted" and len(api.calls) == 3


def test_uncertain_pending_generation_becomes_failure_without_retry(configs,tmp_path):
    model,session = configs
    class InterruptedAPI(FixtureModel):
        def request(self,path,payload):
            raise RuntimeError("interrupted during generation")
    root = tmp_path/"run"
    with pytest.raises(RuntimeError,match="interrupted"):
        adapter.run(root,model,session,api=InterruptedAPI(model["model"]))
    api = FixtureModel(model["model"])
    result = adapter.run(root,model,session,resume=True,api=api)
    assert result["status"] == "failed" and not api.calls
    assert json.loads((root/"result.json").read_text())["model_failure"] == "interrupted_generation_unknown_outcome"


def test_partial_response_publication_is_preserved_without_repeat_generation(configs,tmp_path,monkeypatch):
    model,session = configs
    api = FixtureModel(model["model"])
    root = tmp_path/"run"
    publish = adapter.publish_json
    partial = b'{"request_sha256":"incomplete receipt'
    def interrupt(path,value):
        if path.name == "response.pending.json":
            path.with_suffix(path.suffix+".pending").write_bytes(partial)
            raise OSError("interrupted receipt write")
        publish(path,value)
    monkeypatch.setattr(adapter,"publish_json",interrupt)
    with pytest.raises(OSError,match="receipt write"):
        adapter.run(root,model,session,api=api)
    assert len(api.calls) == 1
    monkeypatch.setattr(adapter,"publish_json",publish)
    manifest = adapter.run(root,model,session,resume=True,api=api)
    assert manifest["status"] == "failed" and len(api.calls) == 1
    assert (root/"decisions/receipt_000000.incomplete").read_bytes() == partial
    assert adapter.verify(root) == manifest
    assert not any(path.name.endswith(".pending") for path in root.rglob("*"))


def test_saved_action_replays_after_interrupted_pending_cleanup(configs,tmp_path,monkeypatch):
    model,session = configs
    api = FixtureModel(model["model"])
    unlink = Path.unlink
    def interrupt(path,*args,**kwargs):
        if path.name == "generation.pending.json":
            raise RuntimeError("interrupted before sending action")
        return unlink(path,*args,**kwargs)
    root = tmp_path/"run"
    monkeypatch.setattr(Path,"unlink",interrupt)
    with pytest.raises(RuntimeError,match="interrupted"):
        adapter.run(root,model,session,api=api)
    assert len(api.calls) == 1
    monkeypatch.setattr(Path,"unlink",unlink)
    adapter.run(root,model,session,resume=True,api=api)
    assert len(api.calls) == 3


def test_rehashed_model_action_forgery_is_rejected(configs,tmp_path):
    model,session = configs
    root = tmp_path/"run"
    adapter.run(root,model,session,api=FixtureModel(model["model"]))
    path = root/"decisions/decision_000000.json"
    record = agent_session.read_signed(path)
    record["action_base64"] = base64.b64encode(b'{}\n').decode()
    adapter.publish_json(path,agent_session.signed(record))
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["files"][str(path.relative_to(root))] = digest(path)
    adapter.publish_json(root/"manifest.json",manifest)
    with pytest.raises(ValueError,match="response/action binding"):
        adapter.verify(root)


def test_changed_model_identity_fails_before_camera_action(configs,tmp_path):
    model,session = configs
    class ChangingAPI(FixtureModel):
        def request(self,path,payload):
            result = super().request(path,payload)
            self.identity["tag"]["digest"] = "2"*64
            return result
    api = ChangingAPI(model["model"])
    root = tmp_path/"run"
    assert adapter.run(root,model,session,api=api)["status"] == "failed"
    result = json.loads((root/"result.json").read_text())
    assert result["model_failure"] == "model_identity_changed_after_generation"
    assert result["final_public_response"]["resources"]["exposures_used"] == 1


def test_admission_change_is_not_counted_as_a_generation(configs,tmp_path):
    model,session = configs
    class BusyAPI(FixtureModel):
        inspections = 0
        def inspect(self,**kwargs):
            self.inspections += 1
            if self.inspections > 1:
                raise ValueError("model_service_not_idle_or_dedicated")
            return super().inspect(**kwargs)
    api = BusyAPI(model["model"])
    result = adapter.run(tmp_path/"run",model,session,api=api)
    assert result["status"] == "failed" and result["decision_count"] == 1
    assert result["generation_attempts"] == 0 and api.calls == []


def test_prepared_but_not_dispatched_generation_can_resume(configs,tmp_path,monkeypatch):
    model,session = configs
    api = FixtureModel(model["model"])
    root = tmp_path/"run"
    publish = adapter.publish_json
    def interrupt(path,value):
        if path.name == "generation.pending.json" and value["generation_started"]:
            raise RuntimeError("interrupted before dispatch intent")
        publish(path,value)
    monkeypatch.setattr(adapter,"publish_json",interrupt)
    with pytest.raises(RuntimeError,match="before dispatch"):
        adapter.run(root,model,session,api=api)
    assert not api.calls
    monkeypatch.setattr(adapter,"publish_json",publish)
    result = adapter.run(root,model,session,resume=True,api=api)
    assert result["status"] == "submitted" and result["generation_attempts"] == 3


@pytest.mark.parametrize("change,reason",[
    ("length","generation_not_stopped_normally"),("wrong_model","incomplete_or_wrong_model_response"),
    ("large","model_action_too_large"),("tool_call","unsupported_model_message"),
    ("null_message","unsupported_model_message"),("list_message","unsupported_model_message"),
    ("string_message","unsupported_model_message"),("number_message","unsupported_model_message")])
def test_unusable_responses_are_retained_without_camera_actions(configs,tmp_path,change,reason):
    model,session = configs
    class UnusableAPI(FixtureModel):
        def request(self,path,payload):
            transport = super().request(path,payload)
            value = json.loads(base64.b64decode(transport["body_base64"]))
            if change == "length":
                value["done_reason"] = "length"
            elif change == "wrong_model":
                value["model"] = "another-model"
            elif change == "large":
                value["message"]["content"] = "x"*17000
            elif change.endswith("_message"):
                value["message"] = {"null_message":None,"list_message":[],
                                    "string_message":"invalid","number_message":1}[change]
            else:
                value["message"]["tool_calls"] = [{"function":{"name":"unavailable"}}]
            transport["body_base64"] = base64.b64encode(encode_message(value).encode()).decode()
            return transport
    root = tmp_path/"run"
    assert adapter.run(root,model,session,api=UnusableAPI(model["model"]))["status"] == "failed"
    result = json.loads((root/"result.json").read_text())
    assert result["model_failure"] == reason
    assert result["final_public_response"]["resources"]["exposures_used"] == 1
    assert result["generation_attempts"] == 1
    assert adapter.verify(root)["status"] == "failed"


def test_optical_observation_rebinding_is_rejected_even_with_updated_hash_chain(configs,tmp_path):
    model,session = configs
    session.update(track="direct_action",max_tool_calls=0)
    root = tmp_path/"run"
    adapter.run(root,model,session,api=FixtureModel(model["model"]))
    header = agent_session.read_signed(root/"model.json")
    reviewer = adapter.ModelAgent.__new__(adapter.ModelAgent)
    reviewer.config = model
    reviewer.messages = [{"role":"system","content":adapter.PROMPT}]
    previous = canonical_hash({"identity":header["identity"],"config":model,"prompt":adapter.PROMPT})
    for path in sorted((root/"decisions").glob("*.json")):
        record = agent_session.read_signed(path)
        record["request"]["public_request"]["latest_observation"]["expected_photons"] += 1
        record["request"]["payload"] = reviewer.payload(record["request"]["public_request"])
        record["request"]["previous_sha256"] = previous
        previous = canonical_hash(record)
        reviewer.messages.extend([adapter.visual_message(record["request"]["public_request"]),record["assistant_message"]])
        adapter.publish_json(path,agent_session.signed(record))
    manifest = json.loads((root/"manifest.json").read_text())
    for name in manifest["files"]:
        manifest["files"][name] = digest(root/name)
    adapter.publish_json(root/"manifest.json",manifest)
    with pytest.raises(ValueError,match="different optical observation"):
        adapter.verify(root)
