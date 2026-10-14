import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest

from probebench.local_ollama import LocalOllama, decode_response, local_address, validate_config


@pytest.fixture
def model_config():
    config = json.loads((Path(__file__).parents[1]/"configs/local_vision_model.json").read_text())
    config.update(model="fixture-vlm:1",http_timeout_seconds=5,operating_reserve_bytes=16*1024**2,
                  max_request_bytes=256*1024,max_response_bytes=128*1024,max_public_message_bytes=512*1024)
    return config


@pytest.fixture
def local_service(model_config):
    state = {"calls":[],"status":200,"chat_delay":0,"chat_body":b'{"ok":true}',
        "tag":{"name":model_config["model"],"model":model_config["model"],"digest":"1"*64,"size":2*1024**2},
        "show":{"details":{"format":"gguf","quantization_level":"fixture"},
                "model_info":{"general.architecture":"fixture"},
                "capabilities":["completion","vision"],"template":"fixture template"},
        "loaded":[]}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_GET(self):
            self.reply(None)
        def do_POST(self):
            self.reply(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        def reply(self,payload):
            state["calls"].append((self.path,payload))
            values = {"/api/version":{"version":"fixture-1"},"/api/tags":{"models":[state["tag"]]},
                "/api/show":state["show"],"/api/ps":{"models":state["loaded"]}}
            if self.path == "/api/chat":
                time.sleep(state["chat_delay"])
                raw = state["chat_callback"](payload) if "chat_callback" in state else state["chat_body"]
                status = state["status"]
            else:
                raw,status = json.dumps(values[self.path]).encode(),200
            try:
                self.send_response(status)
                self.send_header("Content-Length",str(len(raw)))
                if status == 302:
                    self.send_header("Location","https://example.invalid/forbidden")
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker = threading.Thread(target=server.serve_forever,kwargs={"poll_interval":0.01},daemon=True)
    worker.start()
    model_config["endpoint"] = f"http://127.0.0.1:{server.server_port}"
    yield LocalOllama(model_config),state
    server.shutdown()
    server.server_close()
    worker.join()


@pytest.mark.parametrize("endpoint",["https://127.0.0.1:11434","http://example.invalid",
    "http://localhost:11434","http://127.0.0.2:11434","http://127.0.0.1:0",
    "http://name:secret@127.0.0.1:11434","http://127.0.0.1/path","http://127.0.0.1?x=1"])
def test_nonlocal_or_ambiguous_endpoints_are_rejected(endpoint):
    with pytest.raises(ValueError):
        local_address(endpoint)


def test_explicit_sampler_and_local_only_configuration(model_config):
    validate_config(model_config)
    for update in ({"model":"remote:cloud"},{"think":"false"},{"http_timeout_seconds":float("nan")}):
        with pytest.raises(ValueError):
            validate_config({**model_config,**update})
    for field,value in (("temperature",float("inf")),("seed",True),("num_predict",-1),("top_p",0)):
        changed = deepcopy(model_config)
        changed["options"][field] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_local_metadata_and_raw_http_body_are_retained(local_service):
    api,state = local_service
    identity = api.inspect(initially_idle=True)["identity"]
    assert identity["tag"]["digest"] == "1"*64
    assert identity["show"]["template"] == "fixture template"
    reply = api.request("/api/chat",{"messages":[{"role":"user","content":"calibrate"}]})
    assert base64.b64decode(reply["body_base64"]) == state["chat_body"]
    assert decode_response(reply) == {"ok":True}
    assert state["calls"][-1][1]["messages"][0]["content"] == "calibrate"


@pytest.mark.parametrize("change,reason",[
    ("nonvision","vision_model_required"),("remote","local_weight_metadata_required"),
    ("not_installed","installed_model_missing"),("busy","model_service_not_idle"),
    ("missing_digest","model_digest_required")])
def test_admission_rejects_unavailable_or_inappropriate_models(local_service,change,reason):
    api,state = local_service
    if change == "nonvision":
        state["show"]["capabilities"] = ["completion"]
    elif change == "remote":
        state["show"]["remote_host"] = "https://example.invalid"
    elif change == "not_installed":
        state["tag"]["name"] = state["tag"]["model"] = "another-model"
    elif change == "busy":
        state["loaded"] = [{"digest":"1"*64}]
    else:
        state["tag"]["digest"] = "missing"
    with pytest.raises(ValueError,match=reason):
        api.inspect(initially_idle=True)
    assert not any(path == "/api/chat" for path,_ in state["calls"])


def test_redirects_response_limits_and_deadlines_never_retry(local_service):
    api,state = local_service
    state["status"] = 302
    with pytest.raises(ValueError,match="http_status_error"):
        decode_response(api.request("/api/chat",{}))
    state["status"] = 200
    state["chat_body"] = b"a"*(api.config["max_response_bytes"]+10)
    assert api.request("/api/chat",{})["failure"] == "response_too_large"
    state["chat_delay"] = 1.0
    api.config["http_timeout_seconds"] = 0.3
    started = time.monotonic()
    assert api.request("/api/chat",{})["failure"] == "http_deadline"
    assert time.monotonic()-started < 3
    assert len([path for path,_ in state["calls"] if path == "/api/chat"]) == 3


def test_oversized_requests_are_rejected_before_connecting(local_service):
    api,state = local_service
    assert api.request("/api/chat",{"content":"x"*api.config["max_request_bytes"]})["failure"] == "request_too_large"
    assert state["calls"] == []


@pytest.mark.parametrize("track,expected_generations",[("direct_action",2),("tool_assisted",3)])
def test_http_fixture_and_optical_process_end_to_end(local_service,tmp_path,track,expected_generations):
    from probebench import model_agent
    from test_model_agent import FixtureModel
    api,state = local_service
    fixture = FixtureModel(api.config["model"])
    state["chat_callback"] = lambda payload: base64.b64decode(fixture.request("/api/chat",payload)["body_base64"])
    session = json.loads((Path(__file__).parents[1]/"configs/agent_protocol_development.json").read_text())
    session["optics"]["pupil_pixels"] = 12
    session.update(track=track,max_tool_calls=0 if track == "direct_action" else 3,
                   max_requests=4,operating_reserve_bytes=16*1024**2)
    root = tmp_path/"run"
    result = model_agent.run(root,api.config,session)
    assert result["status"] == "submitted" and result["generation_attempts"] == expected_generations
    before = len(state["calls"])
    assert model_agent.verify(root) == result
    assert model_agent.run(root,api.config,session,resume=True) == result
    assert len(state["calls"]) == before
    for payload in fixture.calls:
        assert payload["stream"] is False and payload["format"] == "json"
        assert payload["options"] == api.config["options"]
        assert "hidden_opd_nm" not in json.dumps(payload)
