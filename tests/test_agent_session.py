import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

import probebench.agent_session as transport
import probebench.budget_policies as policies
from probebench.agent_protocol import PROTOCOL, encode_message
from probebench.qualification import canonical_hash
from probebench.run import digest


@pytest.fixture
def config():
    root = Path(__file__).parents[1]
    value = json.loads((root/"configs/agent_protocol_development.json").read_text())
    value["optics"]["pupil_pixels"] = 12
    value["instrument"].update(photons_per_exposure=5000,photon_budget=15000)
    value["operating_reserve_bytes"] = 16*1024**2
    return value


def scripted(output, steps):
    sequence = iter(steps)
    def read():
        message = json.loads(output.getvalue().splitlines()[-1])
        operation, command = next(sequence)
        value = {"protocol":PROTOCOL,"request_id":message["request_id"],"action":operation}
        if command == "estimate":
            command = message["tool_result"]["estimated_opd_nm"]
        if command is not None:
            value["command_opd_nm"] = command
        return {"kind":"action","raw_base64":base64.b64encode(encode_message(value).encode()).decode()}
    return read


def test_completed_session_replays_saved_fits_without_refitting(config,tmp_path,monkeypatch):
    root = tmp_path/"run"
    output = io.StringIO()
    steps = [("expose",[80,0,0,0,0,0]),("estimate",None),("submit","estimate")]
    manifest = transport.run(root,config,output=output,reader=scripted(output,steps))
    assert manifest["status"] == "submitted" and manifest["exposure_count"] == 2
    saved = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    def forbidden(*args,**kwargs):
        raise AssertionError("Published fits or actions must not be repeated")
    monkeypatch.setattr(policies,"estimate_phase_mle",forbidden)
    resumed = io.StringIO()
    assert transport.run(root,config,resume=True,output=resumed,reader=forbidden) == manifest
    assert len(resumed.getvalue().splitlines()) == 1
    assert json.loads(resumed.getvalue())["status"] == "submitted"
    assert all((root/name).read_bytes()==value for name,value in saved.items())


def test_broken_output_after_saved_fit_resumes_with_same_public_reply(config,tmp_path,monkeypatch):
    class InterruptedOutput(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushes = 0
        def flush(self):
            self.flushes += 1
            if self.flushes == 3:
                raise BrokenPipeError("simulated receiver disconnect")
    root = tmp_path/"run"
    output = InterruptedOutput()
    with pytest.raises(BrokenPipeError):
        transport.run(root,config,output=output,reader=scripted(output,[
            ("expose",[80,0,0,0,0,0]),("estimate",None)]))
    saved = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    def forbidden(*args,**kwargs):
        raise AssertionError("A saved fit cannot be recomputed")
    monkeypatch.setattr(policies,"estimate_phase_mle",forbidden)
    resumed = io.StringIO()
    transport.run(root,config,resume=True,output=resumed,reader=scripted(resumed,[("submit","estimate")]))
    assert json.loads(resumed.getvalue().splitlines()[0]) == json.loads(output.getvalue().splitlines()[-1])
    assert all((root/name).read_bytes()==value for name,value in saved.items())
    assert json.loads((root/"evaluator.json").read_text())["exposures"] == 2


def test_interrupted_pending_input_restores_exact_unpublished_camera_response(config,tmp_path,monkeypatch):
    root = tmp_path/"run"
    output = io.StringIO()
    publish = transport.publish_json
    intercepted = {}
    def interrupted(path,value):
        if path.name == "turn_000000.json":
            intercepted.update(deepcopy(value))
            raise RuntimeError("interrupted before response publication")
        publish(path,value)
    monkeypatch.setattr(transport,"publish_json",interrupted)
    with pytest.raises(RuntimeError,match="before response"):
        transport.run(root,config,output=output,reader=scripted(output,[("expose",[80,0,0,0,0,0])]))
    assert (root/"input.pending.json").exists()
    assert len(output.getvalue().splitlines()) == 1
    monkeypatch.setattr(transport,"publish_json",publish)
    resumed = io.StringIO()
    transport.run(root,config,resume=True,output=resumed,reader=scripted(resumed,[("submit",[0]*6)]))
    assert json.loads(resumed.getvalue().splitlines()[0]) == intercepted["response"]
    assert json.loads((root/"evaluator.json").read_text())["expected_photons_used"] == 10000
    assert not (root/"input.pending.json").exists()


def test_resume_recovers_interrupted_pending_cleanup_without_another_exposure(config,tmp_path,monkeypatch):
    root = tmp_path/"run"
    output = io.StringIO()
    unlink = Path.unlink
    def interrupted(path,*args,**kwargs):
        if path.name == "input.pending.json":
            raise RuntimeError("interrupted pending cleanup")
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,"unlink",interrupted)
    with pytest.raises(RuntimeError,match="pending cleanup"):
        transport.run(root,config,output=output,reader=scripted(output,[("expose",[80,0,0,0,0,0])]))
    first = (root/"turns/turn_000000.json").read_bytes()
    monkeypatch.setattr(Path,"unlink",unlink)
    resumed = io.StringIO()
    transport.run(root,config,resume=True,output=resumed,reader=scripted(resumed,[("submit",[0]*6)]))
    assert (root/"turns/turn_000000.json").read_bytes() == first
    assert json.loads((root/"evaluator.json").read_text())["exposures"] == 2


def test_rehashed_false_evaluator_score_is_rejected(config,tmp_path):
    root = tmp_path/"run"
    output = io.StringIO()
    transport.run(root,config,output=output,reader=scripted(output,[("submit",[0]*6)]))
    path = root/"evaluator.json"
    record = json.loads(path.read_text())
    record["final"]["residual_wfe_nm"] += 1
    path.write_text(json.dumps(record))
    manifest = json.loads((root/"manifest.json").read_text())
    manifest["files"]["evaluator.json"] = digest(path)
    (root/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="evaluator evidence changed"):
        transport.verify_session(root)


def test_rehashed_public_image_change_is_rejected_during_resume(config,tmp_path):
    root = tmp_path/"run"
    output = io.StringIO()
    def disconnected():
        raise RuntimeError("receiver unavailable")
    with pytest.raises(RuntimeError):transport.run(root,config,output=output,reader=disconnected)
    path = root/"session.json"
    record = transport.read_signed(path)
    record["initial_request"]["latest_observation"]["image_electrons"][0][0] += 1
    path.write_text(json.dumps(transport.signed(record)))
    with pytest.raises(ValueError,match="Initial public observation changed"):
        transport.run(root,config,resume=True,output=io.StringIO(),reader=disconnected)


def test_resume_binds_display_runtime_before_replaying_camera(config,tmp_path,monkeypatch):
    root = tmp_path/"run"
    output = io.StringIO()
    def disconnected(): raise RuntimeError("receiver unavailable")
    with pytest.raises(RuntimeError):transport.run(root,config,output=output,reader=disconnected)
    monkeypatch.setattr(transport,"display_runtime",lambda:{"matplotlib":"different"})
    def forbidden(*args,**kwargs): raise AssertionError("Camera replay must not begin")
    monkeypatch.setattr(transport,"create_session",forbidden)
    with pytest.raises(ValueError,match="Display runtime changed"):
        transport.run(root,config,resume=True,output=io.StringIO(),reader=disconnected)


@pytest.mark.parametrize("kind,raw", [("input_eof",b""),("input_timeout",b"partial"),
    ("input_too_large",b"x"*(transport.MAX_ACTION_BYTES+1))])
def test_input_termination_retains_failure_without_uncharged_measurements(config,tmp_path,kind,raw):
    output = io.StringIO()
    def reader():
        return {"kind":kind,"raw_base64":base64.b64encode(raw).decode()}
    transport.run(tmp_path/"run",config,output=output,reader=reader)
    final = json.loads(output.getvalue().splitlines()[-1])
    assert final["status"] == "failed" and final["failure_reason"] == kind
    assert final["resources"]["exposures_used"] == 1
    assert final["submitted_command_opd_nm"] == [0]*6


def test_reader_deadline_covers_partial_line_and_preserves_buffered_lines(monkeypatch):
    read_fd,write_fd = os.pipe()
    try:
        os.write(write_fd,b"{}\n[]\npartial")
        reader = transport.ActionReader(read_fd,.02)
        assert base64.b64decode(reader()["raw_base64"]) == b"{}\n"
        assert base64.b64decode(reader()["raw_base64"]) == b"[]\n"
        assert reader()["kind"] == "input_timeout"
    finally:
        os.close(read_fd);os.close(write_fd)
    monkeypatch.setattr(transport,"MAX_ACTION_BYTES",64)
    read_fd,write_fd = os.pipe()
    try:
        os.write(write_fd,b"x"*65)
        assert transport.ActionReader(read_fd,.02)()["kind"] == "input_too_large"
    finally:
        os.close(read_fd);os.close(write_fd)


@pytest.mark.parametrize("track", ["direct_action","tool_assisted"])
def test_real_stdio_process_exposes_no_evaluator_fields(config,tmp_path,track):
    config["track"] = track
    config["max_tool_calls"] = 0 if track == "direct_action" else 3
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    root = Path(__file__).parents[1]
    environment = {**os.environ,"PYTHONPATH":str(root/"src"),"PYTHONDONTWRITEBYTECODE":"1"}
    child = subprocess.Popen([sys.executable,"-m","probebench.agent_session","--config",str(path),
        "--output",str(tmp_path/"run")],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=environment,cwd=root)
    try:
        def receive():
            assert select.select([child.stdout],[],[],20)[0],"Missing protocol response"
            message = json.loads(child.stdout.readline())
            text = json.dumps(message)
            assert all(name not in text for name in ("hidden_opd_nm","detector_seed","residual_wfe_nm","strehl_peak_ratio"))
            return message
        def send(message,operation,command=None):
            value = {"protocol":PROTOCOL,"request_id":message["request_id"],"action":operation}
            if command is not None:value["command_opd_nm"] = command
            child.stdin.write((encode_message(value)+"\n").encode());child.stdin.flush()
            return receive()
        message = receive()
        message = send(message,"expose",[80,0,0,0,0,0])
        command = [0]*6
        if track == "tool_assisted":
            message = send(message,"estimate")
            command = message["tool_result"]["estimated_opd_nm"]
        final = send(message,"submit",command)
        assert final["status"] == "submitted" and final["resources"]["exposures_used"] == 2
        assert child.wait(timeout=20) == 0,child.stderr.read().decode()
        assert child.stdout.read() == b""
        assert transport.verify_session(tmp_path/"run")["status"] == "submitted"
    finally:
        if child.poll() is None:
            child.terminate();child.wait(timeout=10)
