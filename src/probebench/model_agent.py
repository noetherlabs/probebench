"""Auditable local vision-model decisions over the public optical protocol."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from time import monotonic

from . import agent_session
from .agent_protocol import PROTOCOL, MAX_ACTION_BYTES, encode_message, parse_action
from .local_ollama import LocalOllama, decode_response, strict_json, validate_config
from .qualification import canonical_hash, publish_json
from .study import study_directory, publish_manifest, verify_study

PURPOSE = "local_vision_agent_development"
PROMPT = """Control a six-mode optical calibration experiment using only the supplied
camera images, public instrument specification and any permitted estimator replies.
Minimize the final residual wavefront error while respecting the exposure, photon,
request and tool allowances. You do not receive the unknown phase or evaluator scores.
Commands are optical path difference in nanometres in the stated mode order. The
instrument subtracts your commanded phase from the unknown aberration. Probes can
temporarily worsen the image; a final command is not automatically measured.
Return exactly one JSON object, with protocol, the current request_id and action.
For expose or submit, include command_opd_nm: six finite numbers within the instrument
limit. For estimate, omit command_opd_nm. Choose only an available action. Submit
ends the episode. Do not include prose, markdown or extra fields. Estimation is
available only in the tool-assisted track after distinct measurement commands.
Images use the declared fixed display scale. Numeric pixel arrays are not supplied.
Earlier images and actions remain in this conversation. No external tools are available.
"""


def pick(value, fields):
    return {field:deepcopy(value[field]) for field in fields}


def visual_message(message):
    """Project explicit public fields; never accept driver configuration or filenames."""
    if message.get("protocol") != PROTOCOL or message.get("kind") != "decision":
        raise ValueError("Expected a public decision request")
    instrument = message["instrument"]
    visible = pick(message,("protocol","request_id","track","available_actions","error"))
    visible["instrument"] = pick(instrument,("max_exposures","photon_budget","photons_per_exposure",
        "read_noise_electrons","command_limit_nm","modes","command_units",
        "mirror_surface_nm_per_opd_nm","image_units","image_array_row_zero"))
    visible["instrument"]["optics"] = pick(instrument["optics"],("pupil_pixels","oversampling","wavelength_nm"))
    visible["instrument"]["display"] = pick(instrument["display"],("transform","minimum","maximum","colormap"))
    visible["resources"] = pick(message["resources"],("exposures_used","expected_photons_used",
        "exposures_remaining","expected_photons_remaining","requests_used","requests_remaining",
        "tool_calls_used","tool_calls_remaining"))
    observation = message["latest_observation"]
    visible["observation"] = pick(observation,("index","applied_opd_nm","expected_photons","cumulative_expected_photons"))
    visible["estimator"] = None
    visible["tool_result"] = None
    if message["track"] == "tool_assisted":
        visible["estimator"] = pick(message["estimator"],("method","objective","max_iterations_per_start",
            "observations","minimum_distinct_commands","coefficient_bound_opd_nm","initial_defocus_opd_nm"))
        if message["tool_result"] is not None:
            tool = message["tool_result"]
            visible["tool_result"] = pick(tool,("estimated_opd_nm","cached"))
            visible["tool_result"]["fit_status"] = tool["diagnostics"]["fit_status"]
            visible["tool_result"]["optimizer_success"] = tool["diagnostics"]["optimizer_success"]
            visible["tool_result"]["fallback"] = tool["diagnostics"].get("fallback")
    elif message["track"] != "direct_action":
        raise ValueError("Unknown optical track")
    png = observation["image_png_base64"]
    if not base64.b64decode(png,validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid public PNG")
    return {"role":"user","content":encode_message(visible),"images":[png]}


def action_from_response(transport, model):
    """Retain malformed answers as invalid actions; never repair a model's decision."""
    response = decode_response(transport)
    if response.get("model") != model or response.get("done") is not True:
        raise ValueError("incomplete_or_wrong_model_response")
    if response.get("done_reason") != "stop":
        raise ValueError("generation_not_stopped_normally")
    message = response.get("message",{})
    if (message.get("role") != "assistant" or not isinstance(message.get("content"),str)
        or message.get("tool_calls")):
        raise ValueError("unsupported_model_message")
    content = message["content"]
    if len(content.encode("utf-8")) > MAX_ACTION_BYTES-1:
        raise ValueError("model_action_too_large")
    try:
        value = parse_action(content)
    except ValueError:
        value = content  # A JSON string is unambiguously invalid at the driver boundary.
    raw = (encode_message(value)+"\n").encode()
    if len(raw) > MAX_ACTION_BYTES:
        raise ValueError("model_action_too_large")
    return raw, {"role":"assistant","content":content}


class ModelAgent:
    """Durable decisions with at-most-one automatic attempt for each request."""

    def __init__(self, root, config, identity, api):
        self.root,self.config,self.identity,self.api = Path(root),config,identity,api
        (self.root/"decisions").mkdir(exist_ok=True)
        self.records = []
        self.messages = [{"role":"system","content":PROMPT}]
        self.previous = canonical_hash({"identity":identity,"config":config,"prompt":PROMPT})
        names = sorted((self.root/"decisions").glob("*.json"))
        if [p.name for p in names] != [f"decision_{i:06d}.json" for i in range(len(names))]:
            raise ValueError("Noncontiguous model decisions")
        for path in names:
            record = agent_session.read_signed(path)
            self.check_record(record)
            self.adopt(record)
        pending = self.root/"generation.pending.json"
        response = self.root/"response.pending.json"
        if pending.exists() and self.records:
            value = agent_session.read_signed(pending)
            if value["index"] == len(self.records)-1:
                if value != self.records[-1]["request"]:
                    raise ValueError("Pending generation disagrees with saved response")
                pending.unlink()
        if response.exists() and not pending.exists():
            saved = agent_session.read_signed(response)
            if (not self.records or saved["request_sha256"] != canonical_hash(self.records[-1]["request"])
                or saved["transport"] != self.records[-1]["transport"]):
                raise ValueError("Orphaned model response")
            response.unlink()

    def payload(self,message):
        return {"model":self.config["model"],"messages":self.messages+[visual_message(message)],
                "stream":False,"format":"json","think":self.config["think"],"options":self.config["options"]}

    def check_record(self,record):
        request = record["request"]
        if (request["index"] != len(self.records) or request["previous_sha256"] != self.previous
            or request["payload"] != self.payload(request["public_request"])
            or type(request["generation_started"]) is not bool
            or (not request["generation_started"] and record["transport"] is not None)):
            raise ValueError("Changed model request/history binding")
        if record["failure"] is None:
            raw, assistant = action_from_response(record["transport"],self.config["model"])
            if (record["action_base64"] != base64.b64encode(raw).decode()
                or record["assistant_message"] != assistant
                or record["identity_after"] != self.identity):
                raise ValueError("Changed model response/action binding")
        elif record["action_base64"] is not None or record["assistant_message"] is not None:
            raise ValueError("Failed generation cannot provide an action")

    def adopt(self,record):
        self.records.append(record)
        self.previous = canonical_hash(record)
        if record["assistant_message"] is not None:
            self.messages.extend([visual_message(record["request"]["public_request"]),record["assistant_message"]])

    def decide(self,message):
        if self.records:
            last = self.records[-1]
            if last["request"]["public_request"] == message:
                return None if last["failure"] else base64.b64decode(last["action_base64"],validate=True)
            if last["failure"]:
                return None
            if message["resources"]["requests_used"] != last["request"]["public_request"]["resources"]["requests_used"]+1:
                raise ValueError("Unexpected driver decision sequence")
        pending_path = self.root/"generation.pending.json"
        response_path = self.root/"response.pending.json"
        if pending_path.exists():
            request = agent_session.read_signed(pending_path)
            if (request["public_request"] != message or request["payload"] != self.payload(message)
                or request["index"] != len(self.records) or request["previous_sha256"] != self.previous):
                raise ValueError("Pending generation does not match current request")
        else:
            request = {"index":len(self.records),"previous_sha256":self.previous,
                       "public_request":message,"payload":self.payload(message),"generation_started":False}
            publish_json(pending_path,agent_session.signed(request))
        failure,transport,after = None,None,None
        if response_path.exists():
            saved = agent_session.read_signed(response_path)
            if not request["generation_started"] or saved["request_sha256"] != canonical_hash(request):
                raise ValueError("Pending model response binding changed")
            transport = saved["transport"]
        elif request["generation_started"]:
            # A dispatch intent without a receipt does not establish whether the
            # server completed generation. Never silently repeat this uncertain work.
            failure = "interrupted_generation_unknown_outcome"
        try:
            if transport is None and failure is None:
                if self.api.inspect()["identity"] != self.identity:
                    raise ValueError("model_identity_changed_before_generation")
                request["generation_started"] = True
                publish_json(pending_path,agent_session.signed(request))
                transport = self.api.request("/api/chat",request["payload"])
                # Save the response before any further network call, so a crash never
                # causes a second generation or loses a successfully received answer.
                publish_json(response_path,agent_session.signed({
                    "request_sha256":canonical_hash(request),"transport":transport}))
            if failure is None:
                after = self.api.inspect()["identity"]
                if after != self.identity:
                    raise ValueError("model_identity_changed_after_generation")
                action_from_response(transport,self.config["model"])
        except (ValueError,KeyError,TypeError,UnicodeError,RecursionError) as error:
            failure = str(error)
        raw,assistant = (None,None) if failure else action_from_response(transport,self.config["model"])
        record = {"request":request,"transport":transport,"identity_after":after,"failure":failure,
                  "action_base64":None if raw is None else base64.b64encode(raw).decode(),
                  "assistant_message":assistant}
        self.check_record(record)
        publish_json(self.root/"decisions"/f"decision_{len(self.records):06d}.json",agent_session.signed(record))
        self.adopt(record)
        pending_path.unlink()
        if response_path.exists():
            response_path.unlink()
        return raw


def read_public(fd, limit, timeout):
    deadline = monotonic()+timeout
    data = bytearray()
    while True:
        remaining = deadline-monotonic()
        if remaining <= 0 or not select.select([fd],[],[],remaining)[0]:
            raise TimeoutError("Driver response deadline")
        chunk = os.read(fd,min(8192,limit+1-len(data)))
        if not chunk:
            raise RuntimeError("Driver ended before its final message")
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError("Driver message exceeds declared limit")
        if b"\n" in data:
            if not data.endswith(b"\n") or data.count(b"\n") != 1:
                raise ValueError("Unexpected driver message framing")
            return strict_json(data)


def verify_evidence(root,study):
    if any(path.name.endswith(".pending") for path in root.rglob("*")):
        raise ValueError("Unfinished model checkpoint")
    header = agent_session.read_signed(root/"model.json")
    if header["study_binding"] != canonical_hash(study) or header["prompt"] != PROMPT:
        raise ValueError("Changed model study binding")
    config = study["configuration"]
    validate_config(config["model"])
    agent = ModelAgent(root,config["model"],header["identity"],None)
    driver = agent_session.verify_session(root/"optical")
    optical_study = json.loads((root/"optical/study.json").read_text())
    if optical_study["configuration"] != config["session"] or optical_study["source_sha256"] != study["source_sha256"]:
        raise ValueError("Changed optical study binding")
    session,_,_ = agent_session.replay(root/"optical",optical_study)
    initial = agent_session.read_signed(root/"optical/session.json")["initial_request"]
    turns = [agent_session.read_signed(p) for p in sorted((root/"optical/turns").glob("*.json"))]
    current = initial
    for index,record in enumerate(agent.records):
        if record["request"]["public_request"] != current:
            raise ValueError("Model saw a different optical observation")
        if index >= len(turns):
            raise ValueError("Unfinished model/driver exchange")
        incoming = turns[index]["input"]
        if record["failure"] is None:
            timed_out = (index == len(agent.records)-1 and incoming["kind"] == "input_timeout"
                and base64.b64decode(record["action_base64"],validate=True).startswith(
                    base64.b64decode(incoming["raw_base64"],validate=True)))
            if incoming != {"kind":"action","raw_base64":record["action_base64"]} and not timed_out:
                raise ValueError("Driver did not receive the retained model action")
        elif incoming["kind"] not in {"input_eof","input_timeout"}:
            raise ValueError("Failed model call did not terminate the driver")
        current = turns[index]["response"]
    if len(turns) != len(agent.records) or current != session.request():
        raise ValueError("Model/driver exchange length changed")
    result = json.loads((root/"result.json").read_text())
    expected = {"status":session.status,"failure_reason":session.failure_reason,
        "model_failure":next((r["failure"] for r in agent.records if r["failure"]),None),
        "decision_count":len(agent.records),
        "generation_attempts":sum(r["request"]["generation_started"] for r in agent.records),
        "final_public_response":current}
    if result != expected:
        raise ValueError("Changed model result accounting")
    return {"status":driver["status"],"decision_count":len(agent.records),
            "generation_attempts":expected["generation_attempts"]}


def verify(root):
    root = Path(root)
    manifest,study = verify_study(root,PURPOSE)
    counts = verify_evidence(root,study)
    if any(manifest[key] != value for key,value in counts.items()):
        raise ValueError("Changed model manifest accounting")
    return manifest


def run(root, model_config, session_config, *, resume=False, api=None):
    validate_config(model_config)
    agent_session.validate_config(session_config)
    if session_config["decision_timeout_seconds"] < model_config["http_timeout_seconds"]+60:
        raise ValueError("Driver decision timeout must cover HTTP plus metadata checks")
    root = Path(root)
    config = {"model":model_config,"session":session_config,
              "operating_reserve_bytes":max(model_config["operating_reserve_bytes"],session_config["operating_reserve_bytes"])}
    raw_bytes = (session_config["max_requests"]+1)*(2*model_config["max_request_bytes"]
        +4*model_config["max_response_bytes"]+2*model_config["max_public_message_bytes"])
    with study_directory(root,config,raw_bytes,resume) as study:
        if (root/"manifest.json").exists():
            return verify(root)
        api = LocalOllama(model_config) if api is None else api
        header_path = root/"model.json"
        if header_path.exists():
            header = agent_session.read_signed(header_path)
            if header["study_binding"] != canonical_hash(study) or header["prompt"] != PROMPT:
                raise ValueError("Changed model binding")
        else:
            inspection = api.inspect(initially_idle=True)
            header = {"identity":inspection["identity"],"initial_inspection":inspection,
                      "study_binding":canonical_hash(study),"prompt":PROMPT}
            publish_json(header_path,agent_session.signed(header))
        agent = ModelAgent(root,model_config,header["identity"],api)
        session_path = root/"session-config.json"
        if session_path.exists():
            if json.loads(session_path.read_text()) != session_config:
                raise ValueError("Changed driver launch configuration")
        else:
            publish_json(session_path,session_config)
        command = [sys.executable,"-m","probebench.agent_session","--config",str(session_path),"--output",str(root/"optical")]
        if (root/"optical").exists():
            command.append("--resume")
        with (root/"driver.stderr").open("ab") as diagnostic:
            child = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=diagnostic)
            try:
                while True:
                    message = read_public(child.stdout.fileno(),model_config["max_public_message_bytes"],
                                          session_config["decision_timeout_seconds"]+300)
                    if message.get("kind") == "final":
                        break
                    action = agent.decide(message)
                    if action is None:
                        child.stdin.close()
                    else:
                        try:
                            child.stdin.write(action)
                            child.stdin.flush()
                        except BrokenPipeError:
                            pass  # Read the retained final timeout or fail on driver EOF.
                if child.wait(timeout=30) != 0:
                    raise RuntimeError("Optical driver failed")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                child.stdin.close()
                child.stdout.close()
        result = {"status":message["status"],"failure_reason":message["failure_reason"],
            "model_failure":next((r["failure"] for r in agent.records if r["failure"]),None),
            "decision_count":len(agent.records),
            "generation_attempts":sum(r["request"]["generation_started"] for r in agent.records),
            "final_public_response":message}
        if (root/"result.json").exists() and json.loads((root/"result.json").read_text()) != result:
            raise ValueError("Changed model result")
        publish_json(root/"result.json",result)
        counts = verify_evidence(root,study)
        manifest = publish_manifest(root,study,PURPOSE,**counts)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config",type=Path)
    parser.add_argument("--session-config",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--verify",type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    else:
        if not all((args.model_config,args.session_config,args.output)):
            parser.error("Provide model configuration, session configuration and output")
        result = run(args.output,json.loads(args.model_config.read_text()),
                     json.loads(args.session_config.read_text()),resume=args.resume)
    print(encode_message({key:result[key] for key in ("purpose","status","decision_count","generation_attempts")}))
    if result["status"] != "submitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
