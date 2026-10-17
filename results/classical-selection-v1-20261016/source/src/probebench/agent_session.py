"""Source-bound, resumable JSON-lines sessions for trusted local agents."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import select
import sys
from time import monotonic, perf_counter
import zlib

import matplotlib
import numpy as np
import PIL

from .agent_protocol import AgentSession, MAX_ACTION_BYTES, encode_message, validate_limits
from .environment import Episode, EpisodeConfig
from .optics import OpticalModel, OpticsConfig, coefficients
from .qualification import canonical_hash, publish_json
from .study import publish_manifest, study_directory, verify_study

PURPOSE = "agent_protocol_development"


def display_runtime():
    return {"matplotlib":matplotlib.__version__,"pillow":PIL.__version__,
            "zlib":zlib.ZLIB_RUNTIME_VERSION}


class ActionReader:
    """Read bounded lines under a deadline, including partial-line stalls."""

    def __init__(self, fd, timeout_seconds):
        self.fd, self.timeout_seconds = fd, timeout_seconds
        self.buffer = b""

    def __call__(self):
        deadline = monotonic()+self.timeout_seconds
        while True:
            newline = self.buffer.find(b"\n")
            if 0 <= newline < MAX_ACTION_BYTES:
                raw, self.buffer = self.buffer[:newline+1], self.buffer[newline+1:]
                return {"kind":"action", "raw_base64":base64.b64encode(raw).decode()}
            if len(self.buffer) > MAX_ACTION_BYTES:
                return {"kind":"input_too_large", "raw_base64":base64.b64encode(self.buffer).decode()}
            remaining = deadline-monotonic()
            if remaining <= 0 or not select.select([self.fd],[],[],remaining)[0]:
                return {"kind":"input_timeout", "raw_base64":base64.b64encode(self.buffer).decode()}
            chunk = os.read(self.fd, min(8192,MAX_ACTION_BYTES+1-len(self.buffer)))
            if not chunk:
                raw, self.buffer = self.buffer, b""
                return {"kind":"action" if raw else "input_eof", "raw_base64":base64.b64encode(raw).decode()}
            self.buffer += chunk


def validate_config(config):
    fields = {"purpose","optics","instrument","hidden_opd_nm","detector_seed","track",
              "max_requests","max_tool_calls","max_mle_iterations","decision_timeout_seconds",
              "operating_reserve_bytes"}
    if set(config) != fields or config["purpose"] != "public_development_only":
        raise ValueError("Expected an explicit development-session configuration")
    OpticsConfig(**config["optics"])
    validate_limits(config["track"],config["max_requests"],config["max_tool_calls"],config["max_mle_iterations"])
    instrument = EpisodeConfig(**config["instrument"])
    coefficients(config["hidden_opd_nm"])
    if type(config["detector_seed"]) is not int or config["detector_seed"] < 0:
        raise ValueError("Nonnegative integer detector seed required")
    if (type(config["decision_timeout_seconds"]) not in (int,float)
        or not np.isfinite(config["decision_timeout_seconds"])
        or not 0 < config["decision_timeout_seconds"] <= 3600):
        raise ValueError("Decision deadline must be in (0,3600] seconds")
    if instrument.photons_per_exposure > instrument.photon_budget:
        raise ValueError("Budget must admit the initial charged exposure")


def create_session(config, nonce=None):
    model = OpticalModel(OpticsConfig(**config["optics"]))
    episode = Episode(model,EpisodeConfig(**config["instrument"]),config["hidden_opd_nm"],config["detector_seed"])
    return AgentSession(episode,track=config["track"],max_requests=config["max_requests"],
        max_tool_calls=config["max_tool_calls"],max_mle_iterations=config["max_mle_iterations"],nonce=nonce)


def signed(record):
    return {**record,"record_sha256":canonical_hash(record)}


def read_signed(path):
    record = json.loads(path.read_text())
    original = record.pop("record_sha256")
    if canonical_hash(record) != original:
        raise ValueError("Changed protocol checkpoint")
    return record


def apply_input(session, incoming, fits=None):
    if set(incoming) != {"kind","raw_base64"}:
        raise ValueError("Unexpected retained input schema")
    raw = base64.b64decode(incoming["raw_base64"],validate=True)
    kind = incoming["kind"]
    if len(raw) > MAX_ACTION_BYTES+1:
        raise ValueError("Oversized retained input")
    if kind == "action":
        return session.handle(raw,retained_fits=fits)
    if kind not in {"input_eof","input_timeout","input_too_large"}:
        raise ValueError("Unexpected input termination")
    if (kind == "input_eof" and raw or kind == "input_too_large" and len(raw) <= MAX_ACTION_BYTES):
        raise ValueError("Invalid retained termination")
    return session.abort(kind)


def replay(root, study):
    header = read_signed(root/"session.json")
    if header["study_binding"] != canonical_hash(study):
        raise ValueError("Session/source binding changed")
    if header["display_runtime"] != display_runtime():
        raise ValueError("Display runtime changed")
    session = create_session(study["configuration"],header["nonce"])
    if session.request() != header["initial_request"]:
        raise ValueError("Initial public observation changed")
    names = sorted(p.name for p in (root/"turns").glob("*.json"))
    if names != [f"turn_{i:06d}.json" for i in range(len(names))]:
        raise ValueError("Protocol checkpoints must form a contiguous sequence")
    previous = canonical_hash(header)
    for index,name in enumerate(names):
        record = read_signed(root/"turns"/name)
        if (record["index"] != index or record["previous_sha256"] != previous
            or record["request_sha256"] != canonical_hash(session.request())
            or any(type(record[k]) not in (int,float) or not np.isfinite(record[k]) or record[k] < 0
                   for k in ("driver_processing_seconds","input_wait_seconds"))):
            raise ValueError("Changed protocol sequence or accounting")
        response = apply_input(session,record["input"],record["fit_records"])
        if response != record["response"] or session.retained_fits() != record["fit_records"]:
            raise ValueError("Saved public response or estimator checkpoint changed")
        previous = canonical_hash(record)
    return session,len(names),previous


def finish(root,study,session):
    result = session.evaluator_record()
    path = root/"evaluator.json"
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError("Changed final evaluator record")
    else:
        publish_json(path,result)
    if not (root/"manifest.json").exists():
        publish_manifest(root,study,PURPOSE,episode_count=1,
                         exposure_count=result["exposures"],status=result["status"])


def verify_session(root):
    root = Path(root)
    manifest,study = verify_study(root,PURPOSE)
    validate_config(study["configuration"])
    session,_,_ = replay(root,study)
    result = session.evaluator_record()
    if (json.loads((root/"evaluator.json").read_text()) != result
        or manifest["episode_count"] != 1 or manifest["exposure_count"] != result["exposures"]
        or manifest["status"] != result["status"]):
        raise ValueError("Final evaluator evidence changed")
    return manifest


def run(root,config,*,resume=False,reader=None,output=None):
    """Persist each input and complete response before disclosing it to the agent."""
    validate_config(config)
    root = Path(root)
    output = sys.stdout if output is None else output
    reader = ActionReader(sys.stdin.fileno(),config["decision_timeout_seconds"]) if reader is None else reader
    pixels = config["optics"]["pupil_pixels"]*config["optics"]["oversampling"]
    # JSON doubles and display PNGs are larger than raw arrays. Admission includes
    # repeated last-frame messages and complete estimator diagnostics in each turn.
    raw_bytes = (config["max_requests"]+1)*(pixels**2*32+256*1024)
    with study_directory(root,config,raw_bytes,resume) as study:
        if not (root/"turns").exists():
            (root/"turns").mkdir()
        if not (root/"session.json").exists():
            session = create_session(config)
            publish_json(root/"session.json",signed({"study_binding":canonical_hash(study),
                "nonce":session.nonce,"initial_request":session.request(),"display_runtime":display_runtime()}))
        session,index,previous = replay(root,study)
        pending_path = root/"input.pending.json"
        if (root/"manifest.json").exists():
            verify_session(root)
        elif pending_path.exists():
            pending = read_signed(pending_path)
            if pending["index"] == index-1 and index:
                last = read_signed(root/"turns"/f"turn_{index-1:06d}.json")
                if (pending["input"] != last["input"] or pending["request_sha256"] != last["request_sha256"]
                    or pending["input_wait_seconds"] != last["input_wait_seconds"]):
                    raise ValueError("Pending input disagrees with the published turn")
                pending_path.unlink()
            elif pending["index"] != index or pending["request_sha256"] != canonical_hash(session.request()):
                raise ValueError("Pending input does not match the resumed request")

        def emit():
            output.write(encode_message(session.request())+"\n")
            output.flush()

        # A pending response must be recovered before inviting another decision.
        if not pending_path.exists():
            if session.status != "active":
                finish(root,study,session)
            emit()
        while session.status == "active":
            if pending_path.exists():
                pending = read_signed(pending_path)
            else:
                waiting = perf_counter()
                incoming = reader()
                pending = {"index":index,"request_sha256":canonical_hash(session.request()),"input":incoming,
                           "input_wait_seconds":perf_counter()-waiting}
                publish_json(pending_path,signed(pending))
            if (type(pending["input_wait_seconds"]) not in (int,float)
                or not np.isfinite(pending["input_wait_seconds"]) or pending["input_wait_seconds"] < 0):
                raise ValueError("Invalid pending input duration")
            started = perf_counter()
            response = apply_input(session,pending["input"])
            record = {"index":index,"previous_sha256":previous,"request_sha256":pending["request_sha256"],
                "input":pending["input"],"response":response,"fit_records":session.retained_fits(),
                "driver_processing_seconds":perf_counter()-started,
                "input_wait_seconds":pending["input_wait_seconds"]}
            path = root/"turns"/f"turn_{index:06d}.json"
            if path.exists():
                raise ValueError("Refusing to replace a completed protocol turn")
            publish_json(path,signed(record))
            pending_path.unlink()
            previous,index = canonical_hash(record),index+1
            if session.status != "active":
                finish(root,study,session)
            emit()
        return verify_session(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--verify",type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_session(args.verify)
        print(encode_message({k:v for k,v in result.items() if k not in {"files","source_sha256"}}))
    elif args.config and args.output:
        run(args.output,json.loads(args.config.read_text()),resume=args.resume)
    else:
        parser.error("Provide --verify, or --config and --output")


if __name__ == "__main__":
    main()
