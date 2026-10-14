"""Bounded HTTP transport to an explicitly local, installed vision model."""
from __future__ import annotations

import base64
import http.client
import json
import math
import re
import subprocess
import sys
from time import perf_counter
from urllib.parse import urlsplit

ENDPOINTS = {"/api/version", "/api/tags", "/api/show", "/api/ps", "/api/chat"}


def local_address(endpoint):
    value = urlsplit(endpoint)
    if (value.scheme != "http" or value.hostname not in {"127.0.0.1", "::1"}
        or value.username is not None or value.password is not None
        or value.path not in {"", "/"} or value.query or value.fragment):
        raise ValueError("A literal loopback HTTP endpoint is required")
    port = 11434 if value.port is None else value.port
    if not 1 <= port <= 65535:
        raise ValueError("Invalid local endpoint port")
    return value.hostname, port


def validate_config(config):
    expected = {"backend", "model", "endpoint", "think", "options", "http_timeout_seconds",
                "max_response_bytes", "max_request_bytes", "max_public_message_bytes", "operating_reserve_bytes"}
    if set(config) != expected or config["backend"] != "ollama_local_vision_v1":
        raise ValueError("Unexpected local model configuration")
    local_address(config["endpoint"])
    if (not isinstance(config["model"], str) or not config["model"]
        or "cloud" in config["model"].lower() or len(config["model"]) > 200):
        raise ValueError("An installed local model name is required")
    if type(config["think"]) is not bool:
        raise ValueError("Explicit boolean thinking setting required")
    options = config["options"]
    if set(options) != {"temperature", "seed", "num_ctx", "num_predict", "top_p", "top_k"}:
        raise ValueError("Declare all supported sampler options")
    for field, lower, upper in (("seed",0,2**31-1),("num_ctx",512,131072),
                                ("num_predict",1,8192),("top_k",1,1000)):
        if type(options[field]) is not int or not lower <= options[field] <= upper:
            raise ValueError(f"Invalid sampler {field}")
    for field, upper in (("temperature",2),("top_p",1)):
        if type(options[field]) not in (int,float) or not math.isfinite(options[field]) or not 0 <= options[field] <= upper:
            raise ValueError(f"Invalid sampler {field}")
    if options["num_predict"] >= options["num_ctx"] or options["top_p"] == 0:
        raise ValueError("Invalid context or nucleus allowance")
    for field in ("max_response_bytes","max_request_bytes","max_public_message_bytes"):
        if type(config[field]) is not int or not 1024 <= config[field] <= 32*1024**2:
            raise ValueError(f"Invalid {field}")
    if (type(config["http_timeout_seconds"]) not in (int,float)
        or not math.isfinite(config["http_timeout_seconds"])
        or not 0 < config["http_timeout_seconds"] <= 1800):
        raise ValueError("Invalid HTTP deadline")
    if type(config["operating_reserve_bytes"]) is not int or config["operating_reserve_bytes"] < 0:
        raise ValueError("Invalid operating reserve")


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def strict_json(raw):
    value = json.loads(raw, object_pairs_hook=_unique)
    json.dumps(value, allow_nan=False)
    return value


def http_worker(request):
    """One request, no redirects, environment proxies or retry machinery."""
    host, port = local_address(request["endpoint"])
    path = request["path"]
    if path not in ENDPOINTS:
        raise ValueError("Unsupported model endpoint")
    body = None if request["payload"] is None else json.dumps(request["payload"],allow_nan=False).encode()
    connection = http.client.HTTPConnection(host,port,timeout=request["timeout"])
    try:
        connection.request("GET" if body is None else "POST",path,body,{"Content-Type":"application/json"})
        response = connection.getresponse()
        raw = response.read(request["limit"]+1)
        return {"status":response.status,"body_base64":base64.b64encode(raw).decode(),
                "failure":"response_too_large" if len(raw) > request["limit"] else None}
    except (OSError,http.client.HTTPException):
        return {"status":None,"body_base64":"","failure":"http_transport_error"}
    finally:
        connection.close()


class LocalOllama:
    """Killable client requests; stopping the client does not prove server cancellation."""

    def __init__(self, config):
        validate_config(config)
        self.config = config

    def request(self,path,payload=None):
        config = self.config
        timeout = config["http_timeout_seconds"] if path == "/api/chat" else 5
        request = {"endpoint":config["endpoint"],"path":path,"payload":payload,
                   "timeout":timeout,"limit":config["max_response_bytes"]}
        raw = json.dumps(request,allow_nan=False).encode()
        if len(raw) > config["max_request_bytes"]:
            return {"status":None,"body_base64":"","failure":"request_too_large","elapsed_seconds":0.0}
        started = perf_counter()
        child = subprocess.Popen([sys.executable,"-m","probebench.local_ollama","--http-worker"],
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            output, _ = child.communicate(raw,timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            result = {"status":None,"body_base64":"","failure":"http_deadline"}
        else:
            if child.returncode != 0:
                result = {"status":None,"body_base64":"","failure":"http_worker_error"}
            else:
                result = strict_json(output)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
        return {**result,"elapsed_seconds":perf_counter()-started}

    def decoded(self,path,payload=None):
        record = self.request(path,payload)
        return decode_response(record)

    def inspect(self, *, initially_idle=False):
        version = self.decoded("/api/version")
        tags = self.decoded("/api/tags")
        models = tags.get("models")
        if not isinstance(models,list) or not all(isinstance(m,dict) for m in models):
            raise ValueError("invalid_model_metadata")
        matches = [m for m in models if self.config["model"] in (m.get("name"),m.get("model"))]
        if len(matches) != 1:
            raise ValueError("installed_model_missing_or_ambiguous")
        tag = matches[0]
        show = self.decoded("/api/show",{"model":self.config["model"]})
        if (not isinstance(show.get("details"),dict) or not isinstance(show.get("model_info"),dict)
            or not isinstance(show.get("capabilities"),list)):
            raise ValueError("invalid_model_metadata")
        if (any(value.get(key) for value in (show,tag) for key in ("remote_model","remote_host"))
            or show.get("details",{}).get("format") != "gguf"
            or not show.get("model_info",{}).get("general.architecture")
            or type(tag.get("size")) is not int or tag["size"] < 1024**2):
            raise ValueError("local_weight_metadata_required")
        if "vision" not in show.get("capabilities",[]) or "completion" not in show.get("capabilities",[]):
            raise ValueError("vision_model_required")
        if self.config["think"] and "thinking" not in show.get("capabilities",[]):
            raise ValueError("thinking_not_supported")
        if not isinstance(tag.get("digest"),str) or re.fullmatch(r"[0-9a-f]{64}",tag["digest"]) is None:
            raise ValueError("model_digest_required")
        if not isinstance(version.get("version"),str) or not version["version"]:
            raise ValueError("runtime_version_required")
        running = self.decoded("/api/ps")
        loaded = running.get("models")
        if (not isinstance(loaded,list) or not all(isinstance(m,dict) for m in loaded)
            or (initially_idle and loaded) or any(m.get("digest") != tag["digest"] for m in loaded)):
            raise ValueError("model_service_not_idle_or_dedicated")
        return {"identity":{"version":version,"tag":tag,"show":show},"loaded_models":loaded}


def decode_response(record):
    if record["failure"] is not None:
        raise ValueError(record["failure"])
    if record["status"] != 200:
        raise ValueError("http_status_error")
    try:
        value = strict_json(base64.b64decode(record["body_base64"],validate=True))
    except (ValueError,UnicodeError,RecursionError):
        raise ValueError("invalid_http_json") from None
    if not isinstance(value,dict) or "error" in value:
        raise ValueError("invalid_http_object")
    return value


if __name__ == "__main__":
    if sys.argv[1:] != ["--http-worker"]:
        raise SystemExit("This module is an internal HTTP worker")
    print(json.dumps(http_worker(strict_json(sys.stdin.buffer.read())),allow_nan=False))
