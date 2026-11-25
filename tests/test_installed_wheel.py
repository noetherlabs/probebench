"""Run with PROBEBENCH_INSTALLED_PYTHON and PROBEBENCH_INSTALLED_WHEEL.

The separately provisioned interpreter must contain an ordinary installation of
the real project wheel and its locked dependencies. No source-path injection,
network access, installer invocation, model service or live model is used here.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_real_installed_wheel_model_session_and_completed_replay(tmp_path):
    python = os.environ.get("PROBEBENCH_INSTALLED_PYTHON")
    wheel = os.environ.get("PROBEBENCH_INSTALLED_WHEEL")
    if not python and not wheel:
        pytest.skip("Requires a separately provisioned ordinary wheel installation")
    assert python and wheel, "Both installed interpreter and reviewed wheel are required"
    config = {"model": json.loads((ROOT / "configs/local_vision_model.json").read_text()),
              "session": json.loads((ROOT / "configs/agent_protocol_development.json").read_text())}
    config["model"]["operating_reserve_bytes"] = 0
    config["session"]["operating_reserve_bytes"] = 0
    config["session"]["max_requests"] = 1
    config_path = tmp_path / "public-development-config.json"
    config_path.write_text(json.dumps(config))
    script = r'''
import base64, hashlib, json, sys, zipfile
from pathlib import Path
from probebench import model_agent, run

wheel, configuration, output = map(Path, sys.argv[1:])
package = Path(run.__file__).resolve().parent
assert package.is_relative_to(Path(sys.prefix).resolve())
assert package.parent.name == "site-packages"
assert not (package.parent.parent / "pyproject.toml").exists()
source_root, inventory = run.source_inventory()
from probebench import provenance
with zipfile.ZipFile(wheel) as archive:
    modules = {name: archive.read(name) for name in archive.namelist()
               if name.startswith("probebench/") and name.endswith(".py")}
    for name in archive.namelist():
        if name.endswith((".dist-info/METADATA", ".dist-info/WHEEL")):
            assert (package.parent / name).read_bytes() == archive.read(name)
assert modules and "probebench/provenance.py" in modules
assert set(modules) == {path.relative_to(package.parent).as_posix()
                        for path in package.rglob("*.py")}
for name, content in modules.items():
    assert (package.parent / name).read_bytes() == content
assert source_root == package.parent
assert provenance.identity_fields(inventory)["provenance"]["scheme"] == provenance.INSTALLED_SCHEME
config = json.loads(configuration.read_text())

class FixtureService:
    def __init__(self):
        self.calls = 0
    def inspect(self, *, initially_idle=False):
        return {"identity": {"test_fixture": True}, "loaded_models": []}
    def request(self, path, payload):
        assert path == "/api/chat"
        self.calls += 1
        visible = json.loads(payload["messages"][-1]["content"])
        action = {"protocol": visible["protocol"], "request_id": visible["request_id"],
                  "action": "submit", "command_opd_nm": [0.0] * 6}
        answer = {"model": payload["model"], "done": True, "done_reason": "stop",
                  "message": {"role": "assistant", "content": json.dumps(action)}}
        return {"status": 200, "failure": None, "elapsed_seconds": 0.0,
                "body_base64": base64.b64encode(json.dumps(answer).encode()).decode()}

class ForbiddenService:
    def __getattr__(self, name):
        raise AssertionError("Completed replay contacted a model service")

for track in ("direct_action", "tool_assisted"):
    session = dict(config["session"], track=track,
                   max_tool_calls=0 if track == "direct_action" else 1)
    service = FixtureService()
    folder = output / track
    manifest = model_agent.run(folder, config["model"], session, api=service)
    assert manifest["status"] == "submitted" and service.calls == 1
    assert manifest["generation_attempts"] == manifest["decision_count"] == 1
    assert manifest["provenance"]["scheme"] == provenance.INSTALLED_SCHEME
    before = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file()}
    assert model_agent.verify(folder) == manifest
    assert model_agent.run(folder, config["model"], session, resume=True,
                           api=ForbiddenService()) == manifest
    after = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in folder.rglob("*") if p.is_file()}
    assert before == after
print(json.dumps({"status": "passed", "tracks": 2, "live_model_calls": 0,
                  "installed_package": str(package)}))
'''
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(tmp_path / "mpl"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([python, "-I", "-B", "-c", script, wheel, str(config_path),
                             str(tmp_path / "sessions")], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["tracks"] == 2
