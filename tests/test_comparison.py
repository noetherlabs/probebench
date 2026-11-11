"""Injected native execution checks on already exposed mixed_a only."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from probebench import comparison as native
from probebench import comparison_pilot as pilot
from probebench import comparison_report as report_api
from probebench.comparison_inventory import case_inventory, public_fixture_inventory
from probebench.environment import Observation

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT/"results/classical-selection-v1-20261016/selected-settings.json"
RESERVE = 1024**3


class CountingPolicy:
    instances = []
    fail = False

    def __init__(self, name, model, config, settings, seed):
        assert seed == 70000001
        assert set(vars(model)) == {"config", "pupil", "basis", "amplitude", "ideal_peak"}
        assert settings.max_mle_iterations == 200
        self.name, self.config, self.fit_records, self.seen = name, config, [], []
        self.instances.append(self)

    def command(self, history):
        assert type(history) is tuple
        assert all(type(o) is Observation and not o.image_electrons.flags.writeable for o in history)
        self.seen.append(len(history))
        if self.fail and len(history) == 1:
            raise RuntimeError("synthetic interrupted policy")
        return np.zeros(6, dtype=np.float64)

    def final_command(self, history):
        return self.command(history)


@pytest.fixture
def injected(monkeypatch):
    CountingPolicy.instances, CountingPolicy.fail = [], False
    monkeypatch.setattr(pilot, "ClassicalPolicy", CountingPolicy)
    return CountingPolicy


def run(root, **kwargs):
    return native.run_development(root, SELECTED.read_bytes(), operating_reserve_bytes=RESERVE, **kwargs)


def test_bounded_prefix_resumes_only_sealed_cases_and_preserves_bytes(tmp_path, injected):
    root = tmp_path/"comparison"
    first = run(root, limit_cases=1)
    assert first == {"status":"bounded_prefix", "completed":1, "required":36}
    case = case_inventory(public_fixture_inventory())[0]
    folder = root/native._base_name(case["base_wavefront_id"])/case["id"]
    before = {p:p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    second = run(root, resume=True, limit_cases=1)
    assert second["completed"] == 2 and len(injected.instances) == 2
    assert all(p.read_bytes() == value for p,value in before.items())
    assert all(instance.seen == list(range(instance.config.max_exposures+1)) for instance in injected.instances)


def test_unknown_dispatch_without_manifest_never_reexecutes(tmp_path, monkeypatch, injected):
    root = tmp_path/"comparison"
    run(root, limit_cases=1)
    case = case_inventory(public_fixture_inventory())[0]
    folder = root/native._base_name(case["base_wavefront_id"])/case["id"]
    (folder/"manifest.json").unlink()
    # Even complete-looking outputs cannot erase the missing binding/unknown dispatch.
    monkeypatch.setattr(native, "OpticalModel", lambda *_: pytest.fail("Optics constructed on resume"))
    with pytest.raises(ValueError, match="reconciliation"):
        run(root, resume=True)
    assert len(injected.instances) == 1
    assert (folder/"dispatch.json").is_file() and (folder/"records/terminal.json").is_file()


def test_failure_retains_all_intents_and_raw_prefix_and_stops_resume(tmp_path, injected):
    injected.fail = True
    root = tmp_path/"comparison"
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        run(root)
    case = case_inventory(public_fixture_inventory())[0]
    folder = root/native._base_name(case["base_wavefront_id"])/case["id"]
    assert (root/"stop.json").is_file() and not (root/"manifest.json").exists()
    assert (folder/"dispatch.json").is_file()
    assert (folder/"records/00-image.npy").is_file()
    assert (folder/"records/01-policy-intent.json").is_file()
    assert (folder/"records/failure.json").is_file()
    with pytest.raises(ValueError, match="no retry"):
        run(root, resume=True)
    assert len(injected.instances) == 1


@pytest.fixture(scope="module")
def complete_packet(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison-complete")/"packet"
    CountingPolicy.instances, CountingPolicy.fail = [], False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pilot, "ClassicalPolicy", CountingPolicy)
        assert run(root) == {"status":"complete", "completed":36, "required":36}
        assert len(CountingPolicy.instances) == 36
        assert sum(instance.config.max_exposures for instance in CountingPolicy.instances) == 276
    return root


def test_complete_readonly_native_report_budget_and_final_camera_semantics(complete_packet, tmp_path, monkeypatch):
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in complete_packet.rglob("*") if p.is_file()}
    monkeypatch.setattr(native, "OpticalModel", lambda *_: pytest.fail("Verification constructed optics"))
    monkeypatch.setattr(pilot, "ClassicalPolicy", lambda *_: pytest.fail("Verification invoked policy"))
    checked = native.verify_development(complete_packet)
    assert len(checked["outcomes"]) == 36
    assert checked["artifact_verification"] == "complete_native_consistency_checks"
    assert not checked["physical_recomputation"] and not checked["held_out_provenance"]
    assert sum(row["terminal"]["expected_photons_used"] for row in checked["outcomes"]) == 3036000
    assert all(row["evaluation"]["physical"]["final_command_has_camera_observation"] for row in checked["outcomes"])
    assert run(complete_packet, resume=True)["status"] == "complete"
    output = tmp_path/"report"
    report_api.write_development_report(complete_packet, output)
    summary = json.loads((output/"summary.json").read_text())
    assert summary["statistical_inference"] == "none"
    assert not summary["independent_population_sample"]
    assert all(json.loads(path.read_text())["fresh_final_confirmation_exposure"] is False
               for path in output.glob("case-*.json"))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p,value in before.items())


def rehash_node(folder):
    path = folder/"manifest.json"
    value = json.loads(path.read_text())
    value["files"] = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in value["files"]}
    value["children"] = {name:hashlib.sha256((folder/name/"manifest.json").read_bytes()).hexdigest()
                         for name in value["children"]}
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("mutation", ["physical_command", "member_identity", "final_flag", "camera_budget"])
def test_coherent_manifest_rewrite_still_fails_native_semantics(complete_packet, tmp_path, mutation):
    root = tmp_path/"changed"
    shutil.copytree(complete_packet, root)
    case = case_inventory(public_fixture_inventory())[0]
    base = root/native._base_name(case["base_wavefront_id"])
    folder = base/case["id"]
    if mutation == "member_identity":
        path = folder/"dispatch.json"
        value = json.loads(path.read_text())
        value["case"]["member"] = "conjugate"
    else:
        path = folder/"records/evaluation.json"
        value = json.loads(path.read_text())
        if mutation == "physical_command":
            value["physical"]["final_applied_opd_nm"][0] = 1.0
        elif mutation == "final_flag":
            value["physical"]["final_command_has_camera_observation"] = False
        else:
            value["physical"]["expected_photons_used"] -= 1
    path.write_text(json.dumps(value))
    for node in (folder, base, root):
        rehash_node(node)
    with pytest.raises(ValueError):
        native.verify_development(root)


def test_oversized_node_refuses_before_hash_or_json_read(complete_packet, tmp_path, monkeypatch):
    root = tmp_path/"changed"
    shutil.copytree(complete_packet, root)
    with (root/"study.json").open("wb") as stream:
        stream.truncate(pilot.MAX_FILE_BYTES+1)
    original = pilot._read_selected
    def bounded_read(path):
        if Path(path).name == "study.json":
            pytest.fail("Oversized artifact was read before admission")
        return original(path)
    monkeypatch.setattr(pilot, "_read_selected", bounded_read)
    with pytest.raises(ValueError, match="oversized"):
        native.verify_development(root)


def test_removed_or_unlisted_case_cannot_pass_global_closure(complete_packet, tmp_path):
    root = tmp_path/"changed"
    shutil.copytree(complete_packet, root)
    case = case_inventory(public_fixture_inventory())[0]
    base = root/native._base_name(case["base_wavefront_id"])
    shutil.rmtree(base/case["id"])
    path = base/"manifest.json"
    value = json.loads(path.read_text())
    value["children"].pop(case["id"])
    path.write_text(json.dumps(value))
    rehash_node(root)
    with pytest.raises(ValueError):
        native.verify_development(root)


def test_resume_header_is_admitted_before_legacy_reader_or_optics(tmp_path, monkeypatch, injected):
    root = tmp_path/"comparison"
    run(root, limit_cases=1)
    with (root/"study.json").open("wb") as stream:
        stream.truncate(65)
    monkeypatch.setattr(pilot, "MAX_FILE_BYTES", 64)
    monkeypatch.setattr(native, "study_directory", lambda *a, **k: pytest.fail("Legacy reader reached"))
    monkeypatch.setattr(native, "OpticalModel", lambda *_: pytest.fail("Optics constructed"))
    with pytest.raises(ValueError, match="allowance"):
        run(root, resume=True)
    assert len(injected.instances) == 1


def test_consistent_nonzero_first_frame_refused_without_optical_replay(tmp_path, monkeypatch, injected):
    def nonzero(self, history):
        return np.ones(6, dtype=np.float64)
    monkeypatch.setattr(injected, "command", nonzero)
    root = tmp_path/"legacy-pilot"
    # Inject an entire internally consistent recorded episode: commands,
    # observations, physical trajectories and all native hashes agree.
    pilot.run_pilot(root, SELECTED.read_bytes(), operating_reserve_bytes=RESERVE)
    monkeypatch.setattr(pilot, "OpticalModel", lambda *_: pytest.fail("Verifier propagated optics"))
    monkeypatch.setattr(pilot, "ClassicalPolicy", lambda *_: pytest.fail("Verifier replayed policy"))
    with pytest.raises(ValueError, match="initial exposure"):
        pilot.verify_pilot(root)
