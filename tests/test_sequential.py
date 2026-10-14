import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import probebench.sequential as sequential
import probebench.study as study_module
from probebench.qualification import canonical_hash
from probebench.run import digest


@pytest.fixture
def config_path(tmp_path):
    config = json.loads((Path(__file__).parents[1]/"configs/sequential_calibration_development.json").read_text())
    config.update({"replicates":1, "exposure_budgets":[3], "methods":["phase_diversity_mle_v1"],
                   "truths_opd_nm":config["truths_opd_nm"][:1], "operating_reserve_bytes":16*1024**2})
    # Tiny unit fixture; the published experiment retains its independent 1-GiB reserve.
    config["optics"]["pupil_pixels"] = 12
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    return path


def test_interrupted_episode_reuses_images_and_completed_fit_records(config_path,tmp_path,monkeypatch):
    root = tmp_path/"run"
    original = sequential.ClassicalPolicy.final_command
    def interrupt(*args,**kwargs):
        raise RuntimeError("interrupted before final submission")
    monkeypatch.setattr(sequential.ClassicalPolicy,"final_command",interrupt)
    with pytest.raises(RuntimeError,match="interrupted before final"):
        sequential.run(root,config_path)
    saved = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("frame_*.npz")}
    assert len(saved) == 3
    monkeypatch.setattr(sequential.ClassicalPolicy,"final_command",original)
    def forbidden(*args,**kwargs):
        raise AssertionError("A saved exposure must not require another policy decision")
    monkeypatch.setattr(sequential.ClassicalPolicy,"command",forbidden)
    manifest = sequential.run(root,config_path,resume=True)
    assert manifest["episode_count"] == 1 and manifest["exposure_count"] == 3
    assert all((root/p).read_bytes()==b for p,b in saved.items())
    record = json.loads((root/"cases/episode_0000.json").read_text())
    assert [r["observations_used"] for r in record["result"]["fit_records"]] == [2,3]
    assert record["result"]["expected_photons_used"] == 60_000
    monkeypatch.setattr(sequential,"run_episode",forbidden)
    assert sequential.run(root,config_path,resume=True)==manifest


def test_complete_case_resume_and_partial_final_publication(config_path,tmp_path,monkeypatch):
    config = json.loads(config_path.read_text())
    config["methods"] = ["no_action_v2","coordinate_search_v2"]
    config_path.write_text(json.dumps(config))
    root=tmp_path/"run"
    assert sequential.run(root,config_path,limit_episodes=1)=={"complete":False,"completed_episodes":1}
    first=(root/"cases/episode_0000.json").read_bytes()
    manifest=sequential.run(root,config_path,resume=True)
    assert manifest["episode_count"] == 2
    assert (root/"cases/episode_0000.json").read_bytes()==first
    (root/"manifest.json").unlink()
    (root/"manifest.json.pending").write_text('{"interrupted":')
    def forbidden(*args,**kwargs):
        raise AssertionError("Completed episodes must not be executed again")
    monkeypatch.setattr(sequential,"run_episode",forbidden)
    assert sequential.run(root,config_path,resume=True)["episode_count"]==2


def test_physical_verification_rejects_false_metrics_even_with_rehashed_files(config_path,tmp_path):
    root=tmp_path/"run"
    sequential.run(root,config_path)
    path=root/"cases/episode_0000.json"
    record=json.loads(path.read_text())
    record["result"]["final"]["residual_wfe_nm"] += 10
    record.pop("record_sha256")
    record["record_sha256"]=canonical_hash(record)
    path.write_text(json.dumps(record))
    manifest=json.loads((root/"manifest.json").read_text())
    manifest["files"][str(path.relative_to(root))]=digest(path)
    (root/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Evaluator metrics disagree"):
        sequential.verify_run(root)


def test_configuration_source_and_concurrent_writers_refused(config_path,tmp_path,monkeypatch):
    root=tmp_path/"run"
    sequential.run(root,config_path)
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): sequential.run(root,config_path,resume=True)
    changed=json.loads(config_path.read_text())
    changed["policy"]["diversity_probe_nm"] += 1
    config_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="Source, configuration or runtime changed"):
        sequential.run(root,config_path,resume=True)


def test_insufficient_disk_rejected_before_any_acquisition(config_path,tmp_path,monkeypatch):
    root=tmp_path/"run"
    monkeypatch.setattr(study_module.shutil,"disk_usage",lambda path:SimpleNamespace(free=0))
    with pytest.raises(OSError,match="Insufficient disk"):
        sequential.run(root,config_path)
    assert not root.exists()


def test_changed_saved_exposure_refuses_resume(config_path,tmp_path,monkeypatch):
    root=tmp_path/"run"
    original=sequential.ClassicalPolicy.final_command
    def interrupt(*args,**kwargs): raise RuntimeError("interrupt")
    monkeypatch.setattr(sequential.ClassicalPolicy,"final_command",interrupt)
    with pytest.raises(RuntimeError): sequential.run(root,config_path)
    monkeypatch.setattr(sequential.ClassicalPolicy,"final_command",original)
    path=root/"frames/episode_0000/frame_0000.npz"
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    arrays["image_electrons"][0,0] += 1
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match="Changed exposure evidence"):
        sequential.run(root,config_path,resume=True)


def test_resume_rejects_symlink_before_opening_lock(config_path,tmp_path):
    root=tmp_path/"run"
    root.mkdir()
    unrelated=tmp_path/"unrelated.txt"
    unrelated.write_text("preserve")
    (root/".lock").symlink_to(unrelated)
    with pytest.raises(ValueError,match="symbolic links"):
        sequential.run(root,config_path,resume=True)
    assert unrelated.read_text()=="preserve"
    assert sorted(p.name for p in root.iterdir())==[".lock"]


def test_verifier_rejects_source_inventory_escape(config_path,tmp_path):
    root=tmp_path/"run"
    sequential.run(root,config_path)
    outside=tmp_path/"outside.txt"
    outside.write_text("outside artifact root")
    header_path=root/"study.json"
    header=json.loads(header_path.read_text())
    header["source_sha256"]={"../../outside.txt":digest(outside)}
    header_path.write_text(json.dumps(header))
    manifest_path=root/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["source_sha256"]=header["source_sha256"]
    manifest["files"]["study.json"]=digest(header_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Unsafe source snapshot path"):
        sequential.verify_run(root)
