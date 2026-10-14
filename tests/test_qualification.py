import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import probebench.qualification as qualification


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    parent = tmp_path_factory.mktemp("qualification")
    config = json.loads((Path(__file__).parents[1] / "configs/noise_qualification_development.json").read_text())
    config.update({"replicates": 1, "photons_per_exposure": [20_000],
                   "truths_opd_nm": config["truths_opd_nm"][:2]})
    config["optics"]["pupil_pixels"] = 12
    config_path = parent / "config.json"
    config_path.write_text(json.dumps(config))
    root = parent / "run"
    partial = qualification.run_qualification(root, config_path, limit_cases=1)
    assert partial == {"complete": False, "completed_cases": 1}
    first = (root / "cases/case_0000.json").read_bytes()
    assert not (root / "manifest.json").exists()
    finished = qualification.run_qualification(root, config_path, resume=True)
    assert finished["case_count"] == 2 and finished["fit_count"] == 4
    assert (root / "cases/case_0000.json").read_bytes() == first
    return root, config_path


def test_completed_resume_and_report_finalization_do_not_repeat_fits(completed, tmp_path, monkeypatch):
    root, config = completed
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed case must never be fitted again")
    monkeypatch.setattr(qualification, "estimate_phase", forbidden)
    monkeypatch.setattr(qualification, "estimate_phase_mle", forbidden)
    assert qualification.run_qualification(root, config, resume=True)["fit_count"] == 4
    copy = tmp_path / "interrupted_finalization"
    shutil.copytree(root, copy)
    (copy / "manifest.json").unlink()
    (copy / "manifest.json.pending").write_text('{"interrupted":')
    result = qualification.run_qualification(copy, config, resume=True)
    assert result["fit_count"] == 4
    qualification.verify_qualification(copy)


def test_changed_completed_case_is_rejected(completed, tmp_path):
    root, config = completed
    copy = tmp_path / "changed"
    shutil.copytree(root, copy)
    (copy / "manifest.json").unlink()
    path = copy / "cases/case_0000.json"
    record = json.loads(path.read_text())
    record["fits"]["least_squares"]["residual_wfe_nm"] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Changed completed"):
        qualification.run_qualification(copy, config, resume=True)


def test_configuration_change_and_concurrent_writer_rejected(completed, tmp_path):
    root, config = completed
    copy = tmp_path / "run"
    shutil.copytree(root, copy)
    changed = json.loads(config.read_text())
    changed["read_noise_electrons"] = 2
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration or runtime changed"):
        qualification.run_qualification(copy, changed_path, resume=True)
    with (copy / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            qualification.run_qualification(copy, config, resume=True)


@pytest.mark.parametrize("corrupt", [False, True])
def test_interrupted_fit_reuses_and_verifies_saved_acquisition(completed, tmp_path, monkeypatch, corrupt):
    _, config_path = completed
    config = json.loads(config_path.read_text())
    config["truths_opd_nm"] = config["truths_opd_nm"][:1]
    local_config = tmp_path / "config.json"
    local_config.write_text(json.dumps(config))
    root = tmp_path / "interrupted"
    root.mkdir()
    (root / "study.json.pending").write_text('{"interrupted":')
    original_fit = qualification.estimate_phase

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption after acquisition")

    monkeypatch.setattr(qualification, "estimate_phase", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        qualification.run_qualification(root, local_config, resume=True)
    archive = root / "acquisitions/case_0000.npz"
    acquired = archive.read_bytes()
    assert not (root / "cases/case_0000.json").exists()
    monkeypatch.setattr(qualification, "estimate_phase", original_fit)

    def forbid_reacquisition(*args, **kwargs):
        raise AssertionError("Saved measurements must not be acquired again")

    monkeypatch.setattr(qualification.Episode, "expose", forbid_reacquisition)
    if corrupt:
        with np.load(archive) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["images"][0, 0, 0] += 1
        np.savez_compressed(archive, **arrays)
        with pytest.raises(ValueError, match="Acquisition binding or numeric schema changed"):
            qualification.run_qualification(root, local_config, resume=True)
    else:
        assert qualification.run_qualification(root, local_config, resume=True)["fit_count"] == 2
        assert archive.read_bytes() == acquired
