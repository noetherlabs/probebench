import json
import shutil

import numpy as np
import pytest

from probebench.run import run_development, verify_run


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    destination = tmp_path_factory.mktemp("development") / "run"
    summaries = run_development(destination)
    return destination, summaries


def test_complete_evidence_and_policy_boundary(completed_run):
    root, summaries = completed_run
    manifest = verify_run(root)
    assert manifest["episodes"] == 16
    assert len(summaries) == 16
    for row in summaries:
        assert row["exposures"] == 3
        assert row["expected_photons_used"] == 600_000
        public = json.loads((root / "observations" / f'{row["episode_id"]}.json').read_text())
        assert set(public) == {"observations", "public_instrument"}
        with np.load(root / "observations" / f'{row["episode_id"]}.npz', allow_pickle=False) as arrays:
            assert set(arrays.files) == {"image_electrons", "applied_opd_nm", "expected_photons"}
            assert arrays["image_electrons"].shape == (3, 64, 64)
            assert np.isfinite(arrays["image_electrons"]).all()
        private = json.loads((root / "evaluator" / f'{row["episode_id"]}.json').read_text())
        assert private["final"]["residual_wfe_nm"] == row["residual_wfe_nm"]
        if row["method"] == "no_action":
            assert row["residual_wfe_nm"] == row["initial_wfe_nm"]
    assert (root / "ambiguity.png").stat().st_size > 1000


def test_immutable_run_and_tamper_detection(completed_run, tmp_path):
    root, _ = completed_run
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError): run_development(root)
    assert (root / "manifest.json").read_bytes() == before
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    target = copied / "observations" / "episode_0000.json"
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"): verify_run(copied)


def test_missing_manifest_and_unlisted_artifacts_refused(completed_run, tmp_path):
    root, _ = completed_run
    with pytest.raises(FileNotFoundError): verify_run(tmp_path)
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    (copied / "unexpected.txt").write_text("not retained by the original manifest")
    with pytest.raises(ValueError, match="unlisted"): verify_run(copied)
