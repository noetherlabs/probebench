"""Manifest binding and all-start accounting without acquiring or fitting data."""
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def summary_tool(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/summarize_selection_fits.py"
    spec = importlib.util.spec_from_file_location("selection_fit_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A small synthetic grid exercises all 14 production candidates. The public
    # CLI offers no option to override its native manifest pin or study grid.
    monkeypatch.setattr(module, "PHOTONS", (2000,))
    monkeypatch.setattr(module, "BUDGETS", (3,))
    monkeypatch.setattr(module, "BASES", (("low", "rms_selection_v1:low:000000"),))
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def rebind(root, tool, monkeypatch):
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json")
    cases = list((root / "cases").glob("*.json"))
    manifest = {"episode_count": len(cases), "exposure_count": len(cases) * 3,
                "files": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths}}
    write_json(root / "manifest.json", manifest)
    monkeypatch.setattr(tool, "MANIFEST_SHA256", hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())


@pytest.fixture
def archive(tmp_path, summary_tool, monkeypatch):
    root = tmp_path.resolve() / "archive"
    (root / "cases").mkdir(parents=True)
    (root / ".lock").write_bytes(b"")
    for candidate, (member, replicate) in itertools.product(
            summary_tool.CANDIDATES, itertools.product(("primary", "conjugate"), (0, 1))):
        family = summary_tool.CANDIDATES[candidate]
        identity = "case_" + hashlib.sha256(f"{candidate}:{member}:{replicate}".encode()).hexdigest()
        prefixes = ([2, 3] if family == "phase_diversity_mle_v1" else
                    [3] if family in ("unprobed_mle_v1", "fixed_diversity_mle_v1") else [])
        fits = [{"observations_used": prefix, "fit_status": "finite", "cost": 1.0,
                 "optimizer_success": True, "chosen_start": 0,
                 "starts": [{"finite": True, "success": True, "status": 0,
                             "message": "CONVERGENCE", "cost": float(i + 1),
                             "solution_opd_nm": [0.0] * 6} for i in range(3)]}
                for prefix in prefixes]
        write_json(root / "cases" / f"{identity}.json", {
            "case": {"id": identity, "candidate_id": candidate, "family": family, "method": family,
                     "photons_per_exposure": 2000, "exposure_budget": 3, "stratum_id": "low",
                     "base_wavefront_id": "rms_selection_v1:low:000000", "member": member, "replicate": replicate},
            "result": {"exposures": 3, "fit_records": fits}})
    rebind(root, summary_tool, monkeypatch)
    return root


def fit_case(root):
    return next(path for path in sorted((root / "cases").glob("*.json"))
                if json.loads(path.read_text())["result"]["fit_records"])


def test_all_candidates_and_nonwinning_starts_are_counted(archive, summary_tool, tmp_path, monkeypatch):
    path = fit_case(archive)
    record = json.loads(path.read_text())
    starts = record["result"]["fit_records"][0]["starts"]
    starts[1].update(success=False, status=1, message="ITERATION LIMIT")
    starts[2].update(finite=False, success=False, status=None, message="FLOATING POINT ERROR",
                     cost=None, solution_opd_nm=None)
    write_json(path, record)
    rebind(archive, summary_tool, monkeypatch)
    output = tmp_path / "diagnostics.json"
    result = summary_tool.summarize(archive, output)
    assert json.loads(output.read_text()) == result
    assert len(result["candidates"]) == 14
    assert result["total"]["episodes"] == 56
    assert result["total"]["multistart_fit_calls"] == 40
    assert result["total"]["attempted_starts"] == 120
    assert result["total"]["finite_unconverged_starts"] == 1
    assert result["total"]["nonfinite_starts"] == 1
    assert result["total"]["finite_unconverged_fit_calls"] == 0
    assert result["total"]["numerical_failure_fit_calls"] == 0
    assert result["total"]["episodes_with_unsuccessful_starts"] == 1
    assert len(result["unsuccessful_starts"]) == 2
    assert not any(item["chosen"] for item in result["unsuccessful_starts"])
    assert str(tmp_path) not in output.read_text()
    assert "timestamp" not in output.read_text()


@pytest.mark.parametrize("mutation", ["manifest_pin", "case_hash", "missing_case", "extra_case",
                                     "incomplete_starts", "wrong_chosen_start", "duplicate_semantics"])
def test_malformed_or_incomplete_input_creates_no_output(
        mutation, archive, summary_tool, tmp_path, monkeypatch):
    path = fit_case(archive)
    record = json.loads(path.read_text())
    if mutation == "manifest_pin":
        monkeypatch.setattr(summary_tool, "MANIFEST_SHA256", "0" * 64)
    elif mutation == "case_hash":
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "missing_case":
        path.unlink()
        rebind(archive, summary_tool, monkeypatch)
    elif mutation == "extra_case":
        (archive / "cases/extra.json").write_bytes(b"{}")
    else:
        if mutation == "incomplete_starts":
            record["result"]["fit_records"][0]["starts"].pop()
        elif mutation == "wrong_chosen_start":
            record["result"]["fit_records"][0]["chosen_start"] = 2
        elif mutation == "duplicate_semantics":
            record["case"]["replicate"] = 1 - record["case"]["replicate"]
        write_json(path, record)
        rebind(archive, summary_tool, monkeypatch)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        summary_tool.summarize(archive, output)
    assert not output.exists()


@pytest.mark.parametrize("location", ["archive_ancestor", "case", "output_parent"])
def test_symlinks_are_rejected(archive, summary_tool, tmp_path, location):
    output = tmp_path / "must-not-exist.json"
    if location == "archive_ancestor":
        alias = tmp_path / "alias"
        alias.symlink_to(archive, target_is_directory=True)
        archive = alias
    elif location == "case":
        path = fit_case(archive)
        external = tmp_path / "external.json"
        path.rename(external)
        path.symlink_to(external)
    else:
        actual = tmp_path / "destination"
        actual.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        output = alias / "must-not-exist.json"
    with pytest.raises((OSError, ValueError)):
        summary_tool.summarize(archive, output)
    assert not output.exists()


def test_output_overlap_and_existing_destination_preserve_input(archive, summary_tool, tmp_path):
    original = (archive / "manifest.json").read_bytes()
    for output in (archive / "new.json", archive / "manifest.json", archive):
        with pytest.raises(ValueError, match="outside"):
            summary_tool.summarize(archive, output)
    output = tmp_path / "existing.json"
    output.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="exists"):
        summary_tool.summarize(archive, output)
    assert output.read_bytes() == b"preserve"
    assert (archive / "manifest.json").read_bytes() == original
    assert not (archive / "new.json").exists()


def test_active_writer_is_rejected(archive, summary_tool, tmp_path):
    output = tmp_path / "must-not-exist.json"
    with (archive / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            summary_tool.summarize(archive, output)
    assert not output.exists()


def test_fifo_lock_is_rejected_without_waiting(archive, summary_tool, tmp_path):
    (archive / ".lock").unlink()
    os.mkfifo(archive / ".lock")
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="regular archive lock"):
        summary_tool.summarize(archive, output)
    assert not output.exists()


def test_changed_manifest_after_read_creates_no_output(archive, summary_tool, tmp_path, monkeypatch):
    original = summary_tool.read_file
    manifest_reads = 0

    def changed(parent, name):
        nonlocal manifest_reads
        data = original(parent, name)
        if name == "manifest.json":
            manifest_reads += 1
            if manifest_reads == 2:
                return data + b" "
        return data

    monkeypatch.setattr(summary_tool, "read_file", changed)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="changed"):
        summary_tool.summarize(archive, output)
    assert not output.exists()


def test_failure_of_every_start_is_distinct_from_finite_nonconvergence():
    path = Path(__file__).parents[1] / "scripts/summarize_selection_fits.py"
    spec = importlib.util.spec_from_file_location("fit_counting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failed = {"fit_status": "numerical_failure", "optimizer_success": False,
              "chosen_start": None, "cost": None,
              "starts": [{"finite": False, "success": False, "status": None,
                          "message": "FLOATING POINT ERROR", "solution_opd_nm": None,
                          "cost": None} for _ in range(3)]}
    counts, statuses, issues = module.Counter(), module.Counter(), []
    module.count_fit(failed, counts, statuses, issues, {"case_id": "synthetic"}, 0)
    assert counts["numerical_failure_fit_calls"] == 1
    assert counts["nonfinite_starts"] == 3
    assert counts["finite_unconverged_fit_calls"] == 0
    finite = deepcopy(failed)
    finite.update(fit_status="finite", chosen_start=0, cost=1.0)
    for start in finite["starts"]:
        start.update(finite=True, cost=1.0, solution_opd_nm=[0.0] * 6, status=1, message="ITERATION LIMIT")
    module.count_fit(finite, counts, statuses, issues, {"case_id": "synthetic2"}, 0)
    assert counts["finite_unconverged_fit_calls"] == 1
    assert counts["finite_unconverged_starts"] == 3
